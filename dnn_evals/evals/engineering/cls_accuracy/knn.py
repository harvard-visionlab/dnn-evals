import torch
import torch.nn.functional as F
from collections import defaultdict
from fastprogress import progress_bar
from pdb import set_trace

from ...utils.feature_extractor import FeatureExtractor
from ...utils.prototypes import PrototypeActivationMeter

@torch.no_grad()        
def run_kNN(model, train_loader, test_loader, layer_names, num_classes=1000, 
            K=200, sigma=.07, num_chunks=10, out_device=None):
    '''
        we compute the full testFeatures, testLabels,
        
        then we iterate over the training set in batches, accumulating `num_chunks` (should
        be `num_batches`, but keeping the naming the same as run_kNN for api consistency).
        
        Finally we have the paiwise similarity between each val and each train image.
    '''
    
    if isinstance(layer_names, str):
        layer_names = [layer_names]
        
    if out_device is None:
        out_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print("==> extracting test features...")
    testFeatures, testLabels, indexes = get_features(model, test_loader, layer_names,
                                                     out_device=out_device)    
    
    print("==> extracting/comparing to train features...")
    topk_distances = defaultdict(lambda: torch.tensor([], device=out_device))
    trainLabels = defaultdict(lambda: torch.tensor([], device=out_device, dtype=torch.int64))
    trainIndexes = defaultdict(lambda: torch.tensor([], device=out_device, dtype=torch.int64))
    prototype_meters = defaultdict(PrototypeActivationMeter)
    
    generator = gen_features(model, train_loader, layer_names, num_batches=num_chunks,
                             out_device=out_device)
    for batch_num, (trn_feat, trn_labels, trn_indexes) in enumerate(generator):  
        
        for layer_name in layer_names:
            prototype_meters[layer_name].update(trn_labels.cpu(), trn_feat[layer_name].cpu())                                          
            # compute distances between testFeatures and current train features
            d = torch.mm(testFeatures[layer_name], 
                         trn_feat[layer_name].T).to(out_device)
            
            # append these distances to running topk_distances
            topk_distances[layer_name] = torch.cat([topk_distances[layer_name], d], dim=1)
            
            # reshape train_labels (numTestImgs x numTrainImagesThisBatch)
            candidate_labels = trn_labels.view(1,-1).expand(len(testLabels), -1)
            # concat with retained trainLabels
            trainLabels[layer_name] = torch.cat([trainLabels[layer_name], 
                                                 candidate_labels], dim=1)
        
            # reshape train_indexes (numTestImgs x numTrainImagesThisBatch)
            candidate_indexes = trn_indexes.view(1,-1).expand(len(testLabels), -1)
            # concat with retained trainIndexes
            trainIndexes[layer_name] = torch.cat([trainIndexes[layer_name], 
                                                  candidate_indexes], dim=1)
        
            # keep the top K distances and labels  
            yd, yi = topk_distances[layer_name].topk(K, dim=1, largest=True, sorted=True)
            topk_distances[layer_name] = torch.gather(topk_distances[layer_name], 1, yi)
            trainLabels[layer_name] = torch.gather(trainLabels[layer_name], 1, yi)
            trainIndexes[layer_name] = torch.gather(trainIndexes[layer_name], 1, yi)
    
    # After iterating through the full training set, we have retained
    # the topk_distances, topk_labels, topk_indexes for the topk most
    # similar training items for each individual test item
    # generate weighted predictions.
    
    # Finally, we compute the predicted class through a similarity-weighted
    # voting amongst the topK separately for each layer
    print("==> computing top1,top5 accurcy: ...")
    top1_acc = dict()
    top5_acc = dict()
    prototypes = dict()
    
    for layer_name in progress_bar(layer_names):
        distances = topk_distances[layer_name]
        train_labels = trainLabels[layer_name]
        
        pred, top1, top5 = compute_knn_accuracy(distances, 
                                                train_labels, 
                                                testLabels, 
                                                num_classes=num_classes, 
                                                sigma=sigma)
        
        top1 = top1.float().sum(dim=1).mean().item() * 100
        top5 = top5.float().sum(dim=1).mean().item() * 100
        
        print(f"kNN accuracy {layer_name}: top1={top1}, top5={top5}")
        top1_acc[layer_name] = top1
        top5_acc[layer_name] = top5
        
        prototypes[layer_name] = prototype_meters[layer_name].state_dict()
        
    return top1_acc, top5_acc, prototypes, testFeatures, testLabels

def compute_knn_accuracy(distances, train_labels, test_labels, num_classes, sigma):
    """
    Computes the k-NN classification accuracy.

    :param distances: Tensor of distances between test and training features (num_Test x topK_Train)
    :param train_labels: Labels corresponding to the training data (num_Test x topK_Train)
    :param test_labels: Labels corresponding to the test data.
    :param num_classes: Total number of classes.
    :param sigma: Scaling parameter for distance transformation.
    :return: Tuple of (predictions, top1 accuracy, top5 accuracy)
    """
    num_test_images, K = train_labels.shape
    retrieval_one_hot = torch.zeros(K, num_classes).to('cpu')
    retrieval_one_hot.resize_(num_test_images * K, num_classes).zero_()
    retrieval_one_hot.scatter_(1, train_labels.view(-1, 1).cpu(), 1)
    yd_transform = distances.clone().div_(sigma).exp_().cpu()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(num_test_images, -1 , num_classes), 
                                yd_transform.view(num_test_images, -1, 1)), 1)
    _, predictions = probs.sort(1, True)

    # Find which predictions match the target
    correct = predictions.eq(test_labels.view(-1,1).cpu())

    total = correct.size(0)
    top1 = correct.narrow(1,0,1)
    
    # Handle the case where the number of predictions is less than 5
    top_k = min(5, predictions.size(1))
    top5 = correct.narrow(1,0,top_k)
    
    return predictions, top1, top5

@torch.no_grad()
def get_features(model, dataloader, layer_names, device=None, out_device=None):    
    
    if isinstance(layer_names, str):
        layer_names = [layer_names]
        
    if device is None:
        device = next(model.parameters()).device
        
    if out_device is None:
        out_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    model.to(device)
    features,labels,indexes = defaultdict(list),[],[]
    
    with FeatureExtractor(model, layer_names, device=out_device) as extractor:
        for imgs,targs,idxs in progress_bar(dataloader):
            feat = extractor(imgs.to(device, non_blocking=True))
            
            for layer_name,X in feat.items():
                X = X.flatten(start_dim=1)
                X = F.normalize(X, dim=1)
                features[layer_name].append(X.to(out_device))
            labels.append(targs.to(out_device))
            indexes.append(idxs.to(out_device))
    
    for layer_name in layer_names:
        features[layer_name] = torch.cat(features[layer_name])
    labels = torch.cat(labels)
    indexes = torch.cat(indexes)
    
    return features, labels, indexes

@torch.no_grad()
def gen_features(model, dataloader, layer_names, num_batches=10, device=None, out_device=None):    
    
    if isinstance(layer_names, str):
        layer_names = [layer_names]
        
    if device is None:
        device = next(model.parameters()).device
        
    if out_device is None:
        out_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
    model.eval()
    model.to(device)
    features,targets,indexes = defaultdict(list), [], []
    batch_count=0
    for batch_num,(imgs,targs,idxs) in enumerate(progress_bar(dataloader)):
        batch_count+=1
        targets.append(targs.to(out_device))
        indexes.append(idxs.to(out_device))
        imgs = imgs.to(device, non_blocking=True)      
        with FeatureExtractor(model, layer_names, device=out_device) as extractor:
            feat = extractor(imgs)
        
        # normalize and aggregate features
        for layer_name,X in feat.items():
            X = X.flatten(start_dim=1) # flatten from dim1 onward
            X = F.normalize(X, dim=1)  # normalize across features
            features[layer_name].append(X.to(out_device))
        
        if batch_count==num_batches:
            #print(f"==> batch_num={batch_num}, batch_count={batch_count}")
            for layer_name in layer_names:
                features[layer_name] = torch.cat(features[layer_name])
            targets = torch.cat(targets)
            indexes = torch.cat(indexes)   
            yield features, targets, indexes
            features,targets,indexes = defaultdict(list), [], []
            batch_count=0
    
    # yield any remaining features
    if len(features[layer_name]) > 0:
        #print("==> wait, there's more!")
        for layer_name in layer_names:
            features[layer_name] = torch.cat(features[layer_name])
        targets = torch.cat(targets)
        indexes = torch.cat(indexes)

        yield features, targets, indexes