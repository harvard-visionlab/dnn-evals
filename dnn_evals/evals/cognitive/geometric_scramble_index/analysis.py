import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from collections import defaultdict
from sklearn.utils import resample

from pdb import set_trace

__all__ = ['main_analysis', 'compute_geom_scram_index']

@torch.no_grad()
def main_analysis(model, dataloader, config, wandb=False, device=None):
    ''''''
    
    crop_size = config['crop_size']
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # if dataset is an imagenet subset (e.g., imagenetted), compute accuracy for only the subset too
    used_labels = list(np.unique([sample[1] for sample in dataloader.dataset.samples]))
    is_subset = len(used_labels) < max(used_labels)
    
    # run analysis
    model.eval()
    results = defaultdict(list)
    image_index = -1
    for batch_num,(imgs,targets,paths,hash_ids,params) in enumerate(progress_bar(dataloader)):
        batch_size, C, H, W = imgs[0].shape
        num_augs = len(imgs)
        imgs = torch.cat(imgs).to(device)
        # reorder (bs*num_augs) x C x H x W
        imgs = imgs.reshape(num_augs, batch_size, C, H, W).permute(1,0,2,3,4).reshape(-1, C, H, W)
        
        # get model output
        outputs = model(imgs)
        
        # reshape bs x num_augs x C x H x W
        outputs = outputs.reshape(batch_size, num_augs, *outputs.shape[1:])
        for batch_idx,output in enumerate(outputs):
            image_index += 1
            label = targets[batch_idx]
            path = paths[batch_idx]
            
            accuracy = (output.argmax(dim=1) == label).cpu().float()

            # accuracy considering only the subset of labels used
            if is_subset: 
                output_subset = output[:, torch.tensor(used_labels).to(device)]
                subset_label = used_labels.index(label.item())
                accuracy_subset = (output_subset.argmax(dim=1) == subset_label).cpu().float()

            for aug_idx in range(num_augs):  
                # undo annoying batching of params
                p = {k:v[batch_idx].item() if not isinstance (v[batch_idx], str) else v[batch_idx] 
                     for k,v in params[aug_idx].items()}
                hash_id = hash_ids[aug_idx][batch_idx]
                patch_size = crop_size/p['block_dim'] if p['block_dim']>0 else crop_size
                
                correct = accuracy[aug_idx].item()
                if is_subset: correct_subset = accuracy_subset[aug_idx].item()

                if not wandb: 
                    for k,v in config.items(): results[k].append(v)
                results['image_index'].append(image_index)
                results['path'].append(path)
                results['item_num'].append(aug_idx)
                results['item_hash_id'].append(hash_id)
                for k,v in p.items(): results[k].append(v)
                results['patch_size'].append(patch_size)
                results['correct'].append(correct)
                if is_subset: results['correct_subset'].append(correct_subset)
                
    df = pd.DataFrame(results)
    if 'shuffle_seed' in df: df['shuffle_seed'] = df['shuffle_seed'].astype(int)
    if 'rotation_seed' in df: df['rotation_seed'] = df['rotation_seed'].astype(int)
    df['shuffle'] = df['shuffle'].astype(bool)
    df['rotate'] = df['rotate'].astype(bool)

    return df

def bootstrap_CI(data, n_resamples=1000, seed=42):
    
    arr = np.array(data)

    # Number of samples to draw (same as original size)
    n_samples = arr.shape[0]

    # Create a random number generator with a seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate the resampling indices
    indices = rng.integers(0, n_samples, size=(n_samples, n_resamples))

    # Use the indices to generate the resampled arrays
    # This will be a 78500 x 1000 array where each column is a resample of the original data
    resampled_data = arr[indices]

    # Compute the 95% confidence interval
    bs_means = resampled_data.mean(axis=0)
    bs_mean = bs_means.mean()
    bs_lower = np.percentile(bs_means, 2.5)
    bs_upper = np.percentile(bs_means, 97.5)
    
    return bs_mean, bs_lower, bs_upper

def compute_summary(df, n_resamples=1000, seed=42):
    
    # adding text columns back
    agg = {    
        'item_type': 'first',
        'transform': 'first',
        'dx': 'first',
        'dy': 'first',
        'angle': 'first',
        'scale': 'first',
        'block_dim': 'first',
        'patch_size': 'first',
        'shuffle': 'first',
        'rotate': 'first',
        'correct': ['mean'],    
    }
    if 'correct_subset' in df:
        agg['correct_subset'] = ['mean']
    summary = df.groupby('item_num').agg(agg).reset_index()
    new_index = ['_'.join(tup) if tup[1] in ['mean', 'std'] else tup[0] for tup in summary.columns]
    summary.columns = new_index
    
    # compute boostrap ci
    bs = defaultdict(list)
    for item_num in summary.item_num.unique():
        subset = df[df.item_num == item_num]
        bs_mean, bs_lower, bs_upper = bootstrap_CI(subset.correct, n_resamples=n_resamples, seed=seed)
        bs['item_num'].append(item_num)
        bs['correct_lower_ci'].append(bs_lower)
        bs['correct_upper_ci'].append(bs_upper)

        if 'correct_subset' in df:
            bs_mean, bs_lower, bs_upper = bootstrap_CI(subset.correct_subset, n_resamples=n_resamples, seed=seed)
            bs['correct_subset_lower_ci'].append(bs_lower)
            bs['correct_subset_upper_ci'].append(bs_upper)

    bs = pd.DataFrame(bs)
    summary = pd.merge(summary, bs, on='item_num', how='right')
    
    return summary

def compute_geom_scram_index(df, metric):
    intact_acc = df[df['transform'] == 'none'][metric].mean()
    
    # geometric tolerance is the average over translation, rotation, and scale scores:    
    translation_acc = df[df['transform'] == 'translate'][metric].mean()
    scale_acc = df[df['transform'] == 'scale'][metric].mean()
    rotation_acc = df[df['transform'] == 'rotate'][metric].mean()
    geom_tolerance = (translation_acc + rotation_acc + scale_acc)/3
    
    # scramble tolerance is the area-under-the-curve (AUC) across the full range of scrambles
    subset = df[df['transform']=='scramble']
    xvals = subset.patch_size.unique()
    yvals = [subset[subset.patch_size==v][metric].mean() for v in xvals]
    total_area = np.abs(np.trapz(yvals, xvals))  # Calculate area using trapezoidal rule
    x_range = xvals.max() - xvals.min()
    normalized_auc = (total_area / x_range)
    gsi = (geom_tolerance - normalized_auc)
    
    results = dict(
        intact_acc=intact_acc,
        translation_acc=translation_acc,
        scale_acc=scale_acc,
        rotation_acc=rotation_acc,
        geom_tolerance=geom_tolerance,
        scramble_auc=normalized_auc,
        geometric_scramble_index=gsi
    )
    
    return results