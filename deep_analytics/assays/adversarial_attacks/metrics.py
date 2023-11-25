import numpy as np

__all__ = ['dataframe_to_array', 'compute_normalized_auc',  'compute_normalized_acc',  
           'compute_weighted_adv_acc', 'compute_thresh_half_minmax']

def dataframe_to_array(df, data_column):
    '''
        convert dataframe into a numModels x numImageSets x numEpsilons x numItems array
    '''
    model_names = list(df.model_name.unique())
    image_sets = list(df.image_set.unique())
    epsilons = sorted(list(df.epsilon.unique()))
    item_names = list(df.filenames.unique())
    
    D = np.empty((len(model_names),len(image_sets),len(epsilons),len(item_names)))
    D[:] = np.nan
    for rownum,row in progress_bar(df.iterrows(), total=len(df)):
        model_num = model_names.index(row.model_name)
        imageset_num = image_sets.index(row.image_set)
        epsilon_num = epsilons.index(row.epsilon)
        item_num = item_names.index(row.filenames)
        curr_val = D[model_num, imageset_num, epsilon_num, item_num]
        assert np.isnan(curr_val), f"Oops, expected current value to be nan, got {curr_val}"    
        D[model_num, imageset_num, epsilon_num, item_num] = row[data_column]
    assert np.isnan(D).any() == False, "Oops, expected all values to be filled, found nans"    
    
    return dict(
        D=D,
        dims=['model_name', 'image_set', 'epsilon', 'filename'],
        model_names=model_names,
        image_sets=image_sets,
        epsilons=epsilons,
        item_names=item_names
    )

def compute_normalized_auc(data, epsilons, dim=-1):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    clean_accuracy = acc_by_eps[0]
    normed_acc = acc_by_eps / clean_accuracy
    normalized_auc = np.sum(normed_acc)
    return normalized_auc

def compute_normalized_acc(data, epsilons, dim=-1):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    clean_accuracy = acc_by_eps[0]
    normed_acc = acc_by_eps / clean_accuracy
    normalized_acc = np.mean(normed_acc)
    return normalized_acc

def compute_weighted_adv_acc(data, epsilons, dim=-1, normalize=True):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=-1)
    clean_accuracy = acc_by_eps[0]
    adv_accuracies = acc_by_eps[1:]
    if normalize:
        adv_accuracies = adv_accuracies / clean_accuracy
    weighted_adv_acc = np.average(adv_accuracies, weights=1/epsilons[1:])
    return weighted_adv_acc

def compute_thresh_half_minmax(data, epsilons, dim=-1, normalize=True):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    clean_accuracy = acc_by_eps[0]
    normed_acc = acc_by_eps / clean_accuracy
    half_max = normed_acc.min() + (normed_acc.max() - normed_acc.min())/2
    thresh_half_minmax, _ = estimate_thresh_crossing(epsilons, normed_acc, half_max)
    
    return thresh_half_minmax