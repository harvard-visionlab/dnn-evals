import itertools
from pdb import set_trace

import numpy as np
from fastprogress import progress_bar

from ...utils.stats import estimate_thresh_crossing

__all__ = ['dataframe_to_array', 'compute_acc_by_eps', 'compute_normalized_acc_by_eps', 
           'compute_normalized_auc',  'compute_normalized_acc', 'compute_normalized_adv_acc',           
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

def compute_acc_by_eps(data, epsilons, dim=-1):
    acc_by_eps = data.mean(axis=dim)
    return acc_by_eps

def compute_normalized_acc_by_eps(data, epsilons, dim=-1):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    assert len(epsilons)==acc_by_eps.shape[-1], f"oops, final dimension to be size of epsilons ({len(epsilons)}), got {acc_by_eps.shape}"
    
    clean_accuracy = acc_by_eps[..., 0:1]
    normed_acc = acc_by_eps / clean_accuracy
    return normed_acc

def compute_normalized_auc(data, epsilons, dim=-1):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    assert len(epsilons)==acc_by_eps.shape[-1], f"oops, final dimension to be size of epsilons ({len(epsilons)}), got {acc_by_eps.shape}"
    
    clean_accuracy = acc_by_eps[..., 0:1]
    normed_acc = acc_by_eps / clean_accuracy
    normalized_auc = np.sum(normed_acc, axis=-1)
    return normalized_auc

def compute_normalized_acc(data, epsilons, dim=-1):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    assert len(epsilons)==acc_by_eps.shape[-1], f"oops, final dimension to be size of epsilons ({len(epsilons)}), got {acc_by_eps.shape}"
    clean_accuracy = acc_by_eps[..., 0:1]
    normed_acc = acc_by_eps / clean_accuracy
    normalized_acc = np.mean(normed_acc, axis=-1)
    return normalized_acc

def compute_normalized_adv_acc(data, epsilons, dim=-1):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    assert len(epsilons)==acc_by_eps.shape[-1], f"oops, final dimension to be size of epsilons ({len(epsilons)}), got {acc_by_eps.shape}"
    clean_accuracy = acc_by_eps[..., 0:1]
    adv_accuracies = acc_by_eps[...,1:]
    normed_adv_acc = adv_accuracies / clean_accuracy
    normed_adv_acc = np.mean(normed_adv_acc, axis=-1)
    return normed_adv_acc

def compute_weighted_adv_acc(data, epsilons, dim=-1, normalize=True):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)
    
    # last dim is now expected to be the epsilon dimension
    assert len(epsilons)==acc_by_eps.shape[-1], f"oops, final dimension to be size of epsilons ({len(epsilons)}), got {acc_by_eps.shape}"
    
    clean_accuracy = acc_by_eps[...,0:1]
    adv_accuracies = acc_by_eps[...,1:]
    if normalize:
        adv_accuracies = adv_accuracies / clean_accuracy
    
    weighted_adv_acc = np.average(adv_accuracies, weights=1/epsilons[1:], axis=-1)
    return weighted_adv_acc

def compute_thresh_half_minmax(data, epsilons, dim=-1, normalize=True):
    assert epsilons[0]==0, f"oops, expected first epsilon to be zero, got {epsilons[0]}"
    acc_by_eps = data.mean(axis=dim)     # average over item
    assert len(epsilons)==acc_by_eps.shape[-1], f"oops, final dimension to be size of epsilons ({len(epsilons)}), got {acc_by_eps.shape}"
    clean_accuracy = acc_by_eps[...,0:1] # last dim is now epsilon
    normed_acc = acc_by_eps / clean_accuracy
    thresh_half_minmax, _ = estimate_all_thresh_crossings(epsilons, normed_acc)
    
    return thresh_half_minmax

def estimate_all_thresh_crossings(xs, ys):
    # Prepare an array to store the results
    result_shape = ys.shape[:-1]
    esp_at_thresh = np.empty(result_shape, dtype=object)
    thresholds = np.empty(result_shape, dtype=object)
    
    ranges = [range(dim) for dim in result_shape]
    indices = list(itertools.product(*ranges))
    
    for idx in indices:
        current_ys = ys[idx]
        current_threshold = current_ys.min() + (current_ys.max() - current_ys.min()) / 2
        epsilon_at_threshold, _ = estimate_thresh_crossing(xs, current_ys, current_threshold)
        esp_at_thresh[idx] = epsilon_at_threshold
        thresholds[idx] = current_threshold

    return esp_at_thresh, thresholds