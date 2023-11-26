import torch
import numpy as np
from numpy.random import RandomState
from functools import reduce
from fastprogress import progress_bar

__all__ = ['generate_test_data', 'bootstrap_multi_dim']
           
def generate_test_data(num_subjects=3, num_conds=2, num_items=5):
    D = np.zeros((num_subjects, num_conds, num_items))

    # Populate the array with unique values for each subject and each item
    for subject in range(num_subjects):
        for cond in range(num_conds):
            # Offset each condition by 100
            D[subject, cond, :] = np.arange(subject * 100, subject * 100 + num_items) + (cond * 100)
    
    return D

def bootstrap_single_dim(D, dim, n_bootstrap=1000, seed=None):
    if dim < 0:
        dim = D.ndim + dim
        
    num_items = D.shape[dim]
    
    # Initialize the random number generator
    rng = RandomState(seed)
    
    # Generate bootstrap samples
    samples = rng.choice(num_items, size=(n_bootstrap, num_items), replace=True)

    # Select the samples along the specified dimension
    bootstrap_samples = np.take(D, samples, axis=dim)
    
    # Move the bootstrap dimension to the first position
    bootstrap_samples = np.moveaxis(bootstrap_samples, dim, 0)
    
    return bootstrap_samples
           
def bootstrap_multi_dim(D, dims, n_bootstrap=10000, seed=None, vectorized=False):
    """
    Perform bootstrap resampling across specified dimensions of a multi-dimensional array.

    Args:
    D (numpy.ndarray): The input array from which to sample.
    dims (int or tuple of ints): The dimensions over which to perform bootstrapping.
    n_bootstrap (int): The number of bootstrap samples to generate.
    seed (int): Random seed for reproducibility.
    vectorized (bool): Whether to use vectorized indexing (flatten data, use flat_indices)
    
    Returns:
    numpy.ndarray: An array of bootstrapped samples. The shape of the array is 
                   (n_bootstrap, *D.shape), where the first dimension corresponds
                   to the bootstrap samples, and the remaining dimensions correspond
                   to the dimensions of the original array.

    The function generates random indices for the specified dimensions (dims) and 
    uses the original indices for the other dimensions. These indices are expanded 
    and scaled by the array's strides to calculate the flat indices, which are then 
    used to index into a flattened version of the original array. The resulting 
    samples are reshaped to form an array that retains the structure of the original 
    array while incorporating the bootstrap dimension.
    """    
    if isinstance(dims, int): dims = (dims,)
    dims = tuple([dim + D.ndim if dim < 0 else dim for dim in dims])
    
    rng = RandomState(seed)

    # Generate random indices for each specified dimension, repeat original indices for other dimensions
    indices = [rng.choice(D.shape[dim], size=(n_bootstrap, D.shape[dim]), replace=True) 
               if dim in dims else np.arange(D.shape[dim])[None,:].repeat(n_bootstrap, 0)
               for dim in range(D.ndim)]

    if vectorized:
        bootstrap_samples = vectorized_indexing(D, indices)
    else:
        bootstrap_samples = loop_indexing(D, indices)
    
    return bootstrap_samples

def loop_indexing(D, indices):
    
    n_bootstrap = indices[0].shape[0]
    
    # Initialize the bootstrap sample array
    new_shape = [n_bootstrap] + list(D.shape)
    bootstrap_samples = np.empty(new_shape, dtype=D.dtype)

    # Iterate over the indices for each bootstrap sample
    for i, curr_indices in enumerate(progress_bar(zip(*indices), total=n_bootstrap)):
        # Use advanced indexing to select the sample
        bootstrap_samples[i] = D[np.ix_(*curr_indices)]
    
    # Now, bootstrap_samples contains the bootstrapped samples    
    return bootstrap_samples

def vectorized_indexing(D, indices):
    n_bootstrap = indices[0].shape[0]
    
    # Calculate the strides for each dimension in D
    strides = np.array(D.strides) // D.itemsize

    # Flatten the array D
    D_flat = D.reshape(-1)
    
    # lambda function to expand dimensions on indices so they broadcast correctly when summed
    # e.g., if D is 5x2x100, n_bootstrap=1000, then reshape indices dim=0 to be 1000x5x1x1, etc.
    adjusted_axes = lambda dim: [i+1 for i in range(D.ndim) if i != dim]
    
    # Expand Dims and Multiply each set of indices with the corresponding stride before summing
    scaled_indices = [np.ascontiguousarray(np.expand_dims(idx, axis=adjusted_axes(dim)) * stride).astype(np.int32)
                      for dim, (idx, stride) in enumerate(zip(indices, strides))]

    # Perform the direct summation using reduce (np.add performs the correct broadcasting)
    flat_indices = reduce(np.add, scaled_indices)
    
    # Use the flat indices to access the data values
    bootstrap_samples_flat = D_flat[flat_indices]
    
    # reshape 
    bootstrap_samples = bootstrap_samples_flat.reshape(n_bootstrap, *D.shape)
    
    return bootstrap_samples

