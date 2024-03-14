import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from fastprogress import master_bar, progress_bar
from torchvision.datasets.folder import default_loader

__all__ = ['PrototypeActivationMeter', 'Stats', 'ClassActivationMeter']

class Stats:
    """
    The Stats class maintains running statistics for a set of activations.
    It keeps track of the count, sum, sum of squares, mean, standard deviation, 
    and covariance matrix of the activations.

    The covariance is computed as a sample covariance matrix using the formula:
    Cov(X) = E[XX^T] - E[X]E[X]^T
    where E[XX^T] is the expectation of the outer products and E[X]E[X]^T is 
    the outer product of the means.

    Methods:
    - __init__: Initializes the statistics.
    - update: Updates the statistics with a new activation tensor.
    - state_dict: Returns the current state of the statistics as a dictionary.
    """

    def __init__(self):
        self.count = None
        self.sum = None
        self.sumsq = None
        self.mean = None
        self.std = None
        self.sum_outer = None
        self.covariance = None

    def update(self, activations):
        '''assumes dim0 is the batch dimension'''
        batch_size = activations.size(0)

        if self.count is None:
            # Initialize statistics based on the shape of the activation tensor.
            self.count = 0
            self.sum = torch.zeros_like(activations[0])
            self.sumsq = torch.zeros_like(activations[0])
            self.mean = torch.zeros_like(activations[0])
            self.std = torch.zeros_like(activations[0])
            self.sum_outer = torch.zeros(activations.size(1), activations.size(1))
            self.covariance = torch.zeros_like(self.sum_outer)

        # Update running stats for batch
        self.count += batch_size
        self.sum += torch.sum(activations, dim=0)
        self.sumsq += torch.sum(activations ** 2, dim=0)
        
        # Use torch.einsum for batch outer product
        self.sum_outer += torch.einsum('bi,bj->ij', activations, activations)

        self.mean = self.sum / self.count
        self.std = torch.sqrt(self.sumsq / self.count - self.mean ** 2)
        self.std[torch.isnan(self.std)] = 0
        self.covariance = self.sum_outer / self.count - torch.outer(self.mean, self.mean)

    def state_dict(self):
        return {
            'count': self.count,
            'sum': self.sum,
            'sumsq': self.sumsq,
            'mean': self.mean,
            'std': self.std,
            'covariance': self.covariance,
        }

class PrototypeActivationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.prototypes = defaultdict(Stats)

    def update(self, indices, activations):
        unique_indices = indices.unique()

        for index in unique_indices:
            mask = indices == index
            batch_activations = activations[mask]
            self.prototypes[index.item()].update(batch_activations)

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.prototypes.items()}

    def __str__(self):
        fmtstr = 'PrototypeActivationMeter:\n'
        for i in range(self.num_classes):
            fmtstr += f'Class {i}: '
            for j in range(self.num_units):
                fmtstr += f'Unit {j}: Mean={self.means[i,j]:.4f}, Std={self.std[i,j]:.4f}; '
            fmtstr += '\n'
        return fmtstr
    
class ClassActivationMeter(object):
    def __init__(self, num_classes, num_units):
        self.num_classes = num_classes
        self.num_units = num_units
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.sums = torch.zeros(self.num_classes, self.num_units)
        self.sumsq = torch.zeros(self.num_classes, self.num_units)
        self.sum_outer = torch.zeros(self.num_classes, self.num_units, self.num_units)
        self.means = torch.zeros(self.num_classes, self.num_units)
        self.std = torch.zeros(self.num_classes, self.num_units)
        self.covariance = torch.zeros(self.num_classes, self.num_units, self.num_units)

    def update(self, activations, labels):
        for i in range(activations.shape[0]):
            class_idx = labels[i].item()
            self.counts[class_idx] += 1
            activation = activations[i]
            self.sums[class_idx] += activation
            self.sumsq[class_idx] += activation**2
            self.sum_outer[class_idx] += torch.outer(activation, activation)
            self.means[class_idx] = self.sums[class_idx] / self.counts[class_idx]
            self.std[class_idx] = torch.sqrt(self.sumsq[class_idx] / self.counts[class_idx] - self.means[class_idx]**2)
            self.std[class_idx][torch.isnan(self.std[class_idx])] = 0
            self.covariance[class_idx] = self.sum_outer[class_idx] / self.counts[class_idx] - torch.outer(self.means[class_idx], self.means[class_idx])
            
    def __str__(self):
        fmtstr = 'ClassActivationMeter:\n'
        for i in range(self.num_classes):
            fmtstr += f'Class {i}: '
            for j in range(self.num_units):
                fmtstr += f'Unit {j}: Mean={self.means[i,j]:.4f}, Std={self.std[i,j]:.4f}; '
            fmtstr += '\n'
        return fmtstr    