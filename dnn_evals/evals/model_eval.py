import os
import copy
from torch.utils.data import DataLoader

from .. import datasets

class ModelEval(object):
    def __init__(self, dataset_name, dataset=None, **kwargs):
        self.dataset_name = dataset_name
        
        if dataset_name in datasets.__dict__:
            self.dataset = datasets.__dict__[dataset](**kwargs)
        elif dataset is not None:
            self.dataset = dataset
        else:
            raise Exception("`dataset_name` must be a registered dataset, otherwise you must provoide a `dataset`")
            
    def get_dataloader(self, transform, batch_size=256, num_workers=len(os.sched_getaffinity(0)), shuffle=False):
        dataset = copy.deepcopy(self.dataset)
        dataset.transform = transform
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=shuffle, pin_memory=True)
        return dataloader
    
    def compute_metrics(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `compute_metrics`.")
        
    def plot_results(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `plot_results`.")
    
    def run(self, model, transform):
        dataloader = self.get_dataloader(transform)
        raise NotImplementedError("Subclasses of ModelAssay should implement `run`.")
            