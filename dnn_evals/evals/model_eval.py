import os
import copy
from torchvision import transforms
from torch.utils.data import DataLoader

from .. import datasets

class ModelEval(object):
    
    datasets = dict(
        imagenette2=('imagenette2_s320_remap1k', 'val'),
    )
    
    def __init__(self, dataset_name, dataset=None, **kwargs):
        
        self.dataset_name = dataset_name
        
        if dataset_name in self.datasets:
            name, split = self.datasets[dataset_name]
            self.dataset = datasets.__dict__[name](split=split, **kwargs)
        elif dataset is not None:
            self.dataset = dataset
        else:
            dataset_names = list(self.datasets.keys())
            raise Exception(f"`dataset_name` must be a supported dataset {dataset_names}, otherwise you must provoide a `dataset`")
  
    @property
    def config(self):
        '''run config object that can be passed to data storage methods'''
        return dict(
            analysis=self.__class__.__name__,
            #some_property=self.some_property,
        )
    
    def _run_config(self, transform):
        '''run-specific config that includes transform info'''
        config = self.config
        # override the default transforms:
        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                config['img_size'] = t.size
            if isinstance(t, transforms.CenterCrop):
                config['crop_size'] = t.size[0]
            if isinstance(t, transforms.Normalize):
                config['mean'] = t.mean
                config['std'] = t.std
        
        return config
    
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
    
    def run_analysis(self, model, transform):
        dataloader = self.get_dataloader(transform)
        raise NotImplementedError("Subclasses of ModelAssay should implement `run`.")             