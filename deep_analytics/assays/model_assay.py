import copy
import multiprocessing
from torch.utils.data import DataLoader

import ..datasets

class ModelAssay(object):
    def __init__(self, dataset, **kwargs):
        self.dataset = datasets.__dict__[dataset](**kwargs)
    
    def get_dataloader(self, transform, batch_size=256, num_workers=multiprocessing.cpu_count()):
        dataset = copy.deepcopy(self.dataset)
        dataset.transform = transform
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=False, pin_memory=True)
        return dataloader
    
    def plot_results(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `plot_results`.")
    
    def run(self, model, transform):
        dataloader = self.get_dataloader(transform)
        raise NotImplementedError("Subclasses of ModelAssay should implement `run`.")
            