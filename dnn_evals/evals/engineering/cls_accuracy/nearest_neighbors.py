import os
import io
import gc
from collections import defaultdict
from contextlib import redirect_stdout
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchattacks
import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar
from torch.cuda.amp import autocast

from ..model_assay import ModelAssay
from ...utils.bootstrap import bootstrap_multi_dim
from ...utils.stats import AccumMetric
from ...utils.feature_extractor import FeatureExtractor
from .metrics import *

from pdb import set_trace

__all__ = ['ClassificationNearestNeighbors', 'ClassificationNearestPrototype']

datasets = dict(
    imagenette2=('imagenette2_s320_remap1k', 'val'),
    imagenet1k=('imagenet1k_s256', 'val'),
    imagenetV2_top_images=('imagenetV2', 'top-images'),
    imagenetV2_threshold07=('imagenetV2', 'threshold0.7'),
    imagenetV2_matched_frequency=('imagenetV2', 'matched-frequency')
)

class ClassificationNearestNeighbors(ModelAssay):
    
    datasets=datasets

    def compute_metrics(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `compute_metrics`.")
        
    def plot_results(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `plot_results`.")
    
    def __call__(self, model_or_model_loader, transform):
        self.dataloader = self.get_dataloader(transform)        
        
        if isinstance(model_or_model_loader, nn.Module):
            model = model_or_model_loader
        else:
            model = model_or_model_loader()

        df = validate(model, self.dataloader)
        df['model_name'] = model.__dict__.get("model_name", model.__class__.__name__)
        df['dataset'] = self.dataset_name

        # Clear the cache
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
        return df
    
class ClassificationNearestPrototype(ModelAssay):
    
    datasets=datasets

    def compute_metrics(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `compute_metrics`.")
        
    def plot_results(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `plot_results`.")
    
    def __call__(self, model_or_model_loader, transform):
        self.dataloader = self.get_dataloader(transform)        
        
        if isinstance(model_or_model_loader, nn.Module):
            model = model_or_model_loader
        else:
            model = model_or_model_loader()

        df = validate(model, self.dataloader)
        df['model_name'] = model.__dict__.get("model_name", model.__class__.__name__)
        df['dataset'] = self.dataset_name

        # Clear the cache
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
            
        return df    
