import os
import io
import gc
import copy
from collections import defaultdict
from contextlib import redirect_stdout
from functools import partial
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchattacks
import matplotlib.pyplot as plt
import seaborn as sns
from fastprogress import master_bar, progress_bar

from ...utils._wandb import suppress_all_outputs, get_table_by_name
from ....datasets import geometric_scramble_classification, imshow_scrambled
from ...model_eval_wandb import ModelEvalWandb

from .analysis import main_analysis, compute_summary, compute_geom_scram_index
from .plotting import plot_results

from pdb import set_trace

from types import SimpleNamespace

ScrambleTypes = SimpleNamespace(
    BLOCK_SHUFFLE='BLOCK_SHUFFLE',
    BLOCK_ROTATE='BLOCK_ROTATE',
    BLOCK_SHUFFLE_ROTATE='BLOCK_SHUFFLE_ROTATE',
    BLOCK_SHUFFLE_GRID='BLOCK_SHUFFLE_GRID',
    BLOCK_SMOOTHED='BLOCK_SMOOTHED',
    STYLE_TRANSFER='STYLE_TRANSFER',
    DIFFEOMORPHIC='DIFFEOMORPHIC',
    TEXFORMS='TEXFORMS',    
    VISUAL_ANAGRAMS='VISUAL_ANAGRAMS',
)
    
__all__ = [
    'GeometricScrambleIndex_Blocks', 
    'GeometricScrambleIndex_BlockShuffle',
    'GeometricScrambleIndex_BlockRotate',
    'GeometricScrambleIndex_BlockShuffleRotate',
    # 'GeometricScrambleIndex_BlockShuffleGrids',
    # 'GeometricScrambleIndex_BlockShuffleSmoothed',
    # 'GeometricScrambleIndex_StyleTransfer',
    # 'GeometricScrambleIndex_Diffeomorphic',
    # 'GeometricScrambleIndex_Texforms',
    # 'GeometricScrambleIndex_Anabrams',
    'ScrambleTypes'
]

class GeometricScrambleIndex_Blocks(ModelEvalWandb):
    
    datasets = dict(
        imagenette2=('imagenette2_s320_remap1k', 'val'),
    )
    
    def __init__(self, dataset_name, root_dir=None, block_shuffle=True, block_rotate=False, grayscale=True, 
                 img_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 shuffle_seed=12345, rotate_seed=123456, entity=None, project=None):
 
        # these parameters can be over-ridden by values in transforms
        self.img_size = img_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        
        self.grayscale = grayscale
        self.block_shuffle = block_shuffle
        self.block_rotate = block_rotate
        self.shuffle_seed = shuffle_seed
        self.rotate_seed = rotate_seed
        self.entity = entity
        self.project = project
        
        self.init_dataset(dataset_name, root_dir)
        
    def init_dataset(self, dataset_name, root_dir=None):
        kwargs = dict(block_shuffle=self.block_shuffle, block_rotate=self.block_rotate, grayscale=self.grayscale,
                      img_size=self.img_size, crop_size=self.crop_size, mean=self.mean, std=self.std,
                      shuffle_seed=self.shuffle_seed, rotate_seed=self.rotate_seed)
        
        self.dataset_name = dataset_name
        if root_dir is None and dataset_name in self.datasets:
            name, split = self.datasets[dataset_name]
            self.dataset = geometric_scramble_classification(name, split=split, **kwargs)
        elif root_dir is not None:
            self.dataset = geometric_scramble_classification(dataset_name, root_dir=root_dir, **kwargs)
        else:
            dataset_names = list(self.datasets.keys())
            raise Exception(f"`dataset_name` must be a supported dataset {dataset_names}, otherwise you must provoide a `root_dir` to be used with the ImageFolder dataset class")
            
    @property
    def config(self):
        return dict(
            analysis=self.__class__.__name__,
            dataset_name=self.dataset_name,
            img_size=self.img_size,
            crop_size=self.crop_size,
            mean=self.mean,
            std=self.std,
            grayscale=self.grayscale,
            block_shuffle=self.block_shuffle,
            block_rotate=self.block_rotate,
            shuffle_seed=self.shuffle_seed, 
            rotation_seed=self.rotate_seed
        )
    
    def get_dataloader(self, dataset, transform, batch_size=1, num_workers=len(os.sched_getaffinity(0))):
        dataset = copy.deepcopy(self.dataset)
        
        # override the default transforms:
        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                dataset.resize = t
            if isinstance(t, transforms.CenterCrop):
                dataset.crop = t
            if isinstance(t, transforms.Normalize):
                dataset.normalize = t
                
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=False, pin_memory=True)
        return dataloader
    
    def run_analysis(self, model, dataloader, wandb, config):
        df = main_analysis(model, dataloader, config, wandb=wandb)
        metrics = self.compute_metrics(df, config)
        results = {
            "rawdata": df,
            **metrics
        }
        return results
    
    def compute_metrics(self, df, config):
        crop_size = config['crop_size']
        
        # aggregate over images
        summary = compute_summary(df, n_resamples=1000, seed=42)
        
        # compute geometric_scramble_index for each accuracy score
        results = []
        gsi_correct = compute_geom_scram_index(summary, 'correct_mean')
        results.append({ **{"score": 'correct'}, **gsi_correct})

        if 'correct_subset_mean' in summary:
            gsi_correct_subset = compute_geom_scram_index(summary, 'correct_subset_mean')
            results.append({ **{"score": 'correct_subset'}, **gsi_correct_subset})
        gsi_df = pd.DataFrame(results)

        return dict(summary=summary, geometric_scramble_index=gsi_df)
    
    def show_images(self, index, num_cols=4):
        imgs,target,path,hash_ids,params = self.dataset[index]
        img_list = [self.dataset.inv_transform(img) for img in imgs]
        return imshow_scrambled(img_list, params, num_cols=num_cols)

    def plot_results(self, results, subset_accuracy=True, chance=None):
        return plot_results(results, subset_accuracy=subset_accuracy, chance=chance)
        
    def scoreboard(self, filters={}, figsize=(14,6)):
        with suppress_all_outputs():
            api, entity, project = self.get_wandb_api(entity=None, project=None)
            runs = api.runs(entity + "/" + project, filters={**self.wandb_filters, **filters})

            res = defaultdict(list)
            for run in progress_bar(runs):
                geometric_scramble_index = get_table_by_name(run, 'geometric_scramble_index')
                for row_num,row in geometric_scramble_index.iterrows():
                    for k,v in run.config.items(): res[k].append(v)
                    for k,v in row.items(): res[k].append(v)
            res = pd.DataFrame(res)
        
        # quick plot        
        fig, ax = plt.subplots(figsize=figsize)
        subset = res.copy()
        subset = subset.sort_values(by='geometric_scramble_index', ascending=True)
        ax = sns.barplot(data=subset, x="arch", y="geometric_scramble_index", hue='score', ax=ax)
        ax.set_ylim([0,1.0])
        ax.set_ylabel('Geometric Scramble Index\n(holistic shape strength)', fontsize=18, labelpad=10)
        sns.despine()

        return res, ax
        
    def _fetch_results_from_wandb(self, filters, entity=None, project=None):        
        run = self.get_latest_run_by_filters(filters)
        if run is not None:
            results = dict(
                rawdata=get_table_by_name(run, 'rawdata'),
                summary=get_table_by_name(run, 'summary'), 
                geometric_scramble_index=get_table_by_name(run, 'geometric_scramble_index')
            )
            return results
        
        return None
    
    def __call__(self, model, transform, meta={}, wandb=False, hash_id=None, entity=None, project=None, recompute=False):
        # identify this run: add transform params to the run-specific config, append hash_id
        run_config = {**self._run_config(transform), **{"hash_id": hash_id}}
        
        # check for existing results on wandb:
        results = None
        if wandb and recompute==False:
            print("==> checking wandb for existing results...")    
            filters = {f"config.{k}":v for k,v in run_config.items()} # if anything changed it's a new run
            results = self._fetch_results_from_wandb(filters, entity, project)
  
        # no wandb results, run the analysis
        if results is None:
            dataloader = self.get_dataloader(self.dataset, transform)
            results = self.run_analysis(model, dataloader, wandb, {**run_config, **meta})
            if wandb: self.log_tables(results, {**run_config, **meta}, entity, project)
            
        return results, run_config
            
class GeometricScrambleIndex_BlockShuffle(GeometricScrambleIndex_Blocks):

    def __init__(self, dataset_name, root_dir=None, entity=None, project=None):
        
        # these parameters can be over-ridden by values in transforms
        self.img_size = 256
        self.crop_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # these are fixed for the benchmark; for variations use the general GeometricScrambleIndex class
        self.grayscale = True
        self.block_shuffle = True
        self.block_rotate = False
        self.shuffle_seed = 12345
        self.rotate_seed = 123456
        self.entity = entity
        self.project = project
        
        self.init_dataset(dataset_name, root_dir)
        
class GeometricScrambleIndex_BlockRotate(GeometricScrambleIndex_Blocks):

    def __init__(self, dataset_name, root_dir=None, entity=None, project=None):
        
        # these parameters can be over-ridden by values in transforms
        self.img_size = 256
        self.crop_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # these are fixed for the benchmark; for variations use the general GeometricScrambleIndex class
        self.grayscale = True
        self.block_shuffle = False
        self.block_rotate = True
        self.shuffle_seed = 12345
        self.rotate_seed = 123456
        self.entity = entity
        self.project = project    
        
        self.init_dataset(dataset_name, root_dir)
        
class GeometricScrambleIndex_BlockShuffleRotate(GeometricScrambleIndex_Blocks):

    def __init__(self, dataset_name, root_dir=None, entity=None, project=None):
        
        # these parameters can be over-ridden by values in transforms
        self.img_size = 256
        self.crop_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # these are fixed for the benchmark; for variations use the general GeometricScrambleIndex class
        self.grayscale = True
        self.block_shuffle = True
        self.block_rotate = True
        self.shuffle_seed = 12345
        self.rotate_seed = 123456
        self.entity = entity
        self.project = project        
        
        self.init_dataset(dataset_name, root_dir)