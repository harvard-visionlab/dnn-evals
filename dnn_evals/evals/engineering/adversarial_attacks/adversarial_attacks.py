import os
import io
import gc
from collections import defaultdict
from contextlib import redirect_stdout
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchattacks
import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar

from ...model_eval import ModelEval
from ...utils.bootstrap import bootstrap_multi_dim
from ...utils.stats import AccumMetric
from .metrics import *

from pdb import set_trace

from types import SimpleNamespace

AttackTypes = SimpleNamespace(
    FGSM='FGSM'
)
    
default_epsilon = [0., .001, .002, .003, 0.006, 0.012, 0.018, 0.024, 0.03 , 0.036, 0.042, 0.048, 0.054, 0.06 ]

__all__ = ['AdversarialAttacks', 'AttackTypes']

class AdversarialAttacks(ModelEval):
    
    datasets = dict(
        imagenette2=('imagenette2_s320_remap1k', 'val'),
        imagenet1k=('imagenet1k_s256', 'val')
    )
    
    def compute_metrics(self, df, data_column, n_bootstrap=1000, bootstrap_dims=(-1), seed=None):
        
        results = dataframe_to_array(df[df.image_set=='adversarial'], data_column)
        D = results['D']
        epsilons = np.array(results['epsilons'])
        
        samples = bootstrap_multi_dim(D, dims=bootstrap_dims, n_bootstrap=n_bootstrap)

        metrics = dict([
            ('acc_by_eps', AccumMetric(partial(compute_acc_by_eps, epsilons=epsilons))),
            ('norm_acc_by_eps', AccumMetric(partial(compute_normalized_acc_by_eps, epsilons=epsilons))),
            ('norm_auc', AccumMetric(partial(compute_normalized_auc, epsilons=epsilons))),
            ('norm_acc', AccumMetric(partial(compute_normalized_acc, epsilons=epsilons))),
            ('norm_adv_acc', AccumMetric(partial(compute_normalized_adv_acc, epsilons=epsilons))),
            ('weighted_norm_adv_acc', AccumMetric(partial(compute_weighted_adv_acc, epsilons=epsilons, normalize=True))),
            ('thresh_half_minmax', AccumMetric(partial(compute_thresh_half_minmax, epsilons=epsilons, normalize=True)))
        ])

        for sample in progress_bar(samples):
            for metric in metrics.values(): 
                metric(sample)
        
        return metrics
    
    def plot_results(self, df, x="epsilon", y="correct1", hue="image_set", figsize=(6, 4), ylim=None, title=None):
        import seaborn as sns
        
        # Set the figure size here
        plt.figure(figsize=figsize)

        ax = sns.lineplot(data=df, x=x, y=y, hue=hue)

        if ylim is None:
            ylim = ax.get_ylim()
            ax.set_ylim([-.05, 1])
        else:
            ax.set_ylim(ylim)

        ax.set_xlabel('epsilon', fontsize=18, labelpad=12)
        ax.set_ylabel('top1 accuracy' if y == "correct1" else 'top5 accuracy', fontsize=18, labelpad=12)

        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        if title is not None: 
            ax.set_title(title, fontsize=22)
        else:
            ax.set_title(f'adversarial robustness {df.iloc[0].model_name}', fontsize=18)

        # Move the legend outside of the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

        # Remove the top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        return ax
    
    def run(self, load_model, transform, attack, epsilon_values=default_epsilon, atk_args={}):
        self.dataloader = self.get_dataloader(transform)        
        
        # get mean/std from final transform (assumes final transform is a Normalization)
        mean = transform.transforms[-1].mean
        std = transform.transforms[-1].std
        
        mb = master_bar(epsilon_values)
        df = None
        for epsilon in mb:
            model = load_model()

            atk = torchattacks.__dict__[attack](model, eps=epsilon)
            atk.set_normalization_used(mean=mean, std=std)
            df_ = validate_attack(model, self.dataloader, atk, mb=mb, **atk_args)
            df_['model_name'] = model.model_name
            df_['dataset'] = self.dataset_name
            df_['attack'] = attack
            df_['epsilon'] = epsilon
            df = pd.concat([df, df_])

            # Clear the cache
            del model
            del atk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        return df
            
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        corrects = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            corrects.append(correct[:k].any(dim=0).reshape(-1).float())
        return pred, *corrects, *res

def validate_attack(model, val_loader, atk, num_classes = 1000,
                    print_freq=100, mb=None, store_outputs=False):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction='none')
    filepaths = [(os.path.sep).join(f.split(os.path.sep)[-2:]) for f,_ in val_loader.dataset.imgs]

    def run_validate(loader):
        results = defaultdict(list)
        count = 0
        for i, batch in enumerate(progress_bar(loader, parent=mb)):
            batch_size = batch[0].shape[0]
            images = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            index = batch[2].tolist()
            filenames = [filepaths[idx] for idx in index]

            adv_images = atk(images, target)

            with torch.no_grad():
                output_orig = model(images)
                output_atk = model(adv_images)

            #print( (target.cpu()==output_orig.cpu().argmax(dim=1)).float().mean() )
            #print( (target.cpu()==output_atk.cpu().argmax(dim=1)).float().mean() )

            loss_orig = criterion(output_orig, target)
            loss_atk = criterion(output_atk, target)

            # measure accuracy and record loss
            preds_orig, correct1_orig, correct5_orig, _, _ = accuracy(output_orig, target, topk=(1, 5))
            preds_atk, correct1_atk, correct5_atk, _, _ = accuracy(output_atk, target, topk=(1, 5))

            results['image_set'] += ['original'] * batch_size
            results['index'] += index
            results['filenames'] += filenames
            results['label'] += target.tolist()
            results['loss'] += loss_orig.tolist()
            results['pred_label'] += preds_orig[0].tolist()
            results['correct1'] += correct1_orig.tolist()
            results['correct5'] += correct5_orig.tolist()

            results['image_set'] += ['adversarial'] * batch_size
            results['index'] += index
            results['filenames'] += filenames
            results['label'] += target.tolist()
            results['loss'] += loss_atk.tolist()
            results['pred_label'] += preds_atk[0].tolist()
            results['correct1'] += correct1_atk.tolist()
            results['correct5'] += correct5_atk.tolist()

        df = pd.DataFrame(results)

        return df

    df = run_validate(val_loader)

    return df