import os
import io
import gc
from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchattacks
from fastprogress import master_bar, progress_bar

from ..model_assay import ModelAssay

class AdversarialAttacks(ModelAssay):
    def plot_results(self, df):
        raise NotImplementedError("Subclasses of ModelAssay should implement `plot_results`.")
    
    def run(self, model, transform):
        dataloader = self.get_dataloader(transform)
        raise NotImplementedError("Subclasses of ModelAssay should implement `run`.")

