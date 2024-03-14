import os
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision import models, datasets, transforms
from pathlib import Path
from PIL import Image

from . import transforms_functional as FT

class ImageFolderIndex(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        filename = os.path.join(Path(path).parent.name, Path(path).name)
        
        return sample, target, index, filename
    
class RandomTranslate(object):
    def __init__(self, max_dx, max_dy, seed=None):
        """
        Args:
            max_dx (int): Maximum pixel shift in the x-direction.
            max_dy (int): Maximum pixel shift in the y-direction.
            seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Use NumPy's RNG with the provided seed

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be translated.
        
        Returns:
            PIL Image: Randomly translated image.
        """
        dx = self.rng.integers(-self.max_dx, self.max_dx + 1)  # +1 because the upper bound is exclusive
        dy = self.rng.integers(-self.max_dy, self.max_dy + 1)
        print(dx,dy)
        return F.affine(img, angle=0, translate=(dx, dy), scale=1, shear=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(max_dx={0}, max_dy={1}'.format(self.max_dx, self.max_dy)
        format_string += ', seed={0})'.format(self.seed)  # Extract seed for representation
        return format_string

class FixedRandomTranslate(object):
    def __init__(self, dx, dy, seed=None):
        """
        Args:
            dx (int): Fixed pixel shift in the x-direction.
            dy (int): Fixed pixel shift in the y-direction.
            seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.dx = dx
        self.dy = dy
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Use NumPy's RNG with the provided seed
        self.last_dx = None
        self.last_dy = None
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be translated.
        
        Returns:
            PIL Image: Deterministically translated image.
        """
        # Generate translation offsets
        offsets = self.rng.choice([-1, 1], size=2)  # Choose either -1 or 1 for x and y
        dx = offsets[0] * self.dx
        dy = offsets[1] * self.dy
        self.last_dx = dx
        self.last_dy = dy
        return F.affine(img, angle=0, translate=(dx, dy), scale=1, shear=0)

    def __repr__(self):
        return f'{self.__class__.__name__}(dx={self.dx}, dy={self.dy}, seed={self.seed})'
    
class RandomRotate90(object):
    def __init__(self, seed=None):
        """
        Args:
            seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Use NumPy's RNG with the provided seed
        self.last_angle = None
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        
        Returns:
            PIL Image: randomly rotated either 90, 180, or 270 deg
        """
        # Generate translation offsets
        angles = self.rng.choice([90, 180, 270], size=1)  # Choose either -1 or 1 for x and y
        angle = int(angles[0])
        self.last_angle = angle
        return F.affine(img, angle=angle, translate=(0, 0), scale=1, shear=0)

    def __repr__(self):
        return f'{self.__class__.__name__}(seed={self.seed})'
    
class RandomScale(object):
    def __init__(self, min_scale=.4, seed=None):
        """
        Args:
            seed (int, optional): Seed for random number generator for reproducibility.
        """
        self.min_scale = min_scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Use NumPy's RNG with the provided seed
        self.last_scale = None
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        
        Returns:
            PIL Image: randomly rotated either 90, 180, or 270 deg
        """
        # Generate translation offsets
        scales = self.rng.choice([1.2, 1.4, 1.6, 1.8], size=1)  # Choose either -1 or 1 for x and y
        scale = scales[0]
        self.last_scale = scale
        return F.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)

    def __repr__(self):
        return f'{self.__class__.__name__}(min_scale={self.min_scale}, seed={self.seed})' 
    
class GridMask(object):
    def __init__(self, block_dim, line_width=2, grid_color=[0, 0, 0], background_color=[1, 1, 1]):
        self.block_dim = block_dim
        self.line_width = line_width
        self.grid_color = grid_color
        self.background_color = background_color
    
    def __call__(self, image):
        """
        Apply the grid mask to the input image.
        
        :param image: Tensor image of size (C x H x W).
        :return: Tensor image with grid mask applied.
        """
        C, H, W = image.shape
        mask = FT.grid_mask((H, W), self.block_dim, self.line_width, self.grid_color, self.background_color)
        # Assuming the image is already normalized, we might need to adjust mask tensor accordingly
        masked_image = image * mask
        return masked_image    