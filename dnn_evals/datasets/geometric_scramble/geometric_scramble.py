import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import datasets, transforms

from .transforms_functional import block_scramble
from .hashid import hash_pil_image

__all__ = ['GeomScrambleImageFolder']

class GeomScrambleImageFolder(datasets.ImageFolder):
    def __init__(self, root, img_size=256, crop_size=224, grayscale=False,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 block_shuffle=True, block_rotate=False, block_rotation_angles=[90,180,270],
                 shuffle_seed=None, rotation_seed=None):
        """
        Generates a set copies of an object that are translated, rotated, rescaled, and scrambled.
        
        The varitions of translation, rotation, and rescaling weren't chosen to "match" the magnitude
        of the transformations in any intuitive sense. They were set to span the fullest range that
        works for ImageNet images given standard img_size (256px shortest edge), and center cropping (224px).
        
        4 Translations, fixed at +/- offset = (img_size - crop_size)//2 to avoid edge effects.
            (dx,dy): (-offset,-offset), (-offset,offset), (offset,offset), or (offset,-offset)
        
        3 rotations, 90, 180, 270 (limited to work with square images).
        
        5 scales, one zoom out (.9), limited to work with 256/224 crop ratio,
        and four zoom ins (1.2, 1.4, 1.6, 1.8).
        
        """
        # Initialize the parent class
        super().__init__(root)
                
        self.img_size = img_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std    
        self.shuffle_seed = shuffle_seed
        self.rotation_seed = rotation_seed        
        self.grayscale = grayscale
        self.block_rotation_angles = block_rotation_angles
        
        self.resize = transforms.Resize(img_size)
        self.crop = transforms.CenterCrop(crop_size)
        self.to_grayscale = transforms.Grayscale(3)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        
        self.inv_transform = transforms.Compose([
            transforms.Normalize(
                mean= [-m/s for m, s in zip(mean, std)],
                std= [1/s for s in std]
            ),
            transforms.ToPILImage(),
        ])
        
        max_dx = int( (img_size-crop_size) // 2)
        max_dy = int( (img_size-crop_size) // 2)
        
        shuffle = int(block_shuffle)
        rotate = int(block_rotate)
        self.transform_params = [
            dict(item_type='intact', transform='none', dx=0, dy=0, angle=0, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='translate', dx=-max_dx, dy=-max_dy, angle=0, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='translate', dx=-max_dx, dy=max_dy, angle=0, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='translate', dx=max_dx, dy=max_dy, angle=0, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='translate', dx=max_dx, dy=-max_dy, angle=0, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='rotate', dx=0, dy=0, angle=90, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='rotate', dx=0, dy=0, angle=180, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='rotate', dx=0, dy=0, angle=270, scale=1.0, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='scale', dx=0, dy=0, angle=0, scale=0.9, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='scale', dx=0, dy=0, angle=0, scale=1.2, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='scale', dx=0, dy=0, angle=0, scale=1.4, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='scale', dx=0, dy=0, angle=0, scale=1.6, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='geometric', transform='scale', dx=0, dy=0, angle=0, scale=1.8, block_dim=0, shuffle=False, rotate=False),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=2, shuffle=shuffle, rotate=rotate),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=4, shuffle=shuffle, rotate=rotate),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=8, shuffle=shuffle, rotate=rotate),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=16, shuffle=shuffle, rotate=rotate),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=32, shuffle=shuffle, rotate=rotate),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=56, shuffle=shuffle, rotate=rotate),
            dict(item_type='scramble', transform='scramble', dx=0, dy=0, angle=0, scale=1.0, block_dim=112, shuffle=shuffle, rotate=rotate),
        ]
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # for reproducible outcomes for this exact image, set random seed based on stored_seed + index
        if self.shuffle_seed is not None:
            shuffle_rng = np.random.default_rng(self.shuffle_seed+index)
        else:
            shuffle_rng = None
        
        if self.rotation_seed is not None:
            rotation_rng = np.random.default_rng(self.rotation_seed+index)
        else:
            rotation_rng = None
        
        imgs = []
        hash_ids = []
        for params in self.transform_params:
            
            img = sample.copy()
            if self.grayscale:
                img = self.to_grayscale(img)
                
            img = self.resize(img)
            
            # If translating, translate before cropping
            if (params['dx'] != 0) or (params['dy'] != 0):
                img = F.affine(img, angle=0, translate=(params['dx'], params['dy']), scale=1.0, shear=0)
            
            # If zooming out (rescaling < 1.0), rescale before cropping
            if params['scale'] < 1.0:
                img = F.affine(img, angle=0, translate=(0,0), scale=params['scale'], shear=0)            
            
            # crop before rotating or zooming in
            img = self.crop(img)
            
            # zoom in:
            if params['scale'] > 1.0:
                img = F.affine(img, angle=0, translate=(0,0), scale=params['scale'], shear=0)            
            
            # rotate
            if params['angle'] > 0:
                img = F.affine(img, angle=params['angle'], translate=(0,0), scale=1.0, shear=0)            
            
            # scramble
            if params['block_dim'] > 0:
                sz = (int(self.crop.size[0] / params['block_dim']), int(self.crop.size[1] / params['block_dim']))
                img = block_scramble(img, sz, random_shuffle=params['shuffle'], random_rotate=params['rotate'], 
                                     shuffle_rng=shuffle_rng, rotation_rng=rotation_rng, 
                                     rotation_angles=self.block_rotation_angles)
            
            # compute hash_id before normalization stats,
            hash_id = hash_pil_image(img)
            hash_ids.append(hash_id)
                        
            img = self.normalize(self.to_tensor(img))
            imgs.append(img)
            
        return imgs, target, path, hash_ids, self.transform_params
    
    def __repr__(self):
        return f'{self.__class__.__name__}()' 