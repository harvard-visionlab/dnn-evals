from .geometric_scramble import GeomScrambleImageFolder
from .visualization import imshow_scrambled
from ..registry import list_datasets, load_dataset

__all__ = ['geometric_scramble_classification', 'imshow_scrambled']

def geometric_scramble_classification(dataset_name, split='val', root_dir=None, labelmap=None, block_shuffle=True, block_rotate=False, grayscale=True, 
                 img_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 shuffle_seed=12345, rotate_seed=123456, entity=None, project=None):
    
    tmp_dataset = None
    if root_dir is None and dataset_name in list_datasets():
        tmp_dataset = load_dataset(dataset_name, split=split)
        root_dir = tmp_dataset.root
        if labelmap is None and hasattr(tmp_dataset, 'labelmap'):
            labelmap = tmp_dataset.labelmap
        
    dataset = GeomScrambleImageFolder(root_dir, 
                                      img_size=img_size,
                                      crop_size=crop_size,
                                      mean=mean,
                                      std=std,
                                      grayscale=grayscale, 
                                      block_shuffle=block_shuffle, 
                                      block_rotate=block_rotate,
                                      shuffle_seed=shuffle_seed, 
                                      rotation_seed=rotate_seed)
    
    if labelmap is not None:
        dataset.imgs = [(fname, labelmap[label]) for fname,label in dataset.imgs]
        dataset.samples = dataset.imgs
        
    return dataset