from .geometric_scramble import GeomScrambleImageFolder
from .... import datasets

def geometric_scramble_classification(dataset_name, split='val', root_dir=None, block_shuffle=True, block_rotate=False, grayscale=True, 
                 img_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 shuffle_seed=12345, rotate_seed=123456, entity=None, project=None):
    
    if root_dir is None and dataset_name in datasets.__dict__:
        tmp_dataset = datasets.__dict__[dataset_name](split=split)
        root_dir = datasets.root_dir
        
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

    if split=="val":
        assert len(dataset)==3925, f"Oops, expected 3925 images, got {len(dataset)}"
    elif split=="train":
        assert len(dataset)==9469, f"Oops, expected 9469 images, got {len(dataset)}"
        
    return dataset