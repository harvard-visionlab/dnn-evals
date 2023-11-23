import os
from .folder import ImagenetteRemappedLabels, ImageFolderIndex
from .storage import download_if_needed, download_file

__all__ = ['imagenette2_s320', 'imagenette2_s320_remap1k']

def find_folder(data_dir, split):
    for root, dirs, files in os.walk(data_dir):
        if split in dirs:
            return os.path.join(root, split)
    return None  # Return None if not found

def imagenette2_s320(split, transform=None, **kwargs):
    data_dir = download_if_needed('s3://visionlab-datasets/imagenette/imagenette2-320-569b4497.tgz')
    root_dir = find_folder(data_dir, split)
    dataset = ImageFolderIndex(root_dir, transform=transform, **kwargs)
        
    if split=="val":
        assert len(dataset)==3925, f"Oops, expected 3925 images, got {len(dataset)}"
    elif split=="train":
        assert len(dataset)==9469, f"Oops, expected 9469 images, got {len(dataset)}"
        
    return dataset

def imagenette2_s320_remap1k(split, transform=None, **kwargs):
    data_dir = download_if_needed('s3://visionlab-datasets/imagenette/imagenette2-320-569b4497.tgz')
    root_dir = find_folder(data_dir, split)
    dataset = ImagenetteRemappedLabels(root_dir, transform=transform, **kwargs)
        
    if split=="val":
        assert len(dataset)==3925, f"Oops, expected 3925 images, got {len(dataset)}"
    elif split=="train":
        assert len(dataset)==9469, f"Oops, expected 9469 images, got {len(dataset)}"
        
    return dataset