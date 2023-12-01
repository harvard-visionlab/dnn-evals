'''
    https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main
    https://github.com/modestyachts/ImageNetV2
'''

import os
from natsort import natsorted
from pathlib import Path
from .folder import ImageFolderIndex, default_loader
from .storage import download_if_needed, download_file

__all__ = ['imagenetV2']

public_urls = {
    "top-images": (
        'https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-top-images.tar.gz',
        'd9d6134f'
    ),
    "threshold0.7": (
        'https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-threshold0.7.tar.gz',
        '3dc78c9f',
    ),
    "matched-frequency": (
        'https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-matched-frequency.tar.gz',
        '5fbc2174'
    )
}

s3_urls = {
    "top-images": "s3://visionlab-datasets/imagenetV2/imagenetv2-top-images-format-val-d9d6134f.tar.gz",
    "threshold0.7": "s3://visionlab-datasets/imagenetV2/imagenetv2-threshold0.7-format-val-3dc78c9f.tar.gz",
    "matched-frequency": "s3://visionlab-datasets/imagenetV2/imagenetv2-matched-frequency-format-val-5fbc2174.tar.gz"
}

splits = list(s3_urls.keys())

def imagenetV2(split, transform=None, **kwargs):
    url = s3_urls[split]
    # data are decompressed to a folder matching filename, e.g., imagenetv2-top-images-format-val-d9d6134f
    data_dir = download_if_needed(url) 
    # but then we have to find the archive content within that folder (e.g., imagenetv2-top-images-format-val)
    root_dir = find_folder(data_dir, '-val') # will return first subfoldr that ends with -val
    
    # use custom dataset that ensures label corresponds to parent folder name:
    dataset = ImagenetV2Dataset(root_dir, transform=transform, **kwargs)
    
    # make sure we have the right number
    assert len(dataset)==10_000, f"Oops, expected 10,000 images, got {len(dataset)}"
    
    # make sure foldername and label match
    for fname,label in dataset.imgs:
        parent_folder_name = Path(fname).parent.name
        assert parent_folder_name == f"{label}", f"Parent folder {parent_folder_name} doesn't match label {label}"
    
    return dataset

def find_folder(data_dir, endswith):
    for root, dirs, files in os.walk(data_dir):
        for dirname in dirs:
            if dirname.endswith(endswith):
                return os.path.join(root, dirname)
    return None  # Return None if not found

class ImagenetV2Dataset(ImageFolderIndex):
    '''
        Ensure that the label corresponds to the parent folder name: int(parent_folder_name).
    '''
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super(ImagenetV2Dataset, self).__init__(root, transform=transform, 
                                                           target_transform=target_transform, 
                                                           loader=loader, is_valid_file=is_valid_file)
        
        get_class_from_path = lambda f: Path(f).parent.name
        class_to_label = lambda cls_name: int(cls_name)
    
        # get unique classes
        self.classes = natsorted(set([get_class_from_path(path) for path,_ in self.imgs]))
        
        # map class to label
        self.class_to_idx = {cls_name: class_to_label(cls_name) for cls_name in self.classes}
        
        # update images/samples
        self.imgs = [(path,self.class_to_idx[get_class_from_path(path)]) for path,_ in self.imgs]
        self.samples = self.imgs
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        label_check = int(Path(path).parent.name)
        assert target == label_check, f"oops, target={target}, label_check={label_check}"
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
    
