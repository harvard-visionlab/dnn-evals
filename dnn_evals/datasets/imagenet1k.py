from .folder import ImageNetIndex
from .storage import download_if_needed, download_file
from .registry import register_dataset 

__all__ = ['imagenet1k_s256']

source = "imagenet1k"
repo = ""
citation = ''''''

@register_dataset(source, repo, citation)
def imagenet1k_s256(split, transform=None, **kwargs):
    if split=="val":
        data_dir = download_if_needed('s3://visionlab-datasets/imagenet1k-256/in1k-val-d74759d1.tar.gz')
        cached_file = download_file('s3://visionlab-datasets/imagenet1k-256/ILSVRC2012_devkit_t12.tar.gz',
                                    cache_dir=data_dir, check_hash=False)
        dataset = ImageNetIndex(data_dir, split="val", transform=transform, **kwargs)
        assert len(dataset)==50_000, f"Oops, expected 50,000 images, got {len(dataset)}"
    elif split=="train":
        data_dir = download_if_needed('s3://visionlab-datasets/imagenet1k-256/in1k-train-XYZ.tar.gz')
        cached_file = download_file('s3://visionlab-datasets/imagenet1k-256/ILSVRC2012_devkit_t12.tar.gz',
                                    cache_dir=data_dir, check_hash=False)
        dataset = ImageNetIndex(data_dir, split="train", transform=transform, **kwargs)
        # assert len(dataset)==50_000, f"Oops, expected 50,000 images, got {len(dataset)}"
        
    return dataset