from torchvision.datasets import ImageFolder, ImageNet
from torchvision.datasets.folder import default_loader

class ImageFolderIndex(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

class ImageNetIndex(ImageNet):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index  
    
class ImagenetteRemappedLabels(ImageFolderIndex):
    labelmap = {0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=None):
        super(ImagenetteRemappedLabels, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)

        # update labels to use new labels
        self.imgs = [(fname,self.labelmap[label]) for fname,label in self.imgs]
        self.samples = self.imgs
   