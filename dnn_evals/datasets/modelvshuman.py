# TODO: replace "get_remote_data_file" with "get_data_from_bucket" so we can use credentials
# from ..remote_data import get_remote_data_file
# from ..tardataset import TarDataset
# from ..registry import register_dataset 

root_url = "https://s3.us-east-1.wasabisys.com/visionlab-datasets/deep-shape-audit/modelvshuman/"
source = "modelvshuman"
repo = "https://github.com/bethgelab/model-vs-human"
citation = '''
@inproceedings{geirhos2021partial,
  title={Partial success in closing the gap between human and machine vision},
  author={Geirhos, Robert and Narayanappa, Kantharaju and Mitzkus, Benjamin and Thieringer, Tizian and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
  booktitle={{Advances in Neural Information Processing Systems 34}},
  year={2021},
}
'''

urls = {
    "colour":  root_url + "colour-d9d5355e0c.tar.gz",
    "cue-conflict": root_url + "cue-conflict-70459b639d.tar.gz",
    "contrast": root_url + "contrast-82c4433e1e.tar.gz",
    "edge": root_url + "edge-ad4735b43c.tar.gz",
    "eidolonI": root_url + "eidolonI-fbc222fd58.tar.gz",
    "eidolonII": root_url + "eidolonII-02ac4ac842.tar.gz",
    "eidolonIII": root_url + "eidolonIII-f0767e9451.tar.gz",
    "false-colour": root_url + "false-colour-529b7532e9.tar.gz",
    "high-pass": root_url + "high-pass-74831c8ae5.tar.gz",
    "low-pass": root_url + "low-pass-aa3eed2b0c.tar.gz",
    "phase-scrambling": root_url + "phase-scrambling-a3c00193d1.tar.gz",
    "power-equalisation": root_url + "power-equalisation-a949fd043a.tar.gz",
    "rotation": root_url + "rotation-de96b4c788.tar.gz",
    "silhouette": root_url + "silhouette-b89b3aa77b.tar.gz",
    "sketch": root_url + "sketch-c78e21b629.tar.gz",
    "stylized": root_url + "stylized-d43a77e774.tar.gz",
}

__all__ = ['ColourDataset', 'ContrastDataset', 'CueConflictDataset', 
           'EdgeDataset', 'EidolonIDataset', 'EidolonIIDataset', 'EidolonIIIDataset',
           'FalseColourDataSet', 'HighPassDataset', 'LowPassDataset',
           'PhaseScramblingDataset', 'PowerEqualisationDataset', 'RotationDataset',
           'SilhouetteDataset', 'SketchDataset', 'StylizedDataset']

@register_dataset(source, repo, citation)
class ColourDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["colour"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)  
        
@register_dataset(source, repo, citation)        
class ContrastDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["contrast"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)  
        
@register_dataset(source, repo, citation)        
class CueConflictDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["cue-conflict"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)        
class EdgeDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["edge"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform) 
        
@register_dataset(source, repo, citation)        
class EidolonIDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["eidolonI"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform) 
        
@register_dataset(source, repo, citation)        
class EidolonIIDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["eidolonII"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform) 
        
@register_dataset(source, repo, citation)        
class EidolonIIIDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["eidolonIII"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)  
        
@register_dataset(source, repo, citation)        
class FalseColourDataSet(TarDataset):
    def __init__(self, transform=None):
        url = urls["false-colour"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)        
class HighPassDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["high-pass"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)        
class LowPassDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["low-pass"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)    
class PhaseScramblingDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["phase-scrambling"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)        
class PowerEqualisationDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["power-equalisation"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)
class RotationDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["rotation"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)  
    
@register_dataset(source, repo, citation)    
class SilhouetteDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["silhouette"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
    
@register_dataset(source, repo, citation)    
class SketchDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["sketch"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)
        
@register_dataset(source, repo, citation)    
class StylizedDataset(TarDataset):
    def __init__(self, transform=None):
        url = urls["stylized"]
        cached_datafile = get_remote_data_file(url)
        super().__init__(cached_datafile, transform=transform)