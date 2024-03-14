from collections import defaultdict

__all__ = ['list_datasources', 'list_datasets', 'get_dataset_info', 'load_dataset']

_dataset_registry = defaultdict(list)  # mapping of model names to entrypoint fns
_dataset_functions = dict()

def register_dataset(datasource, repo, citation):
    def inner_decorator(fn):
        # add entries to registry dict/sets
        dataset_name = fn.__name__
        _dataset_registry[datasource].append(dict(
            dataset_name=dataset_name,
            repo=repo,
            citation=citation,
        ))
        _dataset_functions[dataset_name] = fn
        return fn
    return inner_decorator

def list_datasources():
    """ Return list of available datas sources
    """
    return list(_dataset_registry.keys())

def list_datasets(datasource):
    """ Return list of available datasets, sorted alphabetically
    """
    return [dset['dataset_name'] for dset in _dataset_registry[datasource]]

def get_dataset_info(datasource, dataset_name):
    for dataset in _dataset_registry[datasource]:
        if dataset['dataset_name'] == dataset_name:
            return dataset
        
def load_dataset(dataset_name, **kwargs):
    dataset = _dataset_functions[dataset_name](**kwargs)
    return dataset