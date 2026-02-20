import importlib
from copy import deepcopy
from os import path as osp

from scd.utils.misc import scandir
from scd.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

# automatically scan and import *_dataset.py files for registry
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'scd.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    return dataset
