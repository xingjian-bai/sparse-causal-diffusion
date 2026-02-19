"""
Dataset module for video prediction.

Primary dataset: MinecraftDataset (minecraft_dataset.py)

Note: Additional datasets (BAIR, DMLab, UCF101, K600, RE10K) are available
in _archive/far/data/ for reference and can be restored if needed.
"""
import importlib
from copy import deepcopy
from os import path as osp

from scd.utils.misc import scandir
from scd.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

# Automatically scan and import dataset modules for registry
# Scans all files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'scd.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    return dataset
