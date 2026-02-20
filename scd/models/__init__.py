import importlib
from os import path as osp

from scd.utils.misc import scandir
from scd.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import *_model.py files for registry
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
_model_modules = [
    importlib.import_module(f'scd.models.{file_name}')
    for file_name in model_filenames
]


def build_model(model_type):
    model = MODEL_REGISTRY.get(model_type)
    return model
