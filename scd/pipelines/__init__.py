import importlib
from os import path as osp

from scd.utils.misc import scandir
from scd.utils.registry import PIPELINE_REGISTRY

__all__ = ['build_pipeline']

# automatically scan and import pipeline_*.py files for registry
pipeline_folder = osp.dirname(osp.abspath(__file__))
pipeline_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(pipeline_folder)
    if v.startswith('pipeline_')
]
_pipeline_modules = [
    importlib.import_module(f'scd.pipelines.{file_name}')
    for file_name in pipeline_filenames
]


def build_pipeline(pipeline_type):
    pipeline = PIPELINE_REGISTRY.get(pipeline_type)
    return pipeline
