import importlib
from os import path as osp

from scd.utils.misc import scandir
from scd.utils.registry import TRAINER_REGISTRY

__all__ = ['build_trainer']

# automatically scan and import trainer_*.py files for registry
trainer_folder = osp.dirname(osp.abspath(__file__))
trainer_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(trainer_folder)
    if v.startswith('trainer_')
]
_trainer_modules = [
    importlib.import_module(f'scd.trainers.{file_name}')
    for file_name in trainer_filenames
]


def build_trainer(trainer_type):
    trainer = TRAINER_REGISTRY.get(trainer_type)
    return trainer
