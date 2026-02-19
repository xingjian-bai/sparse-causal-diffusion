"""
Trainer module for SCD video models.

Key Trainers:
- trainer_scd.py: SCDTrainer for training the decoupled encoder-decoder model
- trainer_dcae.py: DCAETrainer for training the video autoencoder (archived)

The SCDTrainer handles the training loop including:
- Forward pass through encoder and decoder
- Loss computation and gradient updates
- EMA model updates
- Validation/sampling during training
"""
import importlib
from os import path as osp

from scd.utils.misc import scandir
from scd.utils.registry import TRAINER_REGISTRY

__all__ = ['build_trainer']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
trainer_folder = osp.dirname(osp.abspath(__file__))
trainer_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(trainer_folder)
    if v.startswith('trainer_')
]
# import all the model modules
_trainer_modules = [
    importlib.import_module(f'scd.trainers.{file_name}')
    for file_name in trainer_filenames
]


def build_trainer(trainer_type):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    trainer = TRAINER_REGISTRY.get(trainer_type)
    return trainer
