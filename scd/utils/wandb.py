from dataclasses import dataclass
from typing import List, Optional


@dataclass
class wandb_config:
    project: str = 'decoupled-video-model'
    entity: Optional[str] = None  # Set via WANDB_ENTITY env var or override here
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    name: Optional[str] = None
    dir: Optional[str] = None
    resume: str = 'allow'
    id: Optional[str] = None