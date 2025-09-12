"""Models module for Multi-Omics Platform"""

from .lightweight_models import (
    BaseMultiOmicsModel,
    GenomicsModel,
    TranscriptomicsModel,
    ProteomicsModel,
    MultiOmicsFusionModel,
    create_lightweight_model,
    DEFAULT_CONFIG
)

__all__ = [
    'BaseMultiOmicsModel',
    'GenomicsModel', 
    'TranscriptomicsModel',
    'ProteomicsModel',
    'MultiOmicsFusionModel',
    'create_lightweight_model',
    'DEFAULT_CONFIG'
]
