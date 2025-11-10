"""
Adaptive Learning Module

This module provides adaptive and continuous learning capabilities
for financial forecasting models.

Components:
- ModelVersionManager: Version control for models
- PerformanceTracker: Track model accuracy over time
- OnlineLearner: Incremental model updates
- RollingWindowTrainer: Retrain on recent data windows
- AdaptiveEnsemble: Dynamic model weighting
- RetrainingScheduler: Automated retraining triggers
"""

from .model_versioning import ModelVersionManager
from .performance_tracker import PerformanceTracker
from .online_learner import OnlineLearner
from .rolling_window_trainer import RollingWindowTrainer
from .ensemble_rebalancer import AdaptiveEnsemble
from .scheduler import RetrainingScheduler

__all__ = [
    'ModelVersionManager',
    'PerformanceTracker',
    'OnlineLearner',
    'RollingWindowTrainer',
    'AdaptiveEnsemble',
    'RetrainingScheduler'
]

__version__ = '1.0.0'
