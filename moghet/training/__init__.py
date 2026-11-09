"""
Training package.

- train.py: sequential single-device CV
- train_multi_gpu.py: up to 4 GPUs, one fold per GPU (remaining folds queued)
- fold_train.py: shared single-fold training helpers (DRY)
- verbosity.py: quiet|normal|verbose|debug logging levels
- engine.py / loss.py: train step and Cox loss
"""

from .engine import train_one_epoch, evaluate
from .loss import cox_loss

__all__ = [
    "train_one_epoch",
    "evaluate",
    "cox_loss",
]

