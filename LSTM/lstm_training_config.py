"""
Training-only config. Imports torch — this module must NEVER be imported
by predict.py or anything in the production request path (that's the whole
point of splitting it out from lstm_config.py). Only training/export
scripts should import from here.
"""

import torch

from .lstm_config import CHECKPOINT_DIR

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu" )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 4096
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2

PATIENCE = 10
MIN_DELTA = 1e-5

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False
PREFETCH = 4

GRAD_CLIP = 1.0

USE_AMP = True

BEST_CKPT = CHECKPOINT_DIR / "best_model.pt"
LAST_CKPT = CHECKPOINT_DIR / "last_model.pt"
RESUME_CKPT = CHECKPOINT_DIR / "resume.pt"
