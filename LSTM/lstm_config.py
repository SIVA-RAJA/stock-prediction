import torch

from data.config import DRIVE_ROOT, MMAP_DIR


BASE_DIR = DRIVE_ROOT
CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
LOG_DIR = DRIVE_ROOT / "runs"

for d in [CHECKPOINT_DIR, LOG_DIR, MMAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu" )

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


SEQ_LEN = 60
PRED_HORIZON = 1
STRIDE = 1

TICKER_EMB_DIM = 32
MARKET_EMB_DIM = 8
REGION_EMB_DIM = 16
INTERVAL_EMB_DIM = 8

LSTM_HIDDEN = 512
LSTM_LAYERS = 3
LSTM_DROPOUT = 0.3
BIDIRECTIONAL = True

ATTN_HIDDEN = 256

HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3

LAMBDA_PRICE = 1.0
LAMBDA_DIR = 0.5

BATCH_SIZE = 2048
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2

PATIENCE = 15
MIN_DELTA = 1e-5

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

NUM_WORKERS = 2
PIN_MEMORY = True if DEVICE == "cuda" else False
PREFETCH = 4

GRAD_CLIP = 1.0

USE_AMP = True

BEST_CKPT = CHECKPOINT_DIR / "best_model.pt"
LAST_CKPT = CHECKPOINT_DIR / "last_model.pt"
RESUME_CKPT = CHECKPOINT_DIR / "resume.pt"
