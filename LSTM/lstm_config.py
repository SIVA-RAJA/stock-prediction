from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset"
PARQUET_DIR = DATA_DIR / "parquet"
SCALER_DIR = DATA_DIR / "scalers"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "runs"

for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu" )

SEQ_LEN = 60
PRED_HORIZON = 1
STRIDE = 1

TICKER_EMB_DIM = 16
MARKET_EMB_DIM = 4
REGION_EMB_DIM = 8

LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
BIDIRECTIONAL = True

ATTN_HIDDEN = 128

HEAD_HIDDEN = 128
HEAD_DROPOUT = 0.3

LAMBDA_PRICE = 1.0
LAMBDA_DIR = 0.5

BATCH_SIZE = 512
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SCHEDULER_TO = 10
SCHEDULER_T_MULT = 2

PATIENCE = 15
MIN_DELTA = 1e-5

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

NUM_WORKERS = 4
PIN_MEMORY = True if DEVICE == "cuda" else False

GRAD_CLIP = 1.0

BEST_CKPT = CHECKPOINT_DIR / "best_model.pt"
LAST_CKPT = CHECKPOINT_DIR / "last_model.pt"
