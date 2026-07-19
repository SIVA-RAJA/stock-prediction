from data.config import ARTIFACTS_DIR

CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
MODEL_DIR = ARTIFACTS_DIR / "model"
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
LOG_DIR = ARTIFACTS_DIR / "runs"

for d in [CHECKPOINT_DIR, MODEL_DIR, EVAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CACHE_SIZE = 10

SEQ_LEN = 60
PRED_HORIZON = 1
STRIDE = 1

MARKET_EMB_DIM = 8
REGION_EMB_DIM = 16
INTERVAL_EMB_DIM = 8

LSTM_HIDDEN = 128
LSTM_LAYERS = 3
LSTM_DROPOUT = 0.35
BIDIRECTIONAL = True

HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.35

LAMBDA_ATTN = 0.0

EMB_COLS = ["market_id", "region_id", "interval_id", "ticker_id"]
EXCLUDE_COLS = EMB_COLS + ["datetime", "ticker", "market", "region", "interval"]
