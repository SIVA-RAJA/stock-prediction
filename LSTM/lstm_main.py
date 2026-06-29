"""
Usage:
        python lstm_main.py                              # Train and evaluate the LSTM model
        python lstm_main.py --eval-only                  # Evaluate only
        python lstm_main.py --export                     # Export to ONNX
        python lstm_main.py --rebuild                    # Force rebuild LSTM mmap cache

        tensorboard --logdir runs/                       # Start TensorBoard to visualize training metrics (live)

"""


import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))

import argparse
import logging
from data.config import TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID, INTERVAL_TO_ID
from lstm_config import DEVICE, BEST_CKPT
from lstm_dataset import make_dataloaders
from lstm_model import MarketLSTM
from lstm_trainer import train, load_checkpoint
from lstm_evaluate import evaluate
from lstm_export import export_onnx, verify_onnx


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild mmap cache")
    args = parser.parse_args()


    log.info(f"Loading data")
    train_loader, val_loader, test_loader, num_features = make_dataloaders(force_rebuild=args.rebuild)
    model = MarketLSTM(
        num_features=num_features,
        num_tickers=len(TICKER_TO_ID),
        num_markets=len(MARKET_TO_ID),
        num_regions=len(REGION_TO_ID),
        num_intervals=len(INTERVAL_TO_ID),
    )

    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Features: {num_features}")
    log.info(f"Tickers: {len(TICKER_TO_ID)}| Intervals: {len(INTERVAL_TO_ID)}")

    if args.eval_only:
        if not BEST_CKPT.exists():
            raise FileNotFoundError(f"No checkpoint found at {BEST_CKPT}")
        load_checkpoint(model, path=BEST_CKPT)
        log.info("Loaded best checkpoint for evaluation")
    else:
        log.info("Starting training...")
        model = train(model, train_loader, val_loader, run_name="run")

    log.info("Evaluating on test set...")
    metrices = evaluate(
        model, test_loader,
        run_name="eval",
    )

    if args.export:
        log.info("Exporting to INNX...")
        export_onnx(model, num_features)
        verify_onnx(num_features)

    return model, metrices

if __name__ == "__main__":
    main()
