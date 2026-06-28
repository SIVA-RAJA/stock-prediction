"""
Usage:

      python main.py                               # full run
      python main.py --skip-download               # reuse cached raw.pkl if present
      python main.py --rebuild                     # force rebuild LSTM mmap cache
      python main.py --eval-only                   # skip training, load best checkpoint and evaluate
      python main.py --export                      # export trained model to ONNX
"""

import sys
import logging
import time
import argparse
from io import StringIO
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "dataset"
LSTM_DIR = ROOT / "LSTM"

for _p in(str(ROOT), str(DATA_DIR), str(LSTM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


_log_buffer = StringIO()

def _setup_logging():

    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt="%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    buffer_handler = logging.StreamHandler(_log_buffer)
    buffer_handler.setFormatter(formatter)
    root.addHandler(buffer_handler)


_setup_logging()

log = logging.getLogger("main")

import data.pipeline as data_pipeline
from data.config import TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID, INTERVAL_TO_ID
from LSTM.lstm_config import DEVICE, BEST_CKPT
from LSTM.lstm_dataset import make_dataloaders
from LSTM.lstm_model import MarketLSTM
from LSTM.lstm_trainer import train, load_checkpoint
from LSTM.lstm_evaluate import evaluate
from LSTM.lstm_export import export_onnx, verify_onnx


def _write_report(summary_lines: list[str], stamp: str) -> Path:

    report_path = LOG_DIR / f"run_{stamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write(f"RUN SUMMARY\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("FULL LOG\n")
        f.write("=" * 70 + "\n")
        f.write(_log_buffer.getvalue())

    return report_path

def _finish(summary: list[str], stamp: str, t_start: float) -> None:

    elapsed = time.time() - t_start
    summary.append(f"Total time: {elapsed / 60:.1f} minutes")
    summary.append(f"Finished time : {datetime.now().isoformat(timespec='seconds')}")

    report_path = _write_report(summary, stamp)
    log.info(f"Summary saved -> {report_path}")


def main():

    parser = argparse.ArgumentParser(description="Run the full data + LSTM pipeline end to end")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading raw data, reuse cached raw.pkl if available")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild LSTM mmap cache")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, load best checkpoint and evaluate only")
    parser.add_argument("--export", action="store_true", help="Export trained model to ONNX after evaluation")
    args = parser.parse_args()

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = [f"Run started at {datetime.now().isoformat(timespec='seconds')}"]

    log.info("*" * 70)
    log.info(f"STAGE 1/2 : DATA PIPELINE (data/)")
    log.info("*" * 70)

    try:
        data_pipeline.run(skip_download=args.skip_download)
        summary.append("Data pipeline completed successfully")
    except Exception as e:
        log.error(f"Data pipeline failed: {e}")
        summary.append(f"Data pipeline failed: {e}")
        _finish(summary, stamp, t_start)
        raise

    log.info("*" * 70)
    log.info(f"STAGE 2/2 : LSTM TRAINING & EVALUATION (LSTM/)")
    log.info("*" * 70)

    try:

        train_loader, val_loader, test_loader, num_features = make_dataloaders(force_rebuild=args.rebuild)
        model = MarketLSTM(
            num_features=num_features,
            num_tickers=len(TICKER_TO_ID),
            num_markets=len(MARKET_TO_ID),
            num_regions=len(REGION_TO_ID),
            num_intervals=len(INTERVAL_TO_ID),
        )

        summary.append(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        summary.append(f"Device: {DEVICE}")
        summary.append(f"Features: {num_features}")

        if args.eval_only:
            if not BEST_CKPT.exists():
                raise FileNotFoundError(f"No checkpoint found at {BEST_CKPT}")
            epoch, val_loss =load_checkpoint(model, path=BEST_CKPT)
            summary.append(f"Loaded best checkpoint from epoch {epoch} val_loss={val_loss:.6f}")

        else:
            model = train(model, train_loader, val_loader, run_name="run")
            summary.append("Training completed successfully")

        metrics = evaluate(model, test_loader, run_name="eval")
        summary.append(f"Evaluation metric")
        for key, value in metrics.items():
            summary.append(f"  {key : < 15}: {value:.6f}")

        if args.export:
            export_onnx(model, num_features)
            verify_onnx(num_features)
            summary.append("Model exported to ONNX and verified successfully")

    except Exception as e:
        log.exception(f"LSTM stage failed: {e}")
        summary.append(f"LSTM stage failed: {e}")
        _finish(summary, stamp, t_start)
        raise

    _finish(summary, stamp, t_start)


if __name__ == "__main__":
    main()

