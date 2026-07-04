"""
main.py — single entry point for the whole data + LSTM pipeline.

Every command below checks whether the things it depends on already exist.
If something is missing, you'll be asked (y/N) whether to build it first.
Say "y" and it cascades automatically; say "n" (or pipe non-interactively)
and the run stops with an error telling you exactly which command to run.

Usage:

    python main.py --download                 # download fresh data from yfinance, then run the full data pipeline
    python main.py --skip-download             # run the full data pipeline, reusing cached raw data if present
    python main.py --train-model                # train the LSTM (builds data first if missing)
    python main.py --evaluate-model              # evaluate the LSTM (trains first if missing)
    python main.py --train-evaluate-model          # train + evaluate in one go (builds data first if missing)
    python main.py --export-model                   # export trained model to ONNX (evaluates/trains/builds data first if missing)

Optional flags (combine with any command above):
    --rebuild      force-rebuild the LSTM mmap cache even if it's already on disk
    -y / --yes     auto-answer "yes" to every confirmation prompt (non-interactive use)

    tensorboard --logdir runs/   # visualize training metrics (live)
"""

import sys
import logging
import pickle
import time
import argparse
from io import StringIO
from pathlib import Path
from datetime import datetime
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LSTM_DIR = ROOT / "LSTM"

for _p in (str(ROOT), str(DATA_DIR), str(LSTM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

_log_buffer = StringIO()


def _setup_logging():

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
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


# --------------------------------------------------------------------------- #
# Imports (all the pieces that used to be invoked from data/pipeline.py and
# LSTM/lstm_main.py now live here and are driven directly from main.py)
# --------------------------------------------------------------------------- #

from data.config import ARTIFACTS_DIR, DATA_DIR as DATASET_DIR, PARQUET_PATH, TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID, INTERVAL_TO_ID
from data.downloader import download_all
from data.cleaner import clean_all
from data.hdf5_writer import write_hdf5
from data.features import add_features_all
from data.parquet_writer import write_parquet

from LSTM.lstm_config import DEVICE, BEST_CKPT
from LSTM.lstm_dataset import make_dataloaders
from LSTM.lstm_model import MarketLSTM
from LSTM.lstm_trainer import train, load_checkpoint
from LSTM.lstm_evaluate import evaluate, RESULTS_DIR
from LSTM.lstm_export import export_onnx, verify_onnx, ONNX_PATH

RAW_CACHE_PATH = DATASET_DIR / "raw_cache.pkl"
EVAL_CSV_PATH = RESULTS_DIR / "eval_predictions.csv"

LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Small helpers: confirmation prompts / hard aborts / run report
# --------------------------------------------------------------------------- #

def _confirm(question: str, args: argparse.Namespace) -> bool:
    """Ask the user a y/N question. Auto-yes if --yes was passed."""
    if getattr(args, "yes", False):
        log.info(f"{question} [auto-yes]")
        return True
    try:
        answer = input(f"{question} [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _abort(message: str, *suggested_commands: str) -> None:
    log.error(message)
    if suggested_commands:
        log.error("Run one of the following first, then re-run your command:")
        for cmd in suggested_commands:
            log.error(f"    {cmd}")
    sys.exit(1)


def _write_report(summary_lines: list[str], stamp: str) -> Path:

    report_path = LOG_DIR / f"run_{stamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RUN SUMMARY\n")
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
    summary.append(f"Finished time: {datetime.now().isoformat(timespec='seconds')}")

    report_path = _write_report(summary, stamp)
    log.info(f"Summary saved -> {report_path}")


# --------------------------------------------------------------------------- #
# Stage 1: Data pipeline  (used to be data/pipeline.py)
# --------------------------------------------------------------------------- #

def _save_raw_cache(raw: dict) -> None:
    with open(RAW_CACHE_PATH, "wb") as f:
        pickle.dump(raw, f)
    log.info(f"Saved raw cache to {RAW_CACHE_PATH}")


def _load_raw_cache() -> dict:
    with open(RAW_CACHE_PATH, "rb") as f:
        raw = pickle.load(f)
    log.info(f"Loaded raw cache from {RAW_CACHE_PATH}")
    return raw


def data_pipeline_done() -> bool:
    """The final artifact of the data pipeline is the combined Parquet file."""
    return PARQUET_PATH.exists()


def run_data_pipeline(skip_download: bool) -> None:
    """Download (or reuse cache) -> clean -> hdf5 -> features -> scale -> parquet."""

    t0 = time.time()

    if skip_download and RAW_CACHE_PATH.exists():
        log.info("Skipping download and loading raw data from cache...")
        raw = _load_raw_cache()
    else:
        log.info("=" * 60)
        log.info("Step 1/5: Downloading raw data...")
        log.info("=" * 60)
        raw = download_all(batch_size=10, sleep_between_batches=1.5)
        _save_raw_cache(raw)

    log.info("=" * 60)
    log.info("Step 2/5: Cleaning raw data...")
    log.info("=" * 60)
    cleaned = clean_all(raw)

    log.info("=" * 60)
    log.info("Step 3/5: Writing cleaned data to HDF5...")
    log.info("=" * 60)
    write_hdf5(cleaned)

    log.info("=" * 60)
    log.info("Step 4/5: Adding features to cleaned data...")
    log.info("=" * 60)
    featured = add_features_all(cleaned)

    log.info("=" * 60)
    log.info("Step 5/5: Writing featured (unscaled) data to Parquet...")
    log.info("=" * 60)
    write_parquet(featured)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"DATA PIPELINE COMPLETED in {elapsed / 60:.1f} minutes")
    log.info("=" * 60)


def _default_skip_download() -> bool:
    """When the data step is triggered automatically (as a dependency of
    another command), reuse the raw cache if it exists instead of
    re-downloading everything from yfinance."""
    return RAW_CACHE_PATH.exists()


def ensure_data_pipeline(args: argparse.Namespace) -> None:
    """Make sure the processed Parquet dataset exists, building it if needed."""

    if data_pipeline_done():
        log.info(f"Processed dataset already present at {PARQUET_PATH} - skipping data pipeline.")
        return

    if not _confirm(
        "No processed dataset found. The data pipeline (download + clean + "
        "feature engineering + scaling) needs to run first. Do this now?",
        args,
    ):
        _abort(
            "Cannot continue: there is no data to work with.",
            "python main.py --download        # fetch fresh data from yfinance",
            "python main.py --skip-download    # reuse cached raw data, if any",
        )

    run_data_pipeline(skip_download=_default_skip_download())


# --------------------------------------------------------------------------- #
# Stage 2: Build dataloaders + model skeleton
# --------------------------------------------------------------------------- #

def build_model_and_loaders(args: argparse.Namespace):
    """Ensures data exists, then builds dataloaders and a fresh model instance."""

    ensure_data_pipeline(args)

    log.info("Building dataloaders...")
    train_loader, val_loader, test_loader, num_features = make_dataloaders(force_rebuild=args.rebuild)
    import numpy as np
    print(np.mean([float(y_dir) for *_, y_dir in train_loader.dataset]))

    model = MarketLSTM(
        num_features=num_features,
        num_tickers=len(TICKER_TO_ID),
        num_markets=len(MARKET_TO_ID),
        num_regions=len(REGION_TO_ID),
        num_intervals=len(INTERVAL_TO_ID),
    )

    log.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,} | "
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    log.info(f"Device: {DEVICE} | Features: {num_features}")

    return model, train_loader, val_loader, test_loader, num_features


# --------------------------------------------------------------------------- #
# Stage 3: Training
# --------------------------------------------------------------------------- #

def model_trained() -> bool:
    return BEST_CKPT.exists()


def ensure_trained_model(args: argparse.Namespace):
    """Return (model, test_loader, num_features), loading the best checkpoint
    if one already exists, otherwise training a new one (after confirmation)."""

    model, train_loader, val_loader, test_loader, num_features = build_model_and_loaders(args)

    if model_trained():
        log.info(f"Trained checkpoint found at {BEST_CKPT} - loading it instead of retraining.")
        epoch, val_loss = load_checkpoint(model, path=BEST_CKPT)
        log.info(f"Loaded checkpoint from epoch {epoch}, val_loss={val_loss:.6f}")
        return model, test_loader, num_features

    if not _confirm(
        "No trained model found. Training needs to run first. Do this now?",
        args,
    ):
        _abort(
            "Cannot continue: there is no trained model.",
            "python main.py --train-model",
        )

    log.info("Starting training...")
    model = train(model, train_loader, val_loader, run_name="run")
    log.info("Training completed successfully.")

    return model, test_loader, num_features


# --------------------------------------------------------------------------- #
# Stage 4: Evaluation
# --------------------------------------------------------------------------- #

def model_evaluated() -> bool:
    return EVAL_CSV_PATH.exists()


def ensure_evaluated_model(args: argparse.Namespace):
    """Return (model, num_features) for a model that has been evaluated at
    least once, evaluating (and training/building data first if needed) if
    it hasn't been."""

    if model_evaluated() and model_trained():
        log.info(f"Evaluation results already present at {EVAL_CSV_PATH} - loading trained model for export.")
        model, train_loader, val_loader, test_loader, num_features = build_model_and_loaders(args)
        load_checkpoint(model, path=BEST_CKPT)
        return model, num_features

    if not _confirm(
        "Model has not been evaluated yet. Evaluation needs to run first. Do this now?",
        args,
    ):
        _abort(
            "Cannot continue: there are no evaluation results.",
            "python main.py --evaluate-model",
        )

    model, test_loader, num_features = ensure_trained_model(args)
    log.info("Evaluating on test set...")
    evaluate(model, test_loader, run_name="eval")

    return model, num_features


# --------------------------------------------------------------------------- #
# Commands (one per CLI flag)
# --------------------------------------------------------------------------- #

def cmd_download(args: argparse.Namespace, summary: list[str]) -> None:
    log.info("*" * 70)
    log.info("DOWNLOAD + DATA PIPELINE (forced fresh download)")
    log.info("*" * 70)
    run_data_pipeline(skip_download=False)
    summary.append("Data pipeline completed (fresh download)")


def cmd_skip_download(args: argparse.Namespace, summary: list[str]) -> None:
    log.info("*" * 70)
    log.info("DATA PIPELINE (reusing cached raw data if available)")
    log.info("*" * 70)
    run_data_pipeline(skip_download=True)
    summary.append("Data pipeline completed (skip-download)")


def cmd_train_model(args: argparse.Namespace, summary: list[str]) -> None:
    log.info("*" * 70)
    log.info("TRAIN MODEL")
    log.info("*" * 70)
    model, train_loader, val_loader, test_loader, num_features = build_model_and_loaders(args)
    model = train(model, train_loader, val_loader, run_name="run")
    summary.append("Training completed successfully")
    summary.append(f"Checkpoint saved at: {BEST_CKPT}")


def cmd_evaluate_model(args: argparse.Namespace, summary: list[str]) -> None:
    log.info("*" * 70)
    log.info("EVALUATE MODEL")
    log.info("*" * 70)
    model, test_loader, num_features = ensure_trained_model(args)
    log.info("Evaluating on test set...")
    metrics = evaluate(model, test_loader, run_name="eval")
    summary.append("Evaluation completed")
    for key, value in metrics.items():
        summary.append(f"  {key:<15}: {value}")


def cmd_train_evaluate_model(args: argparse.Namespace, summary: list[str]) -> None:
    log.info("*" * 70)
    log.info("TRAIN + EVALUATE MODEL")
    log.info("*" * 70)
    model, train_loader, val_loader, test_loader, num_features = build_model_and_loaders(args)
    model = train(model, train_loader, val_loader, run_name="run")
    summary.append("Training completed successfully")

    log.info("Evaluating on test set...")
    metrics = evaluate(model, test_loader, run_name="eval")
    summary.append("Evaluation completed")
    for key, value in metrics.items():
        summary.append(f"  {key:<15}: {value}")


def cmd_export_model(args: argparse.Namespace, summary: list[str]) -> None:
    log.info("*" * 70)
    log.info("EXPORT MODEL")
    log.info("*" * 70)
    model, num_features = ensure_evaluated_model(args)
    log.info("Exporting to ONNX...")
    export_onnx(model, num_features)
    verify_onnx(num_features)
    summary.append("Model exported to ONNX and verified successfully")
    summary.append(f"ONNX file: {ONNX_PATH}")


COMMANDS = {
    "download": cmd_download,
    "skip_download": cmd_skip_download,
    "train_model": cmd_train_model,
    "evaluate_model": cmd_evaluate_model,
    "train_evaluate_model": cmd_train_evaluate_model,
    "export_model": cmd_export_model,
}


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():

    parser = argparse.ArgumentParser(
        description="Run the data + LSTM pipeline. Pick exactly one command; "
                     "missing prerequisites are built automatically after confirmation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--download", action="store_true", help="Download fresh data from yfinance, then run the full data pipeline")
    action.add_argument("--skip-download", action="store_true", help="Run the full data pipeline, reusing cached raw data if present")
    action.add_argument("--train-model", action="store_true", help="Train the LSTM model (builds data first if missing)")
    action.add_argument("--evaluate-model", action="store_true", help="Evaluate the LSTM model (trains first if missing)")
    action.add_argument("--train-evaluate-model", action="store_true", help="Train and evaluate the LSTM model (builds data first if missing)")
    action.add_argument("--export-model", action="store_true", help="Export the trained model to ONNX (evaluates/trains/builds data first if missing)")

    parser.add_argument("--rebuild", action="store_true", help="Force rebuild LSTM mmap cache")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-confirm all prompts (non-interactive)")

    args = parser.parse_args()

    if args.download:
        command = "download"
    elif args.skip_download:
        command = "skip_download"
    elif args.train_model:
        command = "train_model"
    elif args.evaluate_model:
        command = "evaluate_model"
    elif args.train_evaluate_model:
        command = "train_evaluate_model"
    else:
        command = "export_model"

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = [
        f"Run started at {datetime.now().isoformat(timespec='seconds')}",
        f"Command: --{command.replace('_', '-')}",
    ]

    try:
        COMMANDS[command](args, summary)
    except SystemExit:
        raise
    except Exception as e:
        log.exception(f"Run failed: {e}")
        summary.append(f"Run failed: {e}")
        _finish(summary, stamp, t_start)
        raise

    _finish(summary, stamp, t_start)


if __name__ == "__main__":
    main()
