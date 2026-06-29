"""
Usage:
        python pipeline.py                # Download, clean, feature engineering, scaling, and write to HDF5 and Parquet
        python pipeline.py --skip-download   # Skip downloading raw data and use cached data if available
"""


import argparse
import logging
import pickle
import time

from .config import DATA_DIR
from .downloader import download_all
from .cleaner import clean_all
from .hdf5_writer import write_hdf5
from .features import add_features_all
from .scaler import scale_all
from .parquet_writer import write_parquet

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S", )

CACHE_PATH = DATA_DIR / "raw_cache.pkl"

def _save_cache(raw: dict) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(raw, f)
    log.info(f"Saved raw cache to {CACHE_PATH}")

def _load_cache() -> dict:
    with open(CACHE_PATH, "rb") as f:
        raw = pickle.load(f)
    log.info(f"Loaded raw cache from {CACHE_PATH}")
    return raw

def run(skip_download: bool = False):
    t0 = time.time()

    if skip_download and CACHE_PATH.exists():
        log.info("Skipping download and loading raw data from cache...")
        raw = _load_cache()
    else:
        log.info("=" * 60)
        log.info("Step 1/6: Downloading raw data...")
        log.info("=" * 60)
        raw = download_all(batch_size=10, sleep_between_batches=1.5)
        _save_cache(raw)

    log.info("=" * 60)
    log.info("Step 2/6: Cleaning raw data...")
    log.info("=" * 60)
    cleaned = clean_all(raw)

    log.info("=" * 60)
    log.info("Step 3/6: Writing cleaned data to HDF5...")
    log.info("=" * 60)
    write_hdf5(cleaned)

    log.info("=" * 60)
    log.info("Step 4/6: Adding features to cleaned data...")
    log.info("=" * 60)
    featured = add_features_all(cleaned)

    log.info("=" * 60)
    log.info("Step 5/6: Scaling features...")
    log.info("=" * 60)
    scaled = scale_all(featured)

    log.info("=" * 60)
    log.info("Step 6/6: Writing scaled data to Parquet...")
    log.info("=" * 60)
    write_parquet(scaled)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETED in {elapsed/60:.1f} minutes")
    log.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading raw data and use cached data if available.")
    args = parser.parse_args()

    run(skip_download=args.skip_download)
