import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from .lstm_config import (
    SEQ_LEN, PRED_HORIZON, STRIDE, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, MMAP_DIR
)
from data.config import PARQUET_PATH

log = logging.getLogger(__name__)

EMB_COLS = ["ticker_id", "market_id", "region_id", "interval_id"]

EXCLUDE_COLS = EMB_COLS + ["datetime", "ticker", "market", "region", "interval"]


def _load_parquet() -> pd.DataFrame:

    table = pq.read_table(str(PARQUET_PATH))
    df = table.to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values(["ticker", "interval", "datetime"], inplace = True)
    df.reset_index(drop = True, inplace = True)
    log.info(f"Loaded {len(df):,} rows | {df['ticker'].nunique():,} tickers | {df['interval'].nunique():,} intervals")
    return df

def _build_mmap(df,  split: str, force_rebuild: bool = False) -> tuple:

    outdir = Path(MMAP_DIR) / split
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "X_num": outdir / "X_num.npy",
        "X_emb": outdir / "X_emb.npy",
        "y_price": outdir / "y_price.npy",
        "y_dir": outdir / "y_dir.npy",
        "meta": outdir / "meta.npz",
    }

    if not force_rebuild and all(p.exists() for p in paths.values()):
        log.info(f"Loading cached mmap [{split}] ...")
        meta = np.load(paths["meta"])
        N, F = int(meta["N"]), int(meta["F"])
        X_num = np.load(str(paths["X_num"]), mmap_mode="r")
        X_emb = np.load(str(paths["X_emb"]), mmap_mode="r")
        y_price = np.load(str(paths["y_price"]), mmap_mode="r")
        y_dir = np.load(str(paths["y_dir"]), mmap_mode="r")
        log.info(f"[{split}] {N:,} windows | {F} features")
        return X_num, X_emb, y_price, y_dir, F

    log.info(f"Building mmap [{split}] from {len(df):,} rows ...")

    num_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    num_features = len(num_cols)
    close_idx = num_cols.index("close")

    total_windows = 0
    groups = []

    for (ticker, interval), group in df.groupby(["ticker", "interval"], sort=False):

        g = group.sort_values("datetime").reset_index(drop=True)
        n = len(group)
        n_tr = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        if split == "train":
            g = g.iloc[:n_tr]
        elif split == "val":
            g = g.iloc[n_tr:n_tr + n_val]
        else:
            g = g.iloc[n_tr + n_val:]

        usable = len(g) - SEQ_LEN  - PRED_HORIZON + 1
        if usable <= 0:
            continue

        n_windows = (usable + STRIDE - 1) // STRIDE
        groups.append((ticker, interval, g, n_windows))
        total_windows += n_windows

    if total_windows == 0:
        raise ValueError(f"No usable windows found for split '{split}'")

    size_gb = total_windows * SEQ_LEN * num_features * 4 / 1e9
    log.info(f"[{split}] Pre-allocating {total_windows:,} windows | {num_features} features | {size_gb:.2f} GB")

    X_num = np.lib.format.open_memmap(str(paths["X_num"]), mode="w+", dtype=np.float32, shape=(total_windows, SEQ_LEN, num_features))
    X_emb = np.lib.format.open_memmap(str(paths["X_emb"]), mode="w+", dtype=np.int64, shape=(total_windows, len(EMB_COLS)))
    y_price = np.lib.format.open_memmap(str(paths["y_price"]), mode="w+", dtype=np.float32, shape=(total_windows, ))
    y_dir = np.lib.format.open_memmap(str(paths["y_dir"]), mode="w+", dtype=np.float32, shape=(total_windows, ))

    cursor = 0
    for ticker, interval, g, _ in groups:
        vals = g[num_cols].values.astype(np.float32)
        emb = g[EMB_COLS].values.astype(np.int64)
        row = 0

        while True:
            end = row + SEQ_LEN
            target = end + PRED_HORIZON - 1
            if target >= len(vals):
                break

            window = vals[row:end].copy()
            base = window[0, close_idx]
            if base !=0 and not np.isnan(base):
                for idx in df.columns:        # indices of open/high/low/close/sma_*/ema_*/bb_*/vwap
                    window[:, idx] /= base

            X_num[cursor] = window
            X_emb[cursor] = emb[end - 1]
            y_price[cursor] = vals[target, close_idx]
            y_dir[cursor] = float(vals[target, close_idx] > vals[end - 1, close_idx])
            cursor += 1
            row += STRIDE

    actual = cursor
    log.info(f"[{split}] Built {actual:,} windows")
    np.savez(str(paths["meta"]), N=actual, F=num_features)

    del X_num, X_emb, y_price, y_dir

    X_num = np.load(str(paths["X_num"]), mmap_mode="r")[:actual]
    X_emb = np.load(str(paths["X_emb"]), mmap_mode="r")[:actual]
    y_price = np.load(str(paths["y_price"]), mmap_mode="r")[:actual]
    y_dir = np.load(str(paths["y_dir"]), mmap_mode="r")[:actual]

    return X_num, X_emb, y_price, y_dir, num_features


class MarketDataset(Dataset):

    def __init__(self, X_num, X_emb, y_price, y_dir,):

        self.X_num = X_num
        self.X_emb = X_emb
        self.y_price = y_price
        self.y_dir = y_dir

    def __len__(self):
        return len(self.y_price)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_num[idx].copy()),
            torch.from_numpy(self.X_emb[idx].copy()),
            torch.tensor(self.y_price[idx], dtype=torch.float32),
            torch.tensor(self.y_dir[idx], dtype=torch.float32),
        )


def make_dataloaders(force_rebuild: bool = False):

    df = _load_parquet()
    num_features: int | None = None
    loaders = {}

    for split in ("train", "val", "test"):
        X_num, X_emb, y_price, y_dir, nf = _build_mmap(df, split, force_rebuild=force_rebuild)
        if num_features is None:
            num_features = nf

        ds = MarketDataset(X_num, X_emb, y_price, y_dir)
        loaders[split] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )

        log.info(f"{split:5s}: {len(ds):,} samples")

    assert num_features is not None, "No splits were processed; num_features was never set"

    return loaders["train"], loaders["val"], loaders["test"], num_features
