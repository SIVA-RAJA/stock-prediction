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

def _build_groups(df,  split: str) -> tuple:

    log.info(f"Building Groups...")

    num_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    close_idx = num_cols.index("close")

    PRICE_COLS = (
        ["open", "high", "low", "close", "bb_upper", "bb_mid", "bb_lower", "vwap",
        "macd", "macd_signal", "macd_hist", "atr_14", "hl_range", "co_range"]
        + [c for c in num_cols if c.startswith("sma_")]
        + [c for c in num_cols if c.startswith("ema_")]
        + [c for c in num_cols if c.startswith("close_lag_")]
    )
    price_col_idxs = [num_cols.index(c) for c in PRICE_COLS if c in num_cols]

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

        val = g[num_cols].values.astype(np.float32)
        emb = g[EMB_COLS].values.astype(np.int64)
        groups.append((val, emb))

    return groups, num_cols, close_idx, price_col_idxs


class MarketDataset(Dataset):

    def __init__(self, groups, close_idx, price_col_idxs):

        self.groups = groups
        self.close_idx = close_idx
        self.price_col_idxs = price_col_idxs
        self.index = []

        for gi, (vala, _) in enumerate(groups):
            usable = len(vala) - SEQ_LEN - PRED_HORIZON + 1
            n_windows = (usable + STRIDE - 1) // STRIDE

            for w in range(n_windows):
                start = w * STRIDE
                self.index.append((gi, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        gi, row = self.index[idx]
        vals, emb = self.groups[gi]
        end = row + SEQ_LEN
        target = end + PRED_HORIZON - 1

        window = vals[row:end].copy()
        base = window[0, self.close_idx]
        if base != 0 and not np.isnan(base):
            window[:, self.price_col_idxs] /= base

        return (
            torch.from_numpy(window),
            torch.from_numpy(emb[end - 1]),
            torch.tensor(vals[target, self.close_idx], dtype=torch.float32),
            torch.tensor(float(vals[target, self.close_idx] > vals[end - 1, self.close_idx]), dtype=torch.float32),
        )


def make_dataloaders(force_rebuild: bool = False):

    df = _load_parquet()
    num_features: int | None = None
    loaders = {}

    for split in ("train", "val", "test"):
        groups, num_clos, close_idx, price_col_idxs = _build_groups(df, split)
        if num_features is None:
            num_features = len(num_clos)

        ds = MarketDataset(groups, close_idx, price_col_idxs)
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
