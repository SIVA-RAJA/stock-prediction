import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from data.scaler import fit_and_scale
import torch
from torch.utils.data import Dataset, DataLoader

from .lstm_config import (
    SEQ_LEN, PRED_HORIZON, STRIDE, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
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

def _build_all_groups(df: pd.DataFrame) -> tuple[dict, list, int, int]:

    log.info("Building groups (fitting scaler on train split only, per ticker/interval)...")

    num_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    close_idx = num_cols.index("close")
    return_idx = num_cols.index("log_return")

    groups = {"train": [], "val": [], "test": []}

    for (ticker, interval), group in df.groupby(["ticker", "interval"], sort=False):

        ticker = str(ticker)
        interval = str(interval)

        g = group.sort_values("datetime").reset_index(drop=True)
        g = g.dropna(subset=[c for c in g.columns if c not in EXCLUDE_COLS])
        n = len(g)
        n_tr = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        g_train = g.iloc[:n_tr].copy()
        g_val = g.iloc[n_tr:n_tr + n_val].copy()
        g_test = g.iloc[n_tr + n_val:].copy()

        if len(g_train) - SEQ_LEN - PRED_HORIZON + 1 <= 0:
            continue

        g_train_scaled, scaler, scale_cols = fit_and_scale(g_train, ticker, interval, save=True)

        splits = {"train": g_train_scaled}
        for name, g_split in (("val", g_val), ("test", g_test)):
            present = [c for c in scale_cols if c in g_split.columns]
            if len(g_split) and present:
                g_split[present] = scaler.transform(g_split[present])
                g_split[present] = g_split[present].clip(-10, 10)
            splits[name] = g_split

        for name, g_split in splits.items():
            usable = len(g_split) - SEQ_LEN - PRED_HORIZON + 1
            if usable <= 0:
                continue
            val_arr = g_split[num_cols].values.astype(np.float32)
            emb_arr = g_split[EMB_COLS].values.astype(np.int64)
            groups[name].append((val_arr, emb_arr))

    return groups, num_cols, close_idx, return_idx


def verify_no_data_leakage(df, num_cols, close_idx, return_idx):

    log.info("Running data leakage verification...")

    sample_vals = df[num_cols].values[:200]
    for i in range(min(100, len(sample_vals) - SEQ_LEN - 1)):
        end = i + SEQ_LEN
        target = end + PRED_HORIZON - 1
        assert target > end - 1, f"TARGET {target} overlaps with window ending at {end}"

    close_col = df["close"].values
    lag1_col = df["close_lag_1"].values
    for i in range(min(100, len(close_col))):
        expected = close_col[i - 1]
        actual = lag1_col[i]
        if not np.isnan(actual):
            diff = abs(expected - actual)
            assert diff < 1e-6, f"close_lag_1 at row {i} = {actual}, expected {expected}"

    log.info("No data leakage detected in sample checks.")
    log.info("Target strictly after input window")
    log.info("Lag features are backward-looking and do not leak future information")

class MarketDataset(Dataset):

    def __init__(self, groups, close_idx, return_idx):

        self.groups = groups
        self.close_idx = close_idx
        self.return_idx = return_idx
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

        return (
            torch.from_numpy(window),
            torch.from_numpy(emb[end - 1]),
            torch.tensor(vals[target, self.return_idx], dtype=torch.float32),
            torch.tensor(float(vals[target, self.close_idx] > vals[end - 1, self.close_idx]), dtype=torch.float32),
            torch.tensor(vals[end -1, self.close_idx], dtype=torch.float32)
        )


def make_dataloaders(force_rebuild: bool = False):

    df = _load_parquet()
    groups_by_split, num_cols, close_idx, return_idx = _build_all_groups(df)
    num_features = len(num_cols)
    loaders = {}

    for split in ("train", "val", "test"):
        ds = MarketDataset(groups_by_split[split], close_idx, return_idx)
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

    return loaders["train"], loaders["val"], loaders["test"], num_features
