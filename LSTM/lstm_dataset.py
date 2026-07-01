import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from data.scaler import fit_and_scale
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
    print(f"[DEBUG dataset] loaded shape={df.shape} datetime_range=({df['datetime'].min()}, {df['datetime'].max()}) nan_total={df.isna().sum().sum()}")            #DEBUG
    log.info(f"Loaded {len(df):,} rows | {df['ticker'].nunique():,} tickers | {df['interval'].nunique():,} intervals")
    return df

def _build_all_groups(df: pd.DataFrame) -> tuple[dict, list, int]:

    log.info("Building groups (fitting scaler on train split only, per ticker/interval)...")

    num_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    close_idx = num_cols.index("close")

    groups = {"train": [], "val": [], "test": []}

    for (ticker, interval), group in df.groupby(["ticker", "interval"], sort=False):

        ticker = str(ticker)
        interval = str(interval)

        g = group.sort_values("datetime").reset_index(drop=True)
        n = len(g)
        n_tr = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        print(f"[DEBUG dataset] {ticker}/{interval}: n={n} n_tr={n_tr} n_val={n_val} n_test={n - n_tr - n_val}")           #DEBUG

        g_train = g.iloc[:n_tr].copy()
        g_val = g.iloc[n_tr:n_tr + n_val].copy()
        g_test = g.iloc[n_tr + n_val:].copy()

        if len(g_train) - SEQ_LEN - PRED_HORIZON + 1 <= 0:
            continue

        g_train_scaled, scaler, scale_cols = fit_and_scale(g_train, ticker, interval, save=True)

        print(f"[DEBUG dataset] {ticker}/{interval}: scale_cols={len(scale_cols)} nan_after_fit={g_train_scaled.isna().sum().sum()}")          #DEBUG

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

    print(f"[DEBUG dataset] group counts: train={len(groups['train'])} val={len(groups['val'])} test={len(groups['test'])} num_features={len(num_cols)} close_idx={close_idx}")                    #DEBUG

    return groups, num_cols, close_idx

class MarketDataset(Dataset):

    def __init__(self, groups, close_idx):

        self.groups = groups
        self.close_idx = close_idx
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

        if torch.isnan(torch.from_numpy(window)).any():
            print(f"[DEBUG dataset] NaN detected in window at idx={idx}, group={gi}, row={row}")         #DEBUG

        return (
            torch.from_numpy(window),
            torch.from_numpy(emb[end - 1]),
            torch.tensor(vals[target, self.close_idx], dtype=torch.float32),
            torch.tensor(float(vals[target, self.close_idx] > vals[end - 1, self.close_idx]), dtype=torch.float32),
        )


def make_dataloaders(force_rebuild: bool = False):

    df = _load_parquet()
    groups_by_split, num_cols, close_idx = _build_all_groups(df)
    num_features = len(num_cols)
    print(f"[DEBUG dataset] num_features={num_features}")          #DEBUG
    loaders = {}

    for split in ("train", "val", "test"):
        ds = MarketDataset(groups_by_split[split], close_idx)
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
