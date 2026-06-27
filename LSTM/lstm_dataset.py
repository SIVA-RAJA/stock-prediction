import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from lstm_config import (
    PARQUET_DIR, SEQ_LEN, PRED_HORIZON, STRIDE, TRAIN_FRAC, VAL_FRAC, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
)

log = logging.getLogger(__name__)

EMB_COLS = ["ticker_id", "market_id", "region_id"]

EXCLUDE_COLS = EMB_COLS + ["datetime", "ticker", "market", "region", "interval"]

def _loada_partition(market: str | None = None, interval: str | None = None, region: str | None = None,) -> pd.DataFrame:

    filters = []
    if market:
        filters.append(("market", "=", market))
    if interval:
        filters.append(("interval", "=", interval))
    if region:
        filters.append(("region", "=", region))

    table = pq.read_table(str(PARQUET_DIR), filters=filters if filters else None)
    df = table.to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values(["ticker", "datetime"], inplace = True)
    df.reset_index(drop = True, inplace = True)
    return df

def _build_windows(df_ticker: pd.DataFrame, seq_len: int, horizon: int, stride: int, ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None, None]:
    num_cols = [c for c in df_ticker.columns if c not in EXCLUDE_COLS]
    close_idx = num_cols.index("close")

    values = df_ticker[num_cols].values.astype(np.float32)
    emb_vals = df_ticker[EMB_COLS].values.astype(np.int64)

    X_num, X_emb, y_price, y_dir = [], [], [], []

    total = len(values)
    for start in range(0, total - seq_len - horizon + 1, stride):
        end = start + seq_len
        target = end + horizon - 1

        x_window = values[start:end]
        e_window = emb_vals[end - 1]

        close_now = values[end - 1, close_idx]
        close_future = values[target, close_idx]

        direction = 1.0 if close_future > close_now else 0.0

        X_num.append(x_window)
        X_emb.append(e_window)
        y_price.append(close_future)
        y_dir.append(direction)

    if not X_num:
        return None, None, None, None

    return (
        np.stack(X_num, axis=0),
        np.stack(X_emb, axis=0),
        np.array(y_price, dtype=np.float32),
        np.array(y_dir, dtype=np.float32),
    )

class MarketDataset(Dataset):

    def __init__(self, df: pd.DataFrame, seq_len: int = SEQ_LEN, horizon: int = PRED_HORIZON, stride: int = STRIDE, ):
        self.seq_len = seq_len
        self.horizon = horizon

        all_X_num, all_X_emb, all_y_price, all_y_dir = [], [], [], []

        for ticker, group, in df.groupby("ticker", sort=False):
            group = group.sort_values("datetime").reset_index(drop=True)
            Xn, Xe, yp, yd = _build_windows(group, seq_len, horizon, stride)

            if Xn is None:
                log.warning(f" Not enough rows for {ticker}, skipping")
                continue
            all_X_num.append(Xn)
            all_X_emb.append(Xe)
            all_y_price.append(yp)
            all_y_dir.append(yd)

        if not all_X_num:
            raise ValueError("No windows could be built from this partition.")

        self.X_num = torch.from_numpy(np.concatenate(all_X_num, axis=0))
        self.X_emb = torch.from_numpy(np.concatenate(all_X_emb, axis=0))
        self.y_price = torch.from_numpy(np.concatenate(all_y_price, axis=0))
        self.y_dir = torch.from_numpy(np.concatenate(all_y_dir, axis=0))

        log.info(f" Dataset built:{len(self):,} windows | "
                 f" features={self.X_num.shape[-1]} seq={seq_len}")

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_emb[idx],
            self.y_price[idx],
            self.y_dir[idx],
        )

def chronological_split(dataset: MarketDataset, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC, ) -> tuple[Dataset, Dataset, Dataset]:
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(dataset, range(n_train+n_val, n))

    log.info(f"Split -> tain={len(train_ds):,} val{len(val_ds):,} test={len(test_ds):,}")
    return train_ds, val_ds, test_ds

def make_dataloaders(market: str | None = None, interval: str | None = None, region: str | None = None) -> tuple[DataLoader, DataLoader, DataLoader, int]:

    log.info(f"Loading data: market={market}, interval={interval}, region={region}")
    df = _loada_partition(market, interval, region)
    log.info(f"Loaded {len(df):,} roes, {df['ticker'].nunique()} tickers")

    dataset = MarketDataset(df)
    train_ds, val_ds, test_ds, = chronological_split(dataset)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    num_features = dataset.X_num.shape[-1]

    return train_loader, val_loader, test_loader, num_features
