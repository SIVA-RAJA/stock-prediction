import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path

from .config import SCALER_DIR

log = logging.getLogger(__name__)

_NO_SCALE_PREFIXS = ("sin_", "cos_")
_NO_SCALE = {"ticker_id", "market_id", "region_id", "interval_id", "market_regime"}

def _scalable_cols(df: pd.DataFrame) -> list[str]:

    cols = []
    for col in df.columns:
        if col in _NO_SCALE:
            continue
        if any(col.startswith(prefix) for prefix in _NO_SCALE_PREFIXS):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)
    return cols

def _scaler_path(ticker: str, interval: str, ) -> Path:

    safe_ticker = ticker.replace("/", "_").replace("^", "_").replace("=", "_")
    return SCALER_DIR / f"{safe_ticker}_{interval}.joblib"

def fit_and_scale(df: pd.DataFrame, ticker: str, interval: str, save: bool = True) -> tuple[pd.DataFrame, RobustScaler, list[str]]:

    df = df.copy()
    cols = _scalable_cols(df)

    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    df[cols] = df[cols].clip(-10, 10)

    if save:
        path = _scaler_path(ticker, interval)
        joblib.dump({"scaler": scaler, "cols": cols}, path)
        log.debug(f"Scaler saved to {path.name}")

    return df, scaler, cols


def load_scaler(ticker: str, interval: str) -> tuple[RobustScaler, list[str]] | None:

    path = _scaler_path(ticker, interval)
    if not path.exists():
        log.warning(f"Scaler file not found: {path.name}")
        return None

    data = joblib.load(path)

    return data["scaler"], data["cols"]


def inverse_scale(df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame | None:

    scaler_data = load_scaler(ticker, interval)
    if scaler_data is None:
        return None

    scaler, cols = scaler_data
    cols = [str(c) for c in cols]
    df = df.copy()
    present = [col for col in cols if col in df.columns]

    dummy = np.zeros((len(df), len(cols)))
    for i, col in enumerate(cols):
        if col in present:
            dummy[:, i] = df[col].to_numpy()

    inv = scaler.inverse_transform(dummy)

    for i, col in enumerate(cols):
        if col in present:
            df.loc[:, col] = inv[:, i]

    return df

def scale_all(featured: dict) -> dict:

    scaled = {}
    ok = err = 0
    for market, regions in featured.items():
        scaled[market] = {}
        for region, tickers in regions.items():
            scaled[market][region] = {}
            for ticker, intervals in tickers.items():
                scaled[market][region][ticker] = {}
                for interval, df in intervals.items():
                    try:
                        df_scaled, _, _ = fit_and_scale(df, ticker, interval, save=True)
                        scaled[market][region][ticker][interval] = df_scaled
                        ok += 1
                    except Exception as e:
                        log.error(f"Scaling failed for {ticker} @ {interval}: {e}")
                        err += 1

    log.info(f"Scaling completed: {ok} datasets scaled, {err} errors")
    return scaled
