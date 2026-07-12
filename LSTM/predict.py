"""

USAGE:
        PYTHON:

                     from LSTM.predict import predict
                     result = predict(ticker="AAPL", market="STOCK", region="USA", interval="1d")
                     print(result)

        COMMAND LINE (CLI):

                     python -m LSTM.predict --ticker AAPL --market STOCK --region USA --interval 1d

"""


import logging
import argparse
import os
from typing import Any
import psutil

import numpy as np
import pandas as pd
import yfinance as yf
import onnxruntime as ort

from data.config import CONFIGS, TICKER_TO_ID, MARKET_TO_ID, REGION_TO_ID, INTERVAL_TO_ID
from data.cleaner import clean_data
from data.features import add_featurers
from data.scaler import load_scaler, _scalable_cols

from .lstm_config import SEQ_LEN
from .lstm_dataset import EMB_COLS, EXCLUDE_COLS
from .lstm_export import ONNX_PATH


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

_WARMUP_BUFFER = 250

def _download_latest(ticker: str, interval: str) -> pd.DataFrame:

    if interval not in CONFIGS:
        raise ValueError(f"Invalid interval: {interval}. Must be one of {list(CONFIGS.keys())}")

    period, yf_interval = CONFIGS[interval]

    df = yf.download(tickers=ticker, period=period, interval=yf_interval, auto_adjust=True, actions=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker: {ticker} with interval: {interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower().strip() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}

    if not required.issubset(df.columns):
        raise RuntimeError(f"Download missing required columns in data for ticker: {ticker}. Found columns: {df.columns.tolist()}")

    return df[["open", "high", "low", "close", "volume"]]


def _build_features(raw_df: pd.DataFrame, ticker: str, market: str, region: str, interval: str) -> pd.DataFrame:

    cleaned = clean_data(raw_df, ticker=ticker, market=market, interval=interval)
    if cleaned is None:
        raise RuntimeError(f"Data cleaning failed for ticker: {ticker}, market: {market}, region: {region}, interval: {interval}")

    featured = add_featurers(cleaned, ticker=ticker, market=market, region=region, interval=interval)
    if featured is None:
        raise RuntimeError(f"Feature engineering failed for ticker: {ticker}, market: {market}, region: {region}, interval: {interval}")

    if len(featured) < SEQ_LEN:
        raise RuntimeError(f"Not enough data after feature engineering for ticker: {ticker}, market: {market}, region: {region}, interval: {interval}. Required at least {SEQ_LEN } rows, got {len(featured)}, try longer history interval.")

    return featured


def _apply_scaler(df: pd.DataFrame, market: str, interval: str) -> pd.DataFrame:

    bundle = load_scaler(key=market, interval=interval)
    if bundle is None:
        raise RuntimeError(f"Scaler loading failed for market: {market}, interval: {interval}")

    scaler, saved_cols = bundle

    df = df.copy()

    current_cols = _scalable_cols(df)
    if current_cols != list(saved_cols):
        missing = set(saved_cols) - set(current_cols)
        extra = set(current_cols) - set(saved_cols)
        raise RuntimeError(f"Scaler columns mismatch for market: {market}, interval: {interval}. Expected columns: {saved_cols}, got: {current_cols}. Missing: {missing}, Extra: {extra}"
                           f"This means feature.py / config.py has changed since the scaler was saved. Please re-run the training pipeline to generate a new scaler.")

    df[saved_cols] = scaler.transform(df[saved_cols])
    df[saved_cols] = df[saved_cols].clip(-10, 10)

    return df


def _inverse_close(scaled_value: float, market: str, interval: str) -> float:

    bundle = load_scaler(key=market, interval=interval)
    if bundle is None:
        return scaled_value

    scaler, cols = bundle

    if "close" not in cols:
        return scaled_value

    close_idx = cols.index("close")
    dummmy = np.zeros((1, len(cols)))
    dummmy[:, close_idx] = scaled_value
    inv = scaler.inverse_transform(dummmy)

    return float(inv[0, close_idx])

_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:

    global _session
    if _session is None:
        if not ONNX_PATH.exists():
            raise RuntimeError(f"ONNX model not found at {ONNX_PATH}. Please run 'python main.py --export-model' first.")

        process = psutil.Process(os.getpid())
        log.info(f"Memory before loading ONNX model: {process.memory_info().rss / (1024 ** 2):.2f} MB")
        _session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        log.info(f"Memory after loading ONNX model: {process.memory_info().rss / (1024 ** 2):.2f} MB")
    return _session


def predict(ticker: str, market: str, region: str, interval: str) -> dict[str, Any]:

    if ticker not in TICKER_TO_ID:
        raise ValueError(f"Unknown ticker '{ticker}' - not present in TICKER_TO_ID (config.py TICKERS)")

    if market not in MARKET_TO_ID:
        raise ValueError(f"Unknown market '{market}'")

    if region not in REGION_TO_ID:
        raise ValueError(f"Unknown region '{region}'")

    if interval not in INTERVAL_TO_ID:
        raise ValueError(f"Unknown interval '{interval}' - not present in INTERVAL_TO_ID (config.py CONFIGS)")

    log.info(f"Downloading latest data for {ticker} at interval {interval}....")
    raw_df = _download_latest(ticker=ticker, interval=interval)

    log.info(f"Building features for {ticker} at interval {interval}....")
    featured = _build_features(raw_df, ticker=ticker, market=market, region=region, interval=interval)

    log.info(f"Applying scaler (transform only, no re-fit) for {ticker} at interval {interval}....")
    scaled = _apply_scaler(featured, market=market, interval=interval)

    num_cols = [c for c in scaled.columns if c not in EXCLUDE_COLS]

    window_df = scaled.iloc[-SEQ_LEN:]
    x_num = window_df[num_cols].values.astype(np.float32)[None, ...]
    x_emb = window_df[EMB_COLS].values.astype(np.int64)[-1:, :]

    sess = _get_session()

    expected_features = sess.get_inputs()[0].shape[-1]

    if isinstance(expected_features, int) and expected_features != x_num.shape[-1]:
        raise RuntimeError(f"Model input shape mismatch: expected {expected_features}, got {x_num.shape[-1]}"
                           f"features.py has changed since the model was exported. Please re-run the training pipeline to generate a new model.")

    raw_outputs = sess.run(None, {"x_num": x_num, "x_emb": x_emb})
    dir_pred_logit, attn_weights = (np.array(o) for o in raw_outputs)

    dir_prob = float(1.0 / (1.0 + np.exp(-dir_pred_logit.squeeze())))
    dir_label = "UP" if dir_prob >= 0.5 else "DOWN"
    dir_confidence = dir_prob if dir_label == "UP" else 1.0 - dir_prob

    last_raw_close = float(raw_df["close"].iloc[-1])

    last_timestamp = window_df.index[-1]

    result = {
        "ticker": ticker,
        "market": market,
        "region": region,
        "interval": interval,
        "last_close": round(last_raw_close, 4),
        "direction" : dir_label,
        "direction_confidence": round(dir_confidence, 4),
        "attention_weights": attn_weights.squeeze().tolist(),
    }

    log.info(f"{ticker} @ {interval} | Last Close: {result['last_close']:.4f} | "
             f"Direction: {result['direction']} (Confidence: {result['direction_confidence'] * 100:.2f}%)")

    return result


def _cli():

    parser = argparse.ArgumentParser(description="Run inference with the trained ONNX model.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--market", type=str, required=True, help="Market (e.g., STOCK)")
    parser.add_argument("--region", type=str, required=True, help="Region (e.g., USA)")
    parser.add_argument("--interval", type=str, required=True, help="Interval (e.g., 1d)")

    args = parser.parse_args()

    result = predict(ticker=args.ticker, market=args.market, region=args.region, interval=args.interval)

    for k, v in result.items():
        if k != "attention_weights":
            print(f"{k:25s}: {v}")


if __name__ == "__main__":
    _cli()
