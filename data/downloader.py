import time
import logging
from typing import Dict, Optional
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

from .config import TICKERS, CONFIGS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

def _all_tickers_meta() -> list[dict]:

    meta = []
    for market, regions in TICKERS.items():
        for region, tickers in regions.items():
            for tk in tickers:
                meta.append({"market": market, "region": region, "ticker": tk})
    return meta

def _flat_tickers() -> list[str]:
    return [m["ticker"] for m in _all_tickers_meta()]

@retry(
    retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                   requests.exceptions.Timeout,
                                )),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)

def _download_batch(tickers: list[str], period: str, interval: str) -> Optional[pd.DataFrame]:

    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        actions=False,
        threads=True,
        progress=False,
    )
    return df

def _extract_single(df_multi: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:

    try:
        if isinstance(df_multi.columns, pd.MultiIndex):
            if ticker in df_multi.columns.get_level_values(0):
                df = df_multi[ticker].copy()
            else:
                return None
        else:
            df = df_multi.copy()

        if not isinstance(df, pd.DataFrame):
            return None

        df.columns = [c.lower().strip() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None

        df = df[["open", "high", "low", "close", "volume"]]
        return df

    except Exception as e:
        log.warning(f"Extract failed for {ticker}: {e}")
        return None

def download_all(
    batch_size: int = 10,
    sleep_between_batches: float = 1.5,) -> Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:

    meta = _all_tickers_meta()
    tickers_flat = _flat_tickers()

    raw: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    for m in meta:
        raw.setdefault(m["market"], {}).setdefault(m["region"], {}).setdefault(m["ticker"], {})

    total_intervals = len(CONFIGS)
    total_tickers = len(tickers_flat)
    log.info(f"Starting download: {total_tickers} tickers x {total_intervals} intervals")

    for iv_idx, (interval, (period, _)) in enumerate(CONFIGS.items(), 1):
        log.info(f"\n{'='*60}")
        log.info(f"Interval [{iv_idx}/{total_intervals}: {interval}  (period={period})")
        log.info(f"{'='*60}")

        for batch_start in range(0, total_tickers, batch_size):
            batch = tickers_flat[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_tickers + batch_size - 1) // batch_size
            log.info(f"Batch [{batch_num}/{total_batches}]: {batch}")

            try:
                df_multi = _download_batch(batch, period, interval)
            except Exception as e:
                log.error(f"Batch download failed after retries: {e}")
                continue

            if df_multi is None or df_multi.empty:
                log.warning(f"Empty response for batch at interval={interval}")
                continue

            for tk in batch:
                df_single = _extract_single(df_multi, tk)
                
                if df_single is None or df_single.empty:
                    log.warning(f" No data:{tk} @ {interval}")
                    continue

                tk_meta = next(m for m in meta if m["ticker"] == tk)
                market = tk_meta["market"]
                region = tk_meta["region"]

                raw[market][region][tk][interval] = df_single
                log.info(f"{tk:20s} @ {interval:4s}  rows={len(df_single)}")

            time.sleep(sleep_between_batches)

    log.info("\nDownload complete.")
    _log_summary(raw)
    return raw

def _log_summary(raw: dict):

    total = 0
    missing = 0
    for market, regions in raw.items():
        for region, tickers in regions.items():
            for tk, intervals in tickers.items():
                for interval in CONFIGS:
                    if interval in intervals and not intervals[interval].empty:
                        total += 1
                    else:
                        missing += 1
                        log.debug(f" MISSING: {market}/{region}/{tk}/{interval}")
    log.info(f"Summary -> downloaded={total}, missing/empty={missing}")
