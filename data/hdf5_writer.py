import logging
import pandas as pd
import re
from typing import cast, Any

from .config import HDF5_PATH

log = logging.getLogger(__name__)

def _sanitize_key(s: str) -> str:

    s = re.sub(r"[^A-Za-z0-9_/]", "_", s)
    parts = s.split("/")
    parts = [f"t_{p}" if p and p[0].isdigit() else p for p in parts]
    return "/".join(parts)

def _raw_key(market: str, region: str, ticker: str, interval: str) -> str:
    return f"{market}/{region}/{ticker}/{interval}"

def tickers_to_hdf5_key(market: str, region: str, ticker: str, interval: str) -> str:
    raw_key = _raw_key(market, region, ticker, interval)
    return _sanitize_key(raw_key)

def write_hdf5(cleaned: dict) -> None:

    manifest_rows = []

    log.info(f"Writing HDF5: {HDF5_PATH}")

    with pd.HDFStore(str(HDF5_PATH), mode='w', complevel=5, complib="blosc") as store:
        total = 0
        for market, regions in cleaned.items():
            for region, tickers in regions.items():
                for ticker, intervals in tickers.items():
                    for interval, df in intervals.items():
                        if df is None or df.empty:
                            continue
                        hdf_key = tickers_to_hdf5_key(market, region, ticker, interval)

                        try:
                            store.put(hdf_key, df, format='table', data_columns=True, complevel=5, complib="blosc")

                            print(f"[DEBUG hdf5] writing key={hdf_key} shape={df.shape} nan={df.isna().sum().sum()}")

                            storer = cast(Any, store).get_storer(hdf_key)
                            if storer is not None:
                                storer.attrs.metadata = {
                                    "market": market,
                                    "region": region,
                                    "ticker": ticker,
                                    "interval": interval,
                                    "rows": len(df),
                                    "start": str(df.index[0]),
                                    "end": str(df.index[-1]),
                                }

                            manifest_rows.append({
                                "market": market,
                                "region": region,
                                "ticker": ticker,
                                "interval": interval,
                                "hdf_key": hdf_key,
                                "rows": len(df),
                                "start": str(df.index[0]),
                                "end": str(df.index[-1]),
                            })
                            total += 1
                            log.debug(f"{hdf_key} ({len(df)} rows)")
                        except Exception as e:
                            log.error(f"Failed to write {hdf_key}: {e}")

        if manifest_rows:
            manifest_df = pd.DataFrame(manifest_rows)
            store.put("/manifest", manifest_df, format='table', data_columns=True)
            log.info(f"Manifest written with {len(manifest_df)} entries")

    log.info(f"HDF5 writing completed: {total} datasets written to {HDF5_PATH}")


def read_hdf5(market: str, region: str, ticker: str, interval: str) -> pd.DataFrame | None:
    key = tickers_to_hdf5_key(market, region, ticker, interval)
    try:
        with pd.HDFStore(str(HDF5_PATH), mode='r') as store:
            if key in store:
                df = cast(pd.DataFrame, store[key])
                return df
            else:
                log.warning(f"Key {key} not found in HDF5 store")
                return None
    except Exception as e:
        log.error(f"Failed to read {key} from HDF5: {e}")
        return None


def list_hdf5_keys() -> pd.DataFrame:
    try:
        with pd.HDFStore(str(HDF5_PATH), mode='r') as store:
            if "/manifest" in store:
                manifest_df = cast(pd.DataFrame, store["/manifest"])
                return manifest_df
    except Exception as e:
        log.error(f"Failed to read manifest from HDF5: {e}")
    return pd.DataFrame()
