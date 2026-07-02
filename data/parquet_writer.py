import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PARQUET_PATH

log = logging.getLogger(__name__)

def write_parquet(scaled: dict) -> None:

    log.info(f"Assembeling single Parquet file: {PARQUET_PATH}")
    all_dfs = []

    for market, regions in scaled.items():
        for region, tickers in regions.items():
            for ticker, intervals in tickers.items():
                for interval, df in intervals.items():
                    if df is None or df.empty:
                        continue

                    df_out = df.copy()
                    df_out.reset_index(inplace=True)
                    numeric_cols = [c for c in df_out.columns if c not in ("ticker", "market", "region", "interval", "datetime")]
                    df_out[numeric_cols] = df_out[numeric_cols].apply(pd.to_numeric, errors="coerce")
                    df_out["datetime"] = pd.to_datetime(df_out["datetime"], utc=True).astype(str)

                    df_out.insert(0, "ticker", ticker)
                    df_out.insert(1, "market", market)
                    df_out.insert(2, "region", region)
                    df_out.insert(3, "interval", interval)

                    all_dfs.append(df_out)

    if not all_dfs:
        raise ValueError("No data to write to Parquet")

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Total rows to write: {len(combined):,} | columns: {list(combined.columns)}")

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, str(PARQUET_PATH), compression="snappy", row_group_size=100_000,)
    log.info(f"Parquet writing completed: {PARQUET_PATH} | size: {PARQUET_PATH.stat().st_size / 1e6:.1f} MB")

def read_parquet(ticker: str | None = None, market: str | None = None, interval: str | None = None, region: str | None = None) -> pd.DataFrame:

    filters = []
    if ticker:
        filters.append(("ticker", "=", ticker))
    if market:
        filters.append(("market", "=", market))
    if interval:
        filters.append(("interval", "=", interval))
    if region:
        filters.append(("region", "=", region))

    dataset = pq.read_table(str(PARQUET_PATH), filters=filters if filters else None,)
    df = dataset.to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df
