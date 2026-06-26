import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from config import PARQUET_DIR

log = logging.getLogger(__name__)

def _partition_path(market: str, interval: str, region: str) -> Path:
    p = PARQUET_DIR / f"{market}" / f"{interval}" / f"{region}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_parquet(scaled: dict) -> None:

    log.info(f"Writing Parquet files to {PARQUET_DIR}")

    partitions: dict[tuple, list[pd.DataFrame]] = {}

    for market, regions in scaled.items():
        for region, tickers in regions.items():
            for ticker, intervals in tickers.items():
                for interval, df in intervals.items():
                    if df is None or df.empty:
                        continue

                    df_out = df.copy()
                    df_out.reset_index(inplace=True)
                    df_out["datetime"] = df_out["datetime"].astype(str)

                    df_out.insert(0, "ticker", ticker)
                    df_out.insert(1, "market", market)
                    df_out.insert(2, "region", region)
                    df_out.insert(3, "interval", interval)

                    key = (market, interval, region)
                    partitions.setdefault(key, []).append(df_out)

    total_files = 0
    total_rows = 0

    for (market, interval, region), dfs in partitions.items():
        combined = pd.concat(dfs, ignore_index=True)
        out_path = _partition_path(market, interval, region) / "part.0.parquet"

        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy", row_group_size=50_000,)

        total_files += 1
        total_rows += len(combined)
        log.debug(f"{out_path.relative_to(PARQUET_DIR)}: {len(combined)} rows written")
    log.info(f"Parquet writing completed: {total_files} files, {total_rows} total rows")

def read_parquet(market: str | None = None, interval: str | None = None, region: str | None = None) -> pd.DataFrame:

    filters = []
    if market:
        filters.append(("market", "=", market))
    if interval:
        filters.append(("interval", "=", interval))
    if region:
        filters.append(("region", "=", region))

    dataset = pq.read_table(str(PARQUET_DIR), filters=filters if filters else None,)
    df = dataset.to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df
