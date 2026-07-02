import logging
import pandas as pd

from .config import MIN_ROWS

log = logging.getLogger(__name__)

def _remove_outliers_iqr(df: pd.DataFrame, cols: list[str], factor: float = 5.0) -> pd.DataFrame:

    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= df[col].between(lower, upper)
    removed = (~mask).sum()
    if removed > 0:
        log.debug(f"Outlier removal: dropped {removed} rows")
    return df[mask]

def clean_data(df: pd.DataFrame, ticker: str, interval: str, market: str) -> pd.DataFrame | None:

    if df is None or df.empty:
        return None

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            log.warning(f"{ticker} @ {interval}: index parse failed: {e}")
            return None
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

    df.index.name = "datetime"

    df.columns = [col.lower().strip() for col in df.columns]

    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            if col == 'volume':
                df['volume'] = 0.0
            else:
                log.warning(f"{ticker} @ {interval}: missing column '{col}'")
                return None

    df = df[required]

    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_dup = df.index.duplicated().sum()
    if n_dup:
        log.debug(f"{ticker} @ {interval}: removing {n_dup} duplicate timestamps")
        df = df[~df.index.duplicated(keep='last')]

    df.sort_index(inplace=True)

    price_cols = ['open', 'high', 'low', 'close']
    bad_price = (df[price_cols] <= 0).any(axis=1)
    if bad_price.sum():
        log.debug(f"{ticker} @ {interval}: dropping {bad_price.sum()} zero/negative price rows")
        df = df[~bad_price]

    bad_h = df['high'] < df['low']
    if bad_h.sum():
        log.debug(f"{ticker} @ {interval}: dropping {bad_h.sum()} rows with high < low")
        df = df[~bad_h]

    bad_close = (df['close'] < df['low']) | (df['close'] > df['high'])
    if bad_close.sum():
        log.debug(f"{ticker} @ {interval}: dropping {bad_close.sum()} close-out-of-range rows")
        df = df[~bad_close]

    bad_open = (df['open'] < df['low']) | (df['open'] > df['high'])
    if bad_open.sum():
        log.debug(f"{ticker} @ {interval}: dropping {bad_open.sum()} open-out-of-range rows")
        df = df[~bad_open]

    if market in ("FOREX", "INDICES"):
        df["volume"] = 0.0
    else:
        df["volume"] = df["volume"].clip(lower=0.0)

    df = df.ffill(limit=5).bfill(limit=2)
    n_nan = df.isna().any(axis=1).sum()
    if n_nan:
        log.debug(f"{ticker} @ {interval}: dropping {n_nan} rows with NaN values after ffill")
        df = df.dropna()

    df = _remove_outliers_iqr(df, price_cols, factor=5.0)

    if len(df) < MIN_ROWS:
        log.warning(f"{ticker} @ {interval}: only {len(df)} rows after cleaning(< {MIN_ROWS}), skipping")
        return None
    log.debug(f"{ticker} @ {interval}: {len(df)} rows after cleaning")

    return df

def clean_all(raw: dict) -> dict:
    cleaned = {}
    ok = err = 0
    for market, regions in raw.items():
        cleaned[market] = {}
        for region, tickers in regions.items():
            cleaned[market][region] = {}
            for ticker, intervals in tickers.items():
                cleaned[market][region][ticker] = {}
                for interval, df in intervals.items():
                    try:
                        result = clean_data(df, ticker, interval, market)
                    except Exception as e:
                        log.error(f"{ticker} @ {interval}: Error occurred while cleaning data: {e}")
                        result = None
                    if result is not None:
                        cleaned[market][region][ticker][interval] = result
                        ok += 1
                    else:
                        err += 1
    log.info(f"Data cleaning completed: {ok} cleaned, {err} dropped")
    return cleaned
