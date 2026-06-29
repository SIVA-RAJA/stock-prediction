import logging
import pandas as pd
import numpy as np

from config import *

log = logging.getLogger(__name__)


def _sin_cos(series: pd.Series, period: float):
    return (
        np.sin(2 * np.pi * series / period),
        np.cos(2 * np.pi * series / period),
    )

def _rsi(close: pd.Series, period: int =14) -> pd.Series:

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()

def _stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k, d

def _williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    williams_r = -100 * (highest_high - close) / denom
    return williams_r

def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(close, window=20, n_std=2):
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, width, pct

def _obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv

def _vwap_rolling(high, low, close, volume, window=14):
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    vwap = tpv.rolling(window=window).sum() / volume.rolling(window=window).sum().replace(0, np.nan)
    return vwap


def _mfi(high, low, close, volume, period=14):
    typical = (high + low + close) / 3
    raw_money_flow = typical * volume
    positive_money_flow = raw_money_flow.where(typical > typical.shift(1), 0)
    negative_money_flow = raw_money_flow.where(typical < typical.shift(1), 0)

    positive_sum = positive_money_flow.rolling(window=period).sum()
    negative_sum = negative_money_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (positive_sum / negative_sum.replace(0, np.nan))))
    return mfi

def _cmf(high, low, close, volume, period=20):
    denom = (high - low).replace(0, np.nan)
    mvf = ((close - low) - (high - close)) / denom * volume
    cmf = mvf.rolling(window=period).sum() / volume.rolling(window=period).sum().replace(0, np.nan)
    return cmf

def _cci(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma) / (0.015 * mad.replace(0, np.nan))
    return cci

def _volume_zscore(volume, window=20):
    mean = volume.rolling(window=window).mean()
    std = volume.rolling(window=window).std()
    z_score = (volume - mean) / std.replace(0, np.nan)
    return z_score

def _window_normalize(df):
    base = df["close"].iloc[0]
    if base == 0 or np.isnan(base):
        base = 1.0
    price_cols = ["open", "high", "low", "close", "bb_upper", "bb_mid", "bb_lower", "vwap"] + [f"sma_{w}" for w in SMA_WINDOWS] + [f"ema_{w}" for w in EMA_WINDOWS] + [f"close_lag_{lag}" for lag in LAG_WINDOWS]

    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] / base
    return df

def _market_regime(close, window=20):
    ret = close.pct_change(window)
    std = close.pct_change().rolling(window).std()
    regime = np.where(ret > std, 1, np.where(ret < -std, -1, 0))
    return pd.Series(regime, index=close.index, dtype=np.float32)

def add_featurers(
        df: pd.DataFrame,
        ticker: str,
        market: str,
        region: str,
        interval: str,) -> pd.DataFrame | None:

    if df is None or df.empty:
        return None

    df = df.copy()
    open, high, low, close, volume = df['open'], df['high'], df['low'], df['close'], df['volume']

    df['ticker_id'] = TICKER_TO_ID.get(ticker, 0)
    df['market_id'] = MARKET_TO_ID.get(market, 0)
    df['region_id'] = REGION_TO_ID.get(region, 0)
    df["interval_id"] = INTERVAL_TO_ID.get(interval, 0)

    idx = pd.DatetimeIndex(df.index)
    df["sin_minute"], df["cos_minute"] = _sin_cos(pd.Series(idx.minute, index=idx), 60)
    df["sin_hour"], df["cos_hour"] = _sin_cos(pd.Series(idx.hour, index=idx), 24)
    df["sin_day_of_week"], df["cos_day_of_week"] = _sin_cos(pd.Series(idx.dayofweek, index=idx), 7)
    df["sin_day_of_month"], df["cos_day_of_month"] = _sin_cos(pd.Series(idx.day, index=idx), 31)
    df["sin_month"], df["cos_month"] = _sin_cos(pd.Series(idx.month, index=idx), 12)

    df["log_return"] = np.log(close / close.shift(1))
    df["pct_change"] = close.pct_change()
    df["hl_range"] = high - low
    df["co_range"] = close - open

    for w in SMA_WINDOWS:
        df[f"sma_{w}"] = close.rolling(window=w).mean()

    for w in EMA_WINDOWS:
        df[f"ema_{w}"] = close.ewm(span=w, min_periods=w).mean()

    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    df[f"rsi_{RSI_PERIOD}"] = _rsi(close, period=RSI_PERIOD)

    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, STOCH_WINDOW, STOCH_SMOOTH)

    df["williams_r"] = _williams_r(high, low, close, WILLIAMS_R_PERIOD)

    df[f"roc_{ROC_PERIOD}"] = close.pct_change(ROC_PERIOD) * 100

    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_width"], df["bb_pct"] = _bollinger(close, BB_WINDOW, BB_STD)

    df[f"atr_{ATR_PERIOD}"] = _atr(high, low, close, ATR_PERIOD)

    df[f"cci_{CCI_PERIOD}"] = _cci(high, low, close, CCI_PERIOD)

    df["obv"] = _obv(close, volume)

    df["vwap"] = _vwap_rolling(high, low, close, volume, VWAP_PERIOD)

    if market not in ("FOREX", "INDICES"):
        df[f"mfi_{MFI_PERIOD}"] = _mfi(high, low, close, volume, MFI_PERIOD)
    else:
        df[f"mfi_{MFI_PERIOD}"] = np.nan

    df[f"cmf_{CMF_PERIOD}"] = _cmf(high, low, close, volume, CMF_PERIOD)

    df["volume_zscore"] = _volume_zscore(volume, window=20)

    for lag in LAG_WINDOWS:
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    df["market_regime"] = _market_regime(close, window=MARKET_REGIME_WINDOW)

    df = _window_normalize(df)

    n_before = len(df)
    volume_derived = ["vwap", f"mfi_{MFI_PERIOD}", f"cmf_{CMF_PERIOD}", "volume_zscore"]
    if market in ("FOREX", "INDICES"):
        df[volume_derived] = df[volume_derived].fillna(0)
    df.dropna(inplace=True)
    n_after = len(df)
    log.debug(f"Feature warm-up dropped {n_before - n_after} rows -> {n_after} rows remaining")

    if n_after < MIN_ROWS:
        log.warning(f"{ticker} @ {market}: only {n_after} rows after feature warm-up (< {MIN_ROWS})")
        return None

    return df

def add_features_all(cleaned: dict) -> dict:
    featured = {}
    ok = skip = 0
    for market, regions in cleaned.items():
        featured[market] = {}
        for region, tickers in regions.items():
            featured[market][region] = {}
            for ticker, intervals in tickers.items():
                featured[market][region][ticker] = {}
                for interval, df in intervals.items():
                    result = add_featurers(df, ticker, market, region, interval)
                    if result is not None:
                        featured[market][region][ticker][interval] = result
                        ok += 1
                    else:
                        skip +=1
    log.info(f"Feature engineering completed: {ok} datasets processed, {skip} skipped")
    return featured
