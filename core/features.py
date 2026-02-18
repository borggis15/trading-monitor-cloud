from __future__ import annotations
import numpy as np
import pandas as pd

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def zscore(s: pd.Series, window: int = 48) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd.replace(0, np.nan)

def compute_features(bars: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    df["ema_fast"] = ema(df["close"], 20)
    df["ema_slow"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["zscore"] = zscore(df["close"], 48)
    df["ret_fwd"] = df["close"].shift(-horizon_bars) / df["close"] - 1.0
    return df[["rsi","ema_fast","ema_slow","atr","zscore","ret_fwd"]]
