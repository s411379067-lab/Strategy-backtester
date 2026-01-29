import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def rolling_high(df: pd.DataFrame, length: int) -> pd.Series:
    return df["high"].rolling(length, min_periods=length).max()

def rolling_low(df: pd.DataFrame, length: int) -> pd.Series:
    return df["low"].rolling(length, min_periods=length).min()
