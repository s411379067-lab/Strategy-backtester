import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def rolling_high(df: pd.DataFrame, length: int, column: str = "high") -> pd.Series:
    return df[column].rolling(length, min_periods=length).max()

def rolling_low(df: pd.DataFrame, length: int, column: str = "low") -> pd.Series:
    return df[column].rolling(length, min_periods=length).min()


class IndicatorRegistry:
    def rolling_high(self, df: pd.DataFrame, length: int, column: str = "high") -> pd.Series:
        return rolling_high(df, length, column)

    def rolling_low(self, df: pd.DataFrame, length: int, column: str = "low") -> pd.Series:
        return rolling_low(df, length, column)

    def atr(self, df: pd.DataFrame, length: int) -> pd.Series:
        return atr(df, length)
