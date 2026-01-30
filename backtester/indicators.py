import pandas as pd
import numpy as np
import talib
def atr(df: pd.DataFrame, length: int) -> pd.Series:
    atr_values = talib.ATR(df["high"], df["low"], df["close"], timeperiod=length)
    return pd.Series(atr_values, index=df.index)

def rolling_high(df: pd.DataFrame, length: int, column: str = "high") -> pd.Series:
    return df[column].rolling(length, min_periods=length).max()

def rolling_low(df: pd.DataFrame, length: int, column: str = "low") -> pd.Series:
    return df[column].rolling(length, min_periods=length).min()

def bar_side(df: pd.DataFrame) -> pd.Series:
    body = df["close"] - df["open"]
    side = np.where(body > 0, 1, np.where(body < 0, -1, 0))
    return pd.Series(side, index=df.index)

def rocp(df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
    rocp_values = talib.ROCP(df[column], timeperiod=length)
    return pd.Series(rocp_values, index=df.index)

def bar_range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]

def bar_range_pct(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]) / df["close"]

def bar_body_range(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()

def bar_body_range_pct(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs() / df["close"]

def ma(df: pd.DataFrame, length: int, column: str = "close", ma_type: str = "SMA") -> pd.Series:
    if ma_type == "SMA":
        sma_values = talib.SMA(df[column], timeperiod=length)
        return pd.Series(sma_values, index=df.index)
    elif ma_type == "EMA":
        ema_values = talib.EMA(df[column], timeperiod=length)
        return pd.Series(ema_values, index=df.index)
    else:
        raise ValueError(f"Unsupported ma_type: {ma_type}")
def bar_side_sum(df: pd.DataFrame, length: int) -> pd.Series:
    side = bar_side(df)
    return side.rolling(length, min_periods=length).sum()
def body_strictly_increasing(df: pd.DataFrame, n: int) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    cond = pd.Series(True, index=df.index)
    for j in range(n-1):
        cond &= body.shift(j) > body.shift(j+1)
    # 前 n-1 根不足資料 -> False（避免 NaN 讓結果變成不確定）
    cond = cond.fillna(False)
    return cond




class IndicatorRegistry:
    def rolling_high(self, df: pd.DataFrame, length: int, column: str = "high") -> pd.Series:
        return rolling_high(df, length, column)

    def rolling_low(self, df: pd.DataFrame, length: int, column: str = "low") -> pd.Series:
        return rolling_low(df, length, column)

    def atr(self, df: pd.DataFrame, length: int) -> pd.Series:
        return atr(df, length)
