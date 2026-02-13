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
    ## 嚴格版本：每根都比前一根大
    # for j in range(n-1):
    #     cond &= body.shift(j) > body.shift(j+1)
    ## 寬鬆版本：只要最後一根最大，前面 n-1 根不要求嚴格遞增
    for j in range(n-1):
        cond &= body > body.shift(j+1)
    # 前 n-1 根不足資料 -> False（避免 NaN 讓結果變成不確定）
    cond = cond.fillna(False)
    return cond


def bar_streak(df: pd.DataFrame, doji: str = "reset") -> pd.Series:
    """
    計算連續陽線/陰線根數（signed streak）
    - 陽線連續：+1, +2, +3...
    - 陰線連續：-1, -2, -3...
    - 0：十字線或無連續

    doji:
      - "reset": 十字線視為打斷，該根 streak=0
      - "ignore": 十字線不改變 streak（維持上一根的 streak）
    """
    s = bar_side(df).astype("int64")

    if doji not in ("reset", "ignore"):
        raise ValueError("doji must be 'reset' or 'ignore'")

    if doji == "reset":
        # 每次 side 改變就分段，段內用 cumcount 計數
        grp = (s != s.shift()).cumsum()
        cnt = s.groupby(grp).cumcount() + 1
        streak = cnt * s
        # 十字線段結果自然是 0，但保險起見明確處理
        streak = streak.where(s != 0, 0).astype("int64")
        return streak

    # doji == "ignore"
    # 只在非 0 的 K 線上計算 streak，最後把 doji 用前值填回（維持不變）
    nz = s != 0
    s_nz = s[nz]

    if s_nz.empty:
        return pd.Series(0, index=s.index, name="streak", dtype="int64")

    grp = (s_nz != s_nz.shift()).cumsum()
    cnt = s_nz.groupby(grp).cumcount() + 1
    streak_nz = (cnt * s_nz).astype("int64")

    streak = pd.Series(np.nan, index=s.index, name="streak")
    streak.loc[nz] = streak_nz
    streak = streak.ffill().fillna(0).astype("int64")
    return streak

def hh_ll_check(df: pd.DataFrame, side: str = "hh") -> pd.DataFrame:
    if side == "hh":
        has_bar_hh = df["high"] > df["high"].shift(1)
        return has_bar_hh.fillna(False)
    elif side == "ll":
        has_bar_ll = df["low"] < df["low"].shift(1)
        return has_bar_ll.fillna(False)
    else:
        raise ValueError("side must be 'hh' or 'll'")
    
def hh_ll_streak(df: pd.DataFrame, side: str = "hh") -> pd.Series:
    has_hh_ll = hh_ll_check(df, side)
    grp = (has_hh_ll != has_hh_ll.shift()).cumsum()
    cnt = has_hh_ll.groupby(grp).cumcount() + 1
    streak = cnt * has_hh_ll.astype("int64")
    streak = streak.where(has_hh_ll, 0).astype("int64")
    return streak

def pivot_mask(df: pd.DataFrame, side: str = "low", k: int = 2) -> pd.Series:
    w = 2 * k + 1
    m = df[side].rolling(w, center=True).min()
    mask = df[side] == m
    pivot_mask_series = mask.fillna(False)
    return pivot_mask_series

def efficiency_ratio(df: pd.DataFrame, length: int, ema_length: int, column: str = "close") -> pd.Series:
    change = df[column].diff(length).abs()
    volatility = df[column].diff().abs().rolling(length).sum()
    er = change / volatility
    er = er.fillna(0.0)
    er_ma = talib.EMA(er, timeperiod=ema_length)
    er = pd.Series(er_ma, index=df.index)
    return er

def liner_regression_mid(df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
    mid = talib.LINEARREG(df[column], timeperiod=length)            # 回歸線在「當前這根」的 y 值（x=n-1）
    return pd.Series(mid, index=df.index)

def liner_regression_residuals(df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
    mid = liner_regression_mid(df, length, column)
    residuals = df[column] - mid
    return residuals
    
def bar_regression_residuals_std(df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
    residuals = liner_regression_residuals(df, length, column)
    std = residuals.rolling(length, min_periods=length).std()
    return std

class IndicatorRegistry:
    def rolling_high(self, df: pd.DataFrame, length: int, column: str = "high") -> pd.Series:
        return rolling_high(df, length, column)
    def rolling_low(self, df: pd.DataFrame, length: int, column: str = "low") -> pd.Series:
        return rolling_low(df, length, column)
    def atr(self, df: pd.DataFrame, length: int) -> pd.Series:
        return atr(df, length)
    def bar_side(self, df: pd.DataFrame) -> pd.Series:
        return bar_side(df)
    def rocp(self, df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
        return rocp(df, length, column)
    def bar_range(self, df: pd.DataFrame) -> pd.Series:
        return bar_range(df)
    def bar_range_pct(self, df: pd.DataFrame) -> pd.Series:
        return bar_range_pct(df)
    def bar_body_range(self, df: pd.DataFrame) -> pd.Series:
        return bar_body_range(df)
    def bar_body_range_pct(self, df: pd.DataFrame) -> pd.Series:
        return bar_body_range_pct(df)
    def ma(self, df: pd.DataFrame, length: int, column: str = "close", ma_type: str = "SMA") -> pd.Series:
        return ma(df, length, column, ma_type)
    def bar_side_sum(self, df: pd.DataFrame, length: int) -> pd.Series:
        return bar_side_sum(df, length)
    def body_strictly_increasing(self, df: pd.DataFrame, n: int) -> pd.Series:
        return body_strictly_increasing(df, n)
    def bar_streak(self, df: pd.DataFrame, doji: str = "reset") -> pd.Series:
        return bar_streak(df, doji)
    def hh_ll_check(self, df: pd.DataFrame, side: str = "hh") -> pd.DataFrame:
        return hh_ll_check(df, side)
    def hh_ll_streak(self, df: pd.DataFrame, side: str = "hh") -> pd.Series:
        return hh_ll_streak(df, side)
    def pivot_mask(self, df: pd.DataFrame, side: str = "low", k: int = 2) -> pd.Series:
        return pivot_mask(df, side, k)
    def efficiency_ratio(self, df: pd.DataFrame, length: int, ema_length: int, column: str = "close") -> pd.Series:
        return efficiency_ratio(df, length, ema_length, column)
    def liner_regression_mid(self, df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
        return liner_regression_mid(df, length, column)
    def liner_regression_residuals(self, df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
        return liner_regression_residuals(df, length, column)
    def bar_regression_residuals_std(self, df: pd.DataFrame, length: int, column: str = "close") -> pd.Series:
        return bar_regression_residuals_std(df, length, column)

