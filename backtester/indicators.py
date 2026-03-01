import pandas as pd
import numpy as np
import talib
from typing import Tuple
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

def body_bar_ratio(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    ratio = body / range_
    return ratio

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

def bband(df: pd.DataFrame, length: int, column: str = "close", num_stddev: float = 2.0) -> pd.DataFrame:
    upper, middle, lower = talib.BBANDS(df[column], timeperiod=length, nbdevup=num_stddev, nbdevdn=num_stddev)
    width = upper - lower
    width_pct = width / middle * 100
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower, "width": width, "width_pct": width_pct}, index=df.index)

def cross_bband_mid_count(df: pd.DataFrame, length: int, column: str = "close", side: str = "") -> pd.Series:
    """
    計算價格向上/下穿越布林中軸的次數（side="up" 或 "down" 或 "sum" 或 "net"）
    up: 只計算向上穿越次數
    down: 只計算向下穿越次數
    sum: 向上穿越次數 + 向下穿越次數
    net: 向上穿越次數 - 向下穿越次數
    """
    bband_df = bband(df, length, column)
    mid = bband_df["middle"]
    price = df[column]

    cross_up = (price > mid) & (price.shift(1) <= mid.shift(1))
    cross_down = (price < mid) & (price.shift(1) >= mid.shift(1))

    up = cross_up.astype(int).rolling(length, min_periods=length).sum()
    down = cross_down.astype(int).rolling(length, min_periods=length).sum()

    if side == "up":
        return up
    if side == "down":
        return down
    if side == "sum":
        return up + down
    if side == "net":
        return up - down
    raise ValueError(f"Invalid side: {side}")

def close_loc(df: pd.DataFrame, side: str) -> pd.Series:
    if side == "bull":
        return (df["close"] - df["low"]) / (df["high"] - df["low"])
    elif side == "bear":
        return (df["high"] - df["close"]) / (df["high"] - df["low"])
    else:
        raise ValueError(f"Invalid side: {side}")

def K_bar_score(
    df: pd.DataFrame,
    BBR_thresholds: Tuple[float, float] = lambda: (0.4, 0.7),
    close_loc_thresholds: Tuple[float, float] = lambda: (0.6, 0.8),
    push_thresholds: Tuple[float, float, float] = lambda: (0.5, 1.0, 1.5),
    overlap_thresholds: Tuple[float, float] = lambda: (0.3, 0.5),
) -> pd.DataFrame:
    """
    單根K線分數（bar_score），多頭為正、空頭為負。

    - body_ratio = abs(close-open)/(high-low)
      >0.4 => 0.5, >0.7 => 1.0
    - close_loc (bull) = (close-low)/(high-low)
      >0.6 => 0.5, >0.8 => 1.0
      close_loc (bear) = (high-close)/(high-low)
      >0.6 => -0.5, >0.8 => -1.0
    - push = (close-close[1])/ATR
      bull: >0.5 => 0.4, >1.0 => 0.8, >1.5 => 1.0
      bear: push < -0.5 => -0.4, < -1.0 => -0.8, < -1.5 => -1.0
    - overlap_ratio = overlap_range / range, overlap_range = intersection(high/low with previous bar)
      overlap_ratio < 0.5 => 0.5, < 0.3 => 1.0（空頭同樣給負分）

    注意：這裡回傳的是「單根分數」，連續同向累加請用 K_run_score。
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    bbr1, bbr2 = sorted(BBR_thresholds)
    cl1, cl2 = sorted(close_loc_thresholds)
    p1, p2, p3 = sorted(push_thresholds)
    o_tight, o_loose = sorted(overlap_thresholds)  # tight(小) => 1.0, loose(大) => 0.5

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)

    rng = (high - low)
    rng_safe = rng.replace(0.0, np.nan)

    # --- features ---
    body_ratio = (close - open_).abs() / rng_safe
    bull_close_loc = (close - low) / rng_safe
    bear_close_loc = (high - close) / rng_safe

    atr_14 = atr(df, 14).astype(float).replace(0.0, np.nan)  # 你已有 atr()
    push = (close - close.shift(1)) / atr_14

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    overlap_range = (pd.concat([high, prev_high], axis=1).min(axis=1) -
                     pd.concat([low, prev_low], axis=1).max(axis=1)).clip(lower=0.0)
    overlap_ratio = overlap_range / rng_safe

    # --- bucket helpers (vectorized) ---
    def two_step_pts(x: pd.Series, t1: float, t2: float, mid: float, top: float) -> np.ndarray:
        # >t2 => top, >t1 => mid, else 0
        xa = x.to_numpy(dtype=float)
        return np.where(xa > t2, top, np.where(xa > t1, mid, 0.0))

    def three_step_pts(x: pd.Series, t1: float, t2: float, t3: float, a: float, b: float, c: float) -> np.ndarray:
        # >t3 => c, >t2 => b, >t1 => a, else 0.5
        xa = x.to_numpy(dtype=float)
        return np.where(xa > t3, c, np.where(xa > t2, b, np.where(xa > t1, a, 1.0)))

    # --- points (bull side, positive) ---
    body_pts = two_step_pts(body_ratio, bbr1, bbr2, mid=0.5, top=1.0)
    bull_close_pts = two_step_pts(bull_close_loc, cl1, cl2, mid=0.5, top=1.0)
    bull_push_pts = three_step_pts(push, p1, p2, p3, a=1.3, b=1.6, c=2)

    # overlap: smaller is better
    ov = overlap_ratio.to_numpy(dtype=float)
    overlap_pts = np.where(ov < o_tight, 1.0, np.where(ov < o_loose, 0.5, 0.0))

    # --- bear side points (negative) ---
    bear_close_pts = two_step_pts(bear_close_loc, cl1, cl2, mid=0.5, top=1.0)
    bear_push_pts = three_step_pts((-push), p1, p2, p3, a=1.3, b=1.6, c=2)  # 下跌推進用 -push

    bull_mask = (close > open_).to_numpy()
    bear_mask = (close < open_).to_numpy()

    close_pts = np.where(bull_mask, bull_close_pts, np.where(bear_mask, bear_close_pts, 0.0))
    push_pts = np.where(bull_mask, bull_push_pts, np.where(bear_mask, bear_push_pts, 0.0))


    bull_score = (body_pts + bull_close_pts + overlap_pts) * bull_push_pts
    bear_score = -(body_pts + bear_close_pts + overlap_pts) * bear_push_pts

    score = np.zeros(len(df), dtype=float)
    score[bull_mask] = bull_score[bull_mask]
    score[bear_mask] = bear_score[bear_mask]
    # doji => 0

    s = pd.Series(score, index=df.index, name="bar_score").fillna(0.0)
    k_bar_score_df = pd.DataFrame({
        "body_pts": body_pts,
        "close_pts": close_pts,
        "push_pts": push_pts,
        "overlap_pts": overlap_pts,
        "bar_score": s
    }, index=df.index)
        
    return k_bar_score_df

def K_run_score(df: pd.DataFrame,     
    BBR_thresholds: Tuple[float, float] = lambda: (0.4, 0.7),
    close_loc_thresholds: Tuple[float, float] = lambda: (0.6, 0.8),
    push_thresholds: Tuple[float, float, float] = lambda: (0.5, 1.0, 1.5),
    overlap_thresholds: Tuple[float, float] = lambda: (0.3, 0.5)
    ) -> pd.DataFrame:
    """
    連續同向才累加，方向改變或 doji(0) 就歸零。
    - run_score 在同一段連續多頭/空頭內做 cumsum。
    """
    bar_score = K_bar_score(df, BBR_thresholds, close_loc_thresholds, push_thresholds, overlap_thresholds)["bar_score"]
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    dir_ = np.sign((close - open_).to_numpy())  # +1 bull, -1 bear, 0 doji
    dir_s = pd.Series(dir_, index=df.index)

    # 新段落：方向改變或 doji
    new_seg = (dir_s == 0) | (dir_s != dir_s.shift(1))
    seg_id = new_seg.cumsum()

    # doji 段直接 0；其他段在段內累加
    run = bar_score.where(dir_s != 0, 0.0).groupby(seg_id).cumsum()
    run = run.where(dir_s != 0, 0.0)
    run.name = "run_score"

    k_run_score_df = pd.DataFrame({
        "bar_score": bar_score,
        "seg_id": seg_id,
        "run_score": run
    }, index=df.index)
    return k_run_score_df

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
    def bband(self, df: pd.DataFrame, length: int, column: str = "close", num_stddev: float = 2.0) -> pd.DataFrame:
        return bband(df, length, column, num_stddev)
    def body_bar_ratio(self, df: pd.DataFrame) -> pd.Series:
        return body_bar_ratio(df)
    def cross_bband_mid_count(self, df: pd.DataFrame, length: int, column: str = "close", side: str = "") -> pd.Series:
        return cross_bband_mid_count(df, length, column, side)
    def close_loc(self, df: pd.DataFrame, side: str) -> pd.Series:
        return close_loc(df, side)
    def K_bar_score(self, df: pd.DataFrame, BBR_thresholds: Tuple[float, float] = lambda: (0.4, 0.7), close_loc_thresholds: Tuple[float, float] = lambda: (0.6, 0.8), push_thresholds: Tuple[float, float, float] = lambda: (0.5, 1.0, 1.5), overlap_thresholds: Tuple[float, float] = lambda: (0.3, 0.5)) -> pd.DataFrame:
        return K_bar_score(df, BBR_thresholds, close_loc_thresholds, push_thresholds, overlap_thresholds)
    def K_run_score(self, df: pd.DataFrame, BBR_thresholds: Tuple[float, float] = lambda: (0.4, 0.7), close_loc_thresholds: Tuple[float, float] = lambda: (0.6, 0.8), push_thresholds: Tuple[float, float, float] = lambda: (0.5, 1.0, 1.5), overlap_thresholds: Tuple[float, float] = lambda: (0.3, 0.5)) -> pd.DataFrame:
        return K_run_score(df, BBR_thresholds, close_loc_thresholds, push_thresholds, overlap_thresholds)
