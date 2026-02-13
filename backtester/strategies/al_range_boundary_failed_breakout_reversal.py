from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ALRangeBoundaryFailedBreakoutReversalParams:
    """
    Parameters for ALRangeBoundaryFailedBreakoutReversalStrategy
    
    Attributes:
    er_n (int): Efficiency Ratio 計算K線糾結程度
    er_ma_n (int): Efficiency Ratio 移動平均平滑參數
    er_threshold (float): Efficiency Ratio 糾結閾值，低於此值表示K線糾結
    ema_n (int): 移動平均線長度
    slope_threshold (float): 斜率區分TR/BC閾值
    min_sl_range_pct (float): 最小停損距離百分比
    max_notional_pct (float): 最大可承受損失資金百分比
    min_qty (float): 最小下單數量
    rr (float): 風險報酬比率，停利為停損距離的倍數(或幾倍std距離SL)
    time_exit_bars (int): 時間出場K線數
    allow_side (Optional[Side]): 允許的交易方向，None表示雙向，Side.LONG表示只做多，Side.SHORT表示只做空
    sizing_equity_base (SizingEquityBase): 計算下單資金基礎，初始資金或當前資金
    """
    er_n: int = 40
    er_ma_n: int = 3
    er_threshold: float = 0.2  # K線糾結閾值
    ema_n: int = 20
    slope_threshold: float = 0.1  # 斜率區分TR/BC閾值
    rolling_hh_ll_n: int = 20
    min_boundary_range_pct: float = 1  # 區間邊界最小距離百分比（避免邊界太近造成頻繁失敗訊號）
    max_boundary_range_pct: float = 1  # 區間邊界最大距離百分比
    min_sl_range_pct: float = 0.3  # 最小停損距離百分比
    tp_use_mid_of_range: bool = False  # 停利用區間中點
    max_notional_pct: float = 1.0
    min_qty: float = 0.001
    rr: float = 1.0           # TP = SL距離 * rr
    time_exit_bars: int = 50
    allow_side: Optional[Side] = None  # None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空
    sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL
class ALRangeBoundaryFailedBreakoutReversalStrategy(Strategy):
    """
    Docstring for ALRangeBoundaryFailedBreakoutReversalStrategy
    
    策略邏輯：
    1. 偵測K線效率低(糾結區間)
    2. 以rolling high、low檢視是否更新區間上下邊界
    3. 若有主動更新邊界(非因為rolling window才更新)，等待3根K線內是否有反向K線(突破失敗訊號)
    4. 若有突破失敗訊號，等待下一根K線是否觸及計畫進場價(空=上邊界、多=下邊界)
    5. 進場後，停損設在訊號K線的另一邊，停利設在同等距離的另一邊*n_rr



    """

    def __init__(self, params: ALRangeBoundaryFailedBreakoutReversalParams) -> None:
        self.p = params
        self.state = None  # 可用來記錄策略狀態
        self.touch_upper_boundary_i: Optional[int] = None
        self.long_sb_i: Optional[int] = None
        self.short_sb_i: Optional[int] = None
        self.long_plan_entry_price: Optional[float] = None
        self.short_plan_entry_price: Optional[float] = None
        self.touch_lower_boundary_i: Optional[int] = None
        self.initial_sl_price: Optional[float] = None


    def required_indicators(self) -> Dict[str, Any]:
        return {
            "er": ("efficiency_ratio", self.p.er_n, self.p.er_ma_n, "close"),
            "rolling_high": ("rolling_high", self.p.rolling_hh_ll_n, "high"),
            "rolling_low": ("rolling_low", self.p.rolling_hh_ll_n, "low"),
            "ema": ("ma", self.p.ema_n, "close", "EMA"),
        }

    def generate_intents(self, ctx: StrategyContext) -> List[OrderIntent]:
        i = ctx.i
        df = ctx.df
        pos = ctx.position
        init_equity = ctx.init_equity
        now_equity = ctx.now_equity
        if self.p.sizing_equity_base == SizingEquityBase.INITIAL:
            base_equity = init_equity
        elif self.p.sizing_equity_base == SizingEquityBase.CURRENT:
            base_equity = now_equity
        intents: List[OrderIntent] = []

        open_series = df["open"]
        close_series = df["close"]
        low_series = df["low"]
        high_series = df["high"]



        # 若有倉，只更新出場線（也可以不更新）
        if pos.side is not None and pos.qty > 0:
            
            return intents
        if i < 1:
            return intents
        

        er_series = ctx.indicators["er"]
        ema_series = ctx.indicators["ema"]
        rolling_high_series = ctx.indicators["rolling_high"]
        rolling_low_series = ctx.indicators["rolling_low"]



        ## 亞、歐、美盤前盤後不做
        time_hour = df.index[i].hour
        cond_time_filter = time_hour not in [25]#[8,9,13,14,19,20,21,22,23,0,1]




        
        # 無倉，檢查進場
        if pos.side is None or pos.qty == 0:
            # 沒有倉位時，檢查狀態
            self.initial_sl_price = None

            ## 1. 一開始不管state，檢查有沒有K線糾結
            # 在糾結區間
            cond_er = er_series.iat[i] < self.p.er_threshold
            # 更新上邊界
            cond_update_upper_boundary = high_series.iat[i] > rolling_high_series.iat[i-1]
            # 更新下邊界
            cond_update_lower_boundary = low_series.iat[i] < rolling_low_series.iat[i-1]
            if cond_er and cond_update_upper_boundary:
                self.touch_upper_boundary_i = i
                self.state = "UPPERBO"
            elif cond_er and cond_update_lower_boundary:
                self.touch_lower_boundary_i = i
                self.state = "LOWERBO"
            ## 2. 有state，檢查是否突破失敗
            if self.state == "UPPERBO":
                if i <= self.touch_upper_boundary_i + 3:
                    # 檢查反向K線
                    cond_bar_reversal = close_series.iat[i] < open_series.iat[i]
                    if cond_bar_reversal:
                        self.short_sb_i = i
                        self.state = "SHORT_SB"
                        self.short_plan_entry_price = low_series.iat[self.short_sb_i]
                else:
                    self.state = None
            elif self.state == "LOWERBO":
                if i <= self.touch_lower_boundary_i + 3:
                    # 檢查反向K線
                    cond_bar_reversal = close_series.iat[i] > open_series.iat[i]
                    if cond_bar_reversal:
                        self.long_sb_i = i
                        self.state = "LONG_SB"
                        self.long_plan_entry_price = high_series.iat[self.long_sb_i]
                else:
                    self.state = None
            ## 3. 有突破失敗訊號，檢查進場條件
            
            if self.state == "SHORT_SB":
                # 剛好是SB後一根K線
                cond_shrot1 = i == self.short_sb_i + 1
                # 價格有觸及計畫進場價
                cond_short2 = low_series.iat[i] <= self.short_plan_entry_price
                # 允許做空
                cond_short3 = (self.p.allow_side is None) or (self.p.allow_side == Side.SHORT)
                # 上下邊界距離足夠（避免邊界太近造成頻繁失敗訊號）
                cond_short4a = (rolling_high_series.iat[self.short_sb_i] - rolling_low_series.iat[self.short_sb_i]) >= (self.p.min_boundary_range_pct/100) * rolling_low_series.iat[self.short_sb_i]
                # 上下邊界距離不超過最大範圍
                cond_short4b = (rolling_high_series.iat[self.short_sb_i] - rolling_low_series.iat[self.short_sb_i]) <= (self.p.max_boundary_range_pct/100) * rolling_low_series.iat[self.short_sb_i]
                # 最小停損距離足夠
                cond_short5 = (self.short_plan_entry_price - low_series.iat[self.short_sb_i]) >= (self.p.min_sl_range_pct/100) * self.short_plan_entry_price
                # 斜率條件
                slope = ((high_series.iat[i-10] - low_series.iat[i]) / high_series.iat[i-10])/10
                cond_slope = abs(slope) < self.p.slope_threshold
                if cond_shrot1 and cond_short2 and cond_short3 and cond_short4a and cond_short4b and cond_short5 and cond_slope:
                    # 進場空單
                    entry_price = min(self.short_plan_entry_price, open_series.iat[i])
                    sl_price = high_series.iat[self.short_sb_i]
                    sl_distance = sl_price - entry_price
                    if self.p.tp_use_mid_of_range:
                        range_mid = (rolling_high_series.iat[self.short_sb_i] + rolling_low_series.iat[self.short_sb_i]) / 2
                        tp_rr_pirce = entry_price - sl_distance * self.p.rr
                        tp_price = max(range_mid, tp_rr_pirce)
                    else:
                        tp_price = entry_price - sl_distance * self.p.rr

                    # 計算下單數量
                    max_lose = base_equity * (self.p.max_notional_pct/100)
                    qty = max_lose / sl_distance
                    intents.append(
                        OrderIntent(
                            action=ActionType.ENTRY,
                            side=Side.SHORT,
                            qty=max(qty, self.p.min_qty),
                            entry_price=entry_price,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            priority=100,
                        )
                    )
                    # 進場後重置狀態
                    self.state = None
            elif self.state == "LONG_SB":
                # 剛好是SB後一根K線
                cond_long1 = i == self.long_sb_i + 1
                # 價格有觸及計畫進場價
                cond_long2 = high_series.iat[i] >= self.long_plan_entry_price
                # 允許做多
                cond_long3 = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)
                # 上下邊界距離足夠（避免邊界太近造成頻繁失敗訊號）
                cond_long4a = (rolling_high_series.iat[self.long_sb_i] - rolling_low_series.iat[self.long_sb_i]) >= (self.p.min_boundary_range_pct/100) * rolling_low_series.iat[self.long_sb_i]
                # 上下邊界距離不超過最大範圍
                cond_long4b = (rolling_high_series.iat[self.long_sb_i] - rolling_low_series.iat[self.long_sb_i]) <= (self.p.max_boundary_range_pct/100) * rolling_low_series.iat[self.long_sb_i]
                # 最小停損距離足夠
                cond_long5 = (self.long_plan_entry_price - low_series.iat[self.long_sb_i]) >= (self.p.min_sl_range_pct/100) * self.long_plan_entry_price
                # 斜率條件
                slope = ((high_series.iat[i] - low_series.iat[i-10]) / low_series.iat[i-10])/10
                cond_slope = abs(slope) < self.p.slope_threshold
                if cond_long1 and cond_long2 and cond_long3 and cond_long4a and cond_long4b and cond_long5 and cond_slope:
                    # 進場多單
                    entry_price = max(self.long_plan_entry_price, open_series.iat[i])
                    sl_price = low_series.iat[self.long_sb_i]
                    sl_distance = entry_price - sl_price
                    if self.p.tp_use_mid_of_range:
                        range_mid = (rolling_high_series.iat[self.long_sb_i] + rolling_low_series.iat[self.long_sb_i]) / 2
                        tp_rr_pirce = entry_price + sl_distance * self.p.rr
                        tp_price = min(range_mid, tp_rr_pirce)
                    else:
                        tp_price = entry_price + sl_distance * self.p.rr

                    # 計算下單數量
                    max_lose = base_equity * (self.p.max_notional_pct/100)
                    qty = max_lose / sl_distance
                    intents.append(
                        OrderIntent(
                            action=ActionType.ENTRY,
                            side=Side.LONG,
                            qty=max(qty, self.p.min_qty),
                            entry_price=entry_price,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            priority=100,
                        )
                    )
                    # 進場後重置狀態
                    self.state = None
        return intents






                


                
