from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class TrandingRangeBreakoutParams:
    """參數說明：
    - trading_range_breakout_level_n: 計算突破的價格水平用的n（例如前n根K線的最高價或最低價）
    - er_n: 計算Efficiency Ratio用的n
    - er_ma_n: 計算Efficiency Ratio用的移動平均n
    - er_threshold: TR內effect ratio的門檻
    - bbands_n: 計算布林通道的n
    - bbands_width_pct_threshold: 布林通道寬度的門檻（%）
    - min_cross_bband_mid_count: TR內至少要有幾次K線突破布林中軌
    - min_follow_through_bars: 突破後至少要有幾根K線符合跟進條件（不包含突破K線）
    - breakout_bar_BBR_threshold: 突破K線的實體占比
    - min_breakout_penetration_pct: 突破K線的收盤價需突破突破水平多少%(TR的%)才算有效突破

    - sl_chosen_by: 停損價的選擇方式，"breakout_bar_low"表示以突破K線的最低價作為停損價，"TR_low"表示以TR的最低價作為停損價

    
    """
    trading_range_breakout_level_n: int = 20
    er_n: int = 40
    er_ma_n: int = 5
    er_threshold: float = 0.1
    bbands_n: int = 20
    bbands_width_pct_threshold: List[float] = lambda: [0, 2.0]
    min_cross_bband_mid_count: int = 4 
    min_follow_through_bars: int = 0 
    breakout_bar_BBR_threshold: float = 0.5
    min_breakout_penetration_pct: float = 50

    equity_max_loss_pct: float = 1.0
    min_qty: float = 0.001

    rr: float = 2.0           # TP = SL距離 * rr
    sl_chosen_by: str = "breakout_bar_low"  # "breakout_bar_low" or "TR_low"
    tp_based_by: str = "entry_price"  # "entry_price" or "sl_distance"
    time_exit_bars: int = 50
    allow_side: Optional[Side] = None  # None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空
    sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL
    



class TradingRangeBreakoutStrategy(Strategy):
    def __init__(self, params: TrandingRangeBreakoutParams) -> None:
        self.p = params
        self.state = None  # 可用來記錄策略狀態
        self.pb_start_i = None  # 紀錄pullback開始的index
        self.tp_changed = False  # 是否已經移動過停利
        self.last_hh = None # 紀錄最近一次的最高點


    def required_indicators(self) -> Dict[str, Any]:
        return {
            "bull_breakout_level": ("rolling_high", self.p.trading_range_breakout_level_n, "high"),
            "bear_breakout_level": ("rolling_low", self.p.trading_range_breakout_level_n, "low"),
            "er_ma": ("efficiency_ratio", self.p.er_n, self.p.er_ma_n),
            "bband": ("bband", self.p.bbands_n, "close", 2),
            "BBR": ("body_bar_ratio",),
            "cross_bband_mid_count": ("cross_bband_mid_count", self.p.bbands_n, "close", "sum")
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
        high_series = df["high"]
        low_series = df["low"]
        close_series = df["close"]
        close_p = float(close_series.iat[i])

        # 若有倉，只更新出場線（也可以不更新）
        if pos.side is not None and pos.qty > 0:
            return intents

        # 無倉：
        # 做突破進場（LONG）
        bull_breakout_level = ctx.indicators["bull_breakout_level"]
        bear_breakout_level = ctx.indicators["bear_breakout_level"]
        er_ma = ctx.indicators["er_ma"]
        bband = ctx.indicators["bband"]
        bb_upper = bband["upper"]
        bb_lower = bband["lower"]
        bb_mid = bband["middle"]
        bb_width_pct = bband["width_pct"]
        cross_bband_mid_count = ctx.indicators["cross_bband_mid_count"]

        bbr = ctx.indicators["BBR"]

        if i < 1+self.p.min_follow_through_bars:
            # 檢查time_exit條件

            return intents
        
        # TR條件
        ## er_ma在門檻以內（表示趨勢不明確）
        er_cond = er_ma.iat[i-1-self.p.min_follow_through_bars] < self.p.er_threshold
        ## 布林通道在區間（表示盤整）
        bb_cond = bb_width_pct.iat[i-1-self.p.min_follow_through_bars] < self.p.bbands_width_pct_threshold[1] and bb_width_pct.iat[i-1-self.p.min_follow_through_bars] > self.p.bbands_width_pct_threshold[0]
        ## TR內至少要有幾次K線突破布林中軌
        cross_bband_mid_cond = cross_bband_mid_count.iat[i-1-self.p.min_follow_through_bars] >= self.p.min_cross_bband_mid_count
        
        # 突破條件
        ## 突破K收盤價突破前n根最高價
        bo_cond = close_series.iat[i-self.p.min_follow_through_bars] > bull_breakout_level.iat[i-1-self.p.min_follow_through_bars]
        ## 突破K線的實體占比要夠大
        breakout_bar_BBR = bbr.iat[i-self.p.min_follow_through_bars]
        breakout_bar_BBR_cond = breakout_bar_BBR > self.p.breakout_bar_BBR_threshold
        ## 突破K線的收盤價需突破突破水平多少%(TR的%)才算有效突破
        breakout_level = bull_breakout_level.iat[i-1-self.p.min_follow_through_bars]
        TR_range = bull_breakout_level.iat[i-1-self.p.min_follow_through_bars] - bear_breakout_level.iat[i-1-self.p.min_follow_through_bars]
        min_breakout_price = breakout_level + TR_range * self.p.min_breakout_penetration_pct / 100  # 以TR的百分比作為突破水平的參考，這樣突破條件會隨著TR的大小調整
        breakout_penetration_cond = close_series.iat[i-self.p.min_follow_through_bars] > min_breakout_price

        # 策略條件做多或雙向
        allow_side_cond = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)

        if er_cond and bb_cond and cross_bband_mid_cond and bo_cond and breakout_bar_BBR_cond and breakout_penetration_cond and allow_side_cond:
            ## 有幾根K線符合跟進條件(高於前一根收盤價)
            follow_through_cond = True
            for j in range(self.p.min_follow_through_bars):
                if close_series.iat[i-j] <= close_series.iat[i-j-1]:
                    follow_through_cond = False
                    break
            if follow_through_cond:
                # 計算風險
                ## 進場價為FT bar的收盤價(如果不用FT bar，就是突破K線的收盤價)
                entry_price = close_p
                if self.p.sl_chosen_by == "breakout_bar_low":
                    ## 停損價為突破K線的最低價
                    sl_price = low_series.iat[i-self.p.min_follow_through_bars]
                elif self.p.sl_chosen_by == "TR_low":
                    ## 停損價為TR的最低價
                    sl_price = bear_breakout_level.iat[i-1-self.p.min_follow_through_bars]
                ## 停利價為進場價 + 距離 * rr
                sl_distance = entry_price - sl_price
                tp_price = entry_price + sl_distance * self.p.rr
                ## 計算倉位大小（風險金額 / 單位風險）
                max_notional_lose = base_equity * self.p.equity_max_loss_pct / 100
                qty = max_notional_lose / sl_distance if sl_distance > 0 else 0.0

                intents.append(
                    OrderIntent(
                        action=ActionType.ENTRY,
                        side=Side.LONG,
                        qty=max(qty, self.p.min_qty),  # 最小下單量限制
                        entry_price=entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        be_price=None,
                        priority=50, 
                    )
                )




        return intents
