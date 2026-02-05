from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ALRangeBoundaryFailureParams:
    """
    Parameters for ALRangeBoundaryFailureStrategy
    
    Attributes:
    er_n (int): Efficiency Ratio 計算K線糾結程度
    er_ma_n (int): Efficiency Ratio 移動平均平滑參數
    regression_n (int): 回歸線計算K線數
    std_n (int): 計算回歸邊界標準差倍數
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
    regression_n: int = 100
    std_n: int = 2
    # pullback_ratio: List[float] = field(default_factory=lambda: [0.5])  # 回調百分比
    slope_threshold: float = 0.1  # 斜率區分TR/BC閾值
    min_sl_range_pct: float = 0.3  # 最小停損距離百分比
    max_notional_pct: float = 1.0
    min_qty: float = 0.001
    rr: float = 1.0           # TP = SL距離 * rr
    time_exit_bars: int = 50
    allow_side: Optional[Side] = None  # None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空
    sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL
class ALRangeBoundaryFailureStrategy(Strategy):
    """
    Docstring for ALRangeBoundaryFailureStrategy
    
    策略邏輯：
    1. 偵測K線效率低(糾結區間)
    2. 計算回歸線、斜率靠斜率區分TR、BC
    3. - TR: 無方向所以多空都可以做，打到回歸邊界(n倍標準差)limit order進場
        - BC: 看看斜率方向，只做多或只做空，打到回歸邊界(n倍標準差)limit order進場
    4. 停損設在回歸線同側n*2倍標準差
    5. 停利設在回歸線另一側(n倍標準差)
    rr應該約2:1(相對位置: entry:-1, sl:-2, tp:+1)
    """

    def __init__(self, params: ALRangeBoundaryFailureParams) -> None:
        self.p = params
        self.state = None  # 可用來記錄策略狀態
        self.initial_sl_price: Optional[float] = None


    def required_indicators(self) -> Dict[str, Any]:
        return {
            "er": ("efficiency_ratio", self.p.er_n, self.p.er_ma_n, "close"),
            "regression_mid": ("liner_regression_mid", self.p.regression_n, "close"),
            "regression_residuals": ("liner_regression_residuals", self.p.regression_n, "close"),
            "regression_residuals_std": ("bar_regression_residuals_std", self.p.regression_n, "close"),
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
            regression_mid_series = ctx.indicators["regression_mid"]
            regression_residuals_series = ctx.indicators["regression_residuals"]
            regression_residuals_std_series = ctx.indicators["regression_residuals_std"]
            upper_boundary_outer = float(regression_mid_series.iat[i]) + self.p.std_n * float(regression_residuals_std_series.iat[i])
            lower_boundary_outer = float(regression_mid_series.iat[i]) - self.p.std_n * float(regression_residuals_std_series.iat[i])
            upper_boundary_inner = float(regression_mid_series.iat[i]) + (self.p.std_n/2) * float(regression_residuals_std_series.iat[i])
            lower_boundary_inner = float(regression_mid_series.iat[i]) - (self.p.std_n/2) * float(regression_residuals_std_series.iat[i])

            if pos.side == Side.LONG:
                # 多單持倉，更新移動停損價格，下一根K線生效
                new_sl_price = max(lower_boundary_outer, self.initial_sl_price if self.initial_sl_price is not None else lower_boundary_outer)
                new_tp_price = upper_boundary_inner
                
                if self.initial_sl_price is not None:
                    intents.append(
                        OrderIntent(
                            action=ActionType.UPDATE,
                            side=Side.LONG,
                            qty=pos.qty,
                            sl_price=new_sl_price,
                            tp_price=new_tp_price,
                            be_price=None,
                            priority=90,
                        )
                    )
            elif pos.side == Side.SHORT:
                # 空單持倉，更新移動停損價格，下一根K線生效
                new_sl_price = min(upper_boundary_outer, self.initial_sl_price if self.initial_sl_price is not None else upper_boundary_outer)
                new_tp_price = lower_boundary_inner
                
                if self.initial_sl_price is not None:
                    intents.append(
                        OrderIntent(
                            action=ActionType.UPDATE,
                            side=Side.SHORT,
                            qty=pos.qty,
                            sl_price=new_sl_price,
                            tp_price=new_tp_price,
                            be_price=None,
                            priority=90,
                        )
                    )
            return intents
        
        regression_mid_series = ctx.indicators["regression_mid"]
        regression_residuals_series = ctx.indicators["regression_residuals"]
        regression_residuals_std_series = ctx.indicators["regression_residuals_std"]
        er_series = ctx.indicators["er"]
        regression_residual = float(regression_residuals_series.iat[i])
        regression_residual_std = float(regression_residuals_std_series.iat[i])
        upper_boundary_outer = float(regression_mid_series.iat[i]) + self.p.std_n * regression_residual_std
        upper_boundary_inner = float(regression_mid_series.iat[i]) + (self.p.std_n/2) * regression_residual_std
        lower_boundary_outer = float(regression_mid_series.iat[i]) - self.p.std_n * regression_residual_std
        lower_boundary_inner = float(regression_mid_series.iat[i]) - (self.p.std_n/2) * regression_residual_std
        ## 亞、歐、美盤前盤後不做
        time_hour = df.index[i].hour
        cond_time_filter = time_hour not in [8,9,13,14,19,20,21,22,23,0,1]




        
        # 無倉，檢查進場
        if pos.side is None or pos.qty == 0:
            # 沒有倉位時，檢查狀態
            self.initial_sl_price = None

            ## 1. 一開始默認state = None，檢查有沒有K線糾結
            if self.state is None:
                er = float(er_series.iat[i])

                ##  2. 確認是 TR 或 BC 形成
                if er < 0.2:
                    
                    
                    self.state = "TR"
            if self.state == "TR":
                # 檢查是否突破內層邊界
                cond1 = high_series.iat[i] >= lower_boundary_inner >= low_series.iat[i]
                cond1_short = low_series.iat[i] <= upper_boundary_inner <= high_series.iat[i]
                # 多單方向檢查
                cond2 = self.p.allow_side != Side.SHORT
                # 空單方向檢查
                cond2_short = self.p.allow_side != Side.LONG


                if cond1 and cond2 and cond_time_filter:
                    # 碰到下邊界，做多
                    entry_price = close_series.iat[i] # 用收盤價進場，因為是當下這根K線突破內邊界才進場，所以不會有look-ahead問題
                    self.initial_sl_price = lower_boundary_outer # SL設在回歸線同側n倍標準差(外層邊界)移動停損
                    tp_price = upper_boundary_inner # TP設在回歸線另一側n/2倍標準差(內層邊界)移動停利
                    max_sl_notional = base_equity * self.p.max_notional_pct
                    qty = max_sl_notional / (entry_price - self.initial_sl_price) if (entry_price - self.initial_sl_price) > 0 else 0.0
                    intents.append(
                        OrderIntent(
                            action=ActionType.ENTRY,
                            side=Side.LONG,
                            qty=max(qty, self.p.min_qty),
                            entry_price=entry_price,
                            sl_price=self.initial_sl_price,
                            tp_price=tp_price,
                            be_price=None,
                            priority=100, 
                        )
                    )

                elif cond1_short and cond2_short and cond_time_filter:
                    # 碰到上邊界，做空
                    entry_price = close_series.iat[i] # 用收盤價進場，因為是當下這根K線突破內邊界才進場，所以不會有look-ahead問題
                    self.initial_sl_price = upper_boundary_outer # SL設在回歸線同側n倍標準差(外層邊界)移動停損
                    tp_price = lower_boundary_inner # TP設在回歸線另一側n/2倍標準差(內層邊界)移動停利
                    max_sl_notional = base_equity * self.p.max_notional_pct
                    qty = max_sl_notional / (self.initial_sl_price - entry_price) if (self.initial_sl_price - entry_price) > 0 else 0.0
                    intents.append(
                        OrderIntent(
                            action=ActionType.ENTRY,
                            side=Side.SHORT,
                            qty=max(qty, self.p.min_qty),
                            entry_price=entry_price,
                            sl_price=self.initial_sl_price,
                            tp_price=tp_price,
                            be_price=None,
                            priority=100, 
                        )
                    )
                self.state = None  # 無論是否進場，回到初始狀態重新檢查K線糾結

        return intents

                


                
