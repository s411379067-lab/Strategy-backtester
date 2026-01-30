from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position
from ..strategy_base import Strategy, StrategyContext
import pandas as pd

@dataclass(frozen=True)
class ALBOParams:
    break_out_series_n: int = 3
    break_out_n_bars: int = 10
    max_notional_pct: float = 1.0
    min_qty: float = 0.001
    sl_atr_like: float = 0.0  # MVP不做ATR，示範保留欄位
    fixed_sl_pct: float = 0.01
    rr: float = 2.0           # TP = SL距離 * rr
    time_exit_bars: int = 50


class ALBOStrategy(Strategy):
    def __init__(self, params: ALBOParams) -> None:
        self.p = params

    def required_indicators(self) -> Dict[str, Any]:
        n = self.p.break_out_series_n
        return {
            "strong_bull_bar_series": ("body_strictly_increasing", n),
            "bull_bar_series": ("bar_side_sum", n),
            "rocp_1": ("rocp", 1),
            f"rocp_{n}": ("rocp", n),
            "hh": ("rolling_high", self.p.break_out_n_bars, "high"),
            "atr": ("atr", 14),

        }

    def generate_intents(self, ctx: StrategyContext) -> List[OrderIntent]:
        i = ctx.i
        df = ctx.df
        pos = ctx.position
        init_equity = ctx.init_equity
        intents: List[OrderIntent] = []

        open_series = df["open"]
        close_p = float(df["close"].iat[i])

        # 若有倉，只更新出場線（也可以不更新）
        if pos.side is not None and pos.qty > 0:
            return intents

        # 無倉：做突破進場（LONG）
        strong_bull_bar_series = ctx.indicators["strong_bull_bar_series"]
        bull_bar_series = ctx.indicators["bull_bar_series"]
        rocp_1 = ctx.indicators["rocp_1"]
        rocp_n = ctx.indicators[f"rocp_{self.p.break_out_series_n}"]
        hh = ctx.indicators["hh"]
        hh_prev = float(hh.iat[i - 1]) if i - 1 >= 0 else float("nan")


        if i < self.p.break_out_series_n - 1:
            return intents
        atr_i = float(ctx.indicators["atr"].iat[i])
        rocp1_i = float(rocp_1.iat[i])

        # nan檢查
        if pd.isna(atr_i) or pd.isna(rocp1_i) or pd.isna(hh_prev):
            return intents
        # 最近 N 根body越來越強
        cond1 = strong_bull_bar_series.iat[i]
        # 最近 N 根都是 bull bar
        cond2 = bull_bar_series.iat[i] == self.p.break_out_series_n
        # 最後一根漲幅>1倍 atr
        cond3 = rocp_1.iat[i] > float(ctx.indicators["atr"].iat[i]) / close_p
        # 突破前n根最高價
        cond4 = close_p > hh_prev
        
        
        

        # 突破：close > 前一根 rolling high
        if cond1 and cond2 and cond3 and cond4:
            # 停損第一根K線開盤
            sl_price = open_series.iat[i - self.p.break_out_series_n + 1]
            # TP = SL距離 * rr
            tp_price = close_p + (close_p - sl_price) * self.p.rr
            entry_price = close_p
            max_notional = init_equity * self.p.max_notional_pct
            qty = max_notional / entry_price

            intents.append(
                OrderIntent(
                    action=ActionType.ENTRY,
                    side=Side.LONG,
                    qty=max(self.p.min_qty, float(qty)),
                    tp_price=tp_price,
                    sl_price=sl_price,
                    be_price=None,
                    priority=10,
                )
            )

        return intents
