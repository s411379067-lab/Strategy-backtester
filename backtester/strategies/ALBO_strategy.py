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
    BO_n_times_atr: float = 1.0
    max_notional_pct: float = 1.0
    min_qty: float = 0.001
    sl_atr_like: float = 0.0  # MVP不做ATR，示範保留欄位
    fixed_sl_pct: float = 0.01
    rr: float = 2.0           # TP = SL距離 * rr
    time_exit_bars: int = 50
    allow_side: Optional[Side] = None  # None表示雙向進出場，Side.LONG表示只做多，Side.SHORT表示只做空


class ALBOStrategy(Strategy):
    def __init__(self, params: ALBOParams) -> None:
        self.p = params

    def required_indicators(self) -> Dict[str, Any]:
        n = self.p.break_out_series_n
        return {
            "strong_bar_series": ("body_strictly_increasing", n),
            "bar_series": ("bar_side_sum", n),
            "rocp_1": ("rocp", 1),
            f"rocp_{n}": ("rocp", n),
            "hh": ("rolling_high", self.p.break_out_n_bars, "high"),
            "ll": ("rolling_low", self.p.break_out_n_bars, "low"),
            "atr": ("atr", 14),
            "ma": ("ma", 20, "close", "EMA"),

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

        # 無倉：
        # 做突破進場（LONG）
        strong_bar_series = ctx.indicators["strong_bar_series"]
        bar_series = ctx.indicators["bar_series"]
        rocp_1 = ctx.indicators["rocp_1"]
        rocp_n = ctx.indicators[f"rocp_{self.p.break_out_series_n}"]
        hh = ctx.indicators["hh"]
        hh_prev = float(hh.iat[i - 1]) if i - 1 >= 0 else float("nan")


        if i < self.p.break_out_series_n - 1:
            # 檢查time_exit條件

            return intents
        atr_i = float(ctx.indicators["atr"].iat[i])
        rocp1_i = float(rocp_1.iat[i])

        # nan檢查
        if pd.isna(atr_i) or pd.isna(rocp1_i) or pd.isna(hh_prev):
            return intents
        # 最近 N 根body越來越強
        cond1 = strong_bar_series.iat[i]
        # 最近 N 根都是 bull bar
        cond2 = bar_series.iat[i] == self.p.break_out_series_n
        # 最後一根漲幅>1倍 atr
        cond3 = rocp_1.iat[i] > float(ctx.indicators["atr"].iat[i])*self.p.BO_n_times_atr/ close_p
        # 突破前n根最高價
        cond4 = close_p > hh_prev
        ma = ctx.indicators["ma"].iat[i]
        # 收盤價高於均線
        cond5 = close_p > ma
        # 策略條件做多或雙向
        cond6 = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)


        # 突破：close > 前一根 rolling high
        if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
            # 停損第一根K線開盤
            sl_price = open_series.iat[i - self.p.break_out_series_n + 1]
            # TP = SL距離 * rr
            tp_price = close_p + (close_p - sl_price) * self.p.rr
            entry_price = close_p
            max_notional_lose = init_equity * self.p.max_notional_pct / 100
            qty = max_notional_lose / (abs(entry_price - sl_price)) if abs(entry_price - sl_price) > 0 else 0.0

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
            
        # 做突破進場（SHORT）
        ll = ctx.indicators["ll"]
        ll_prev = float(ll.iat[i - 1]) if i - 1 >= 0 else float("nan")
        # 最近 N 根body越來越大
        cond1_short = strong_bar_series.iat[i]
        # 最近 N 根都是 bear bar
        cond2_short = bar_series.iat[i] == -self.p.break_out_series_n
        # 最後一根跌幅>1倍 atr
        cond3_short = rocp_1.iat[i] < -float(ctx.indicators["atr"].iat[i])*self.p.BO_n_times_atr/ close_p
        # 最後一根收盤突破前n根最低價
        cond4_short = close_p < ll_prev
        # 收盤價低於均線
        cond5_short = close_p < ma
        # 策略條件做空或雙向
        cond6_short = (self.p.allow_side is None) or (self.p.allow_side == Side.SHORT)

        if cond1_short and cond2_short and cond3_short and cond4_short and cond5_short and cond6_short:
            # 停損第一根K線開盤
            sl_price = open_series.iat[i - self.p.break_out_series_n + 1]
            # TP = SL距離 * rr
            tp_price = close_p - (sl_price - close_p) * self.p.rr
            entry_price = close_p
            max_notional_lose = init_equity * self.p.max_notional_pct / 100
            qty = max_notional_lose / (abs(entry_price - sl_price)) if abs(entry_price - sl_price) > 0 else 0.0

            intents.append(
                OrderIntent(
                    action=ActionType.ENTRY,
                    side=Side.SHORT,
                    qty=max(self.p.min_qty, float(qty)),
                    tp_price=tp_price,
                    sl_price=sl_price,
                    be_price=None,
                    priority=10,
                )
            )


        return intents
