from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position
from ..strategy_base import Strategy, StrategyContext


@dataclass(frozen=True)
class XYZParams:
    breakout_lookback: int = 20
    qty: float = 1.0
    sl_atr_like: float = 0.0  # MVP不做ATR，示範保留欄位
    fixed_sl_pct: float = 0.01
    rr: float = 2.0           # TP = SL距離 * rr
    time_exit_bars: int = 50


class XYZStrategy(Strategy):
    def __init__(self, params: XYZParams) -> None:
        self.p = params

    def required_indicators(self) -> Dict[str, Any]:
        # 用前 N 根最高價做突破（不含當根）
        n = self.p.breakout_lookback
        return {
            "hh": ("rolling_high", n, "high"),
        }

    def generate_intents(self, ctx: StrategyContext) -> List[OrderIntent]:
        i = ctx.i
        df = ctx.df
        pos = ctx.position
        intents: List[OrderIntent] = []

        close = float(df["close"].iat[i])

        # time-exit：用 bars_held 由 engine 記錄更好；MVP用 entry_time + i 推估做不到精準
        # 這裡改用「策略不處理 bars_held」，先不做 time-exit（由你日後接 engine 提供 bars_held）
        # -> 為了示範，我們先用很粗的方式：若 entry_time 存在且 i - entry_bar_i >= N 才 time-exit。
        # 因 StrategyContext 未含 entry_bar_i，MVP先不做。

        # 若有倉，只更新出場線（也可以不更新）
        if pos.side is not None and pos.qty > 0:
            return intents

        # 無倉：做突破進場（LONG）
        hh_series = ctx.indicators["hh"]
        hh_prev = float(hh_series.iat[i - 1]) if i - 1 >= 0 else float("nan")
        if i < self.p.breakout_lookback or hh_prev != hh_prev:  # nan
            return intents

        # 突破：close > 前一根 rolling high
        if close > hh_prev:
            sl_price = close * (1.0 - self.p.fixed_sl_pct)
            tp_price = close + (close - sl_price) * self.p.rr

            intents.append(
                OrderIntent(
                    action=ActionType.ENTRY,
                    side=Side.LONG,
                    qty=self.p.qty,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    be_price=None,
                    priority=10,
                )
            )

        return intents
