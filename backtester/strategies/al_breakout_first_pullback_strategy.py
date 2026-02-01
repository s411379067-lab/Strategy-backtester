from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ALBreakoutFirstPullbackParams:
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
    sizing_equity_base: SizingEquityBase = SizingEquityBase.INITIAL
class ALBreakoutFirstPullbackStrategy(Strategy):
    """
    Docstring for ALBreakoutFirstPullbackStrategy
    
    策略邏輯：
    1. 偵測突破訊號
    2. 突破後等待第一根反向K線R1
    3. 如果反向K線下一根碰到R1低點(多頭為例)，則在R1低點下單進場
    4. 停損設在突破點下方(多頭為例)，停利設在停損距離的RR倍
    5. 移動停損() -- todo
    """

    def __init__(self, params: ALBreakoutFirstPullbackParams) -> None:
        self.p = params

    def required_indicators(self) -> Dict[str, Any]:
        n = self.p.break_out_series_n
        return {
            "rocp_1": ("rocp", 1),
            f"rocp_{n}": ("rocp", n),
            "hh": ("rolling_high", self.p.break_out_n_bars, "high"),
            "ll": ("rolling_low", self.p.break_out_n_bars, "low"),
            "atr": ("atr", 14),
            "ma": ("ma", 20, "close", "EMA"),
            "streak": ("bar_streak", "ignore"),
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
            pass  # 這邊不更新出場線

        streak = ctx.indicators["streak"]
        streak_prev = int(streak.iat[i - 1]) if i - 1 >= 0 else 0
        streak_prev2 = int(streak.iat[i-2]) if i - 2 >= 0 else 0
        streak_curr = int(streak.iat[i])
        # 如果還沒出現第一根反向K線，直接返回
        if (np.sign(streak_prev2) == np.sign(streak_prev)):
            return intents
        if i < 2:
            return intents
        # i為進場K線，i-1為反向K線R1，i-2為突破的最後一根K線



        
        # 無倉，檢查進場
        else:
            # 做突破進場（LONG）
            

            
            hh = ctx.indicators["hh"]
            hh_prev3 = float(hh.iat[i - 3]) if i - 3 >= 0 else float("nan")
            # 前streak_prev根開盤到收盤的幅度(BO range)
            bo_open_p = float(open_series.iat[i - 1 - streak_prev2]) if streak_prev2 > 0 else 0.0
            bo_range = abs(close_series.iat[i - 2] - bo_open_p) if streak_prev2 > 0 else 0.0


            # 檢查前兩根是否為多頭突破

            ## 連續n根同向K線以上
            cond1 = streak_prev2 >= self.p.break_out_series_n
            ## 突破前高
            cond2 = close_series.iat[i - 2] > hh_prev3
            ## 突破幅度>=atr最小幅度
            cond3 = bo_range >= self.p.BO_n_times_atr * float(ctx.indicators["atr"].iat[i - 2])
            ## 這根K線觸及R1低點
            cond4 = low_series.iat[i] <= low_series.iat[i - 1]
            ## 策略條件做多或雙向
            cond5 = (self.p.allow_side is None) or (self.p.allow_side == Side.LONG)

            if cond1 and cond2 and cond3 and cond4 and cond5:
                # 進場價格設在R1低點buy limit
                entry_price = low_series.iat[i - 1]
                # 停損設在突破點下方
                sl_price = bo_open_p
                sl_distance = entry_price - sl_price
                # 停利設在停損距離的RR倍
                tp_price = entry_price + sl_distance * self.p.rr

                # 計算可用資金與下單數量
                max_notional = base_equity * self.p.max_notional_pct
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
            debug_info = {
                "cond1": cond1,
                "cond2": cond2,
                "cond3": cond3,
                "cond4": cond4,
                "cond5": cond5,
                "streak_prev2": streak_prev2,
                "streak_prev": streak_prev,
                "streak_curr": streak_curr,
            }
            return intents
