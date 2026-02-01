from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..models import OrderIntent, ActionType, Side, ExitType, Position, SizingEquityBase
from ..strategy_base import Strategy, StrategyContext
import pandas as pd

@dataclass(frozen=True)
class ALBO_first_PBParams:
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