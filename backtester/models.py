from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import pandas as pd


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class ActionType(str, Enum):
    ENTRY = "entry"
    ADD = "add"
    EXIT = "exit"


class ExitType(str, Enum):
    TP = "tp"
    SL = "sl"
    BE = "be"
    TIME = "time"
    MANUAL = "manual"


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0004            # 0.04% 例
    slippage_bps: float = 0.0           # 1 bps = 0.01%
    conservative_intrabar: bool = True  # 保守規則

    # 若同一根bar同時觸發TP/SL，保守規則採「對你最不利」：
    # LONG: 先SL；SHORT: 先SL（價格先往不利方向）
    time_exit_on_close: bool = True     # time-exit 用 close 出場


@dataclass(frozen=True)
class OrderIntent:
    action: ActionType
    side: Side
    qty: float
    # exit controls (若 action=EXIT 才會用)
    exit_type: Optional[ExitType] = None
    # 若策略用固定價位TP/SL/BE，可填入；否則留 None 代表由 engine/portfolio 當前狀態決定
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    be_price: Optional[float] = None
    priority: int = 100  # 數字越小越先處理


@dataclass
class Position:
    side: Optional[Side] = None
    qty: float = 0.0
    avg_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    # 策略常用的動態出場線（可被 strategy 更新）
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    be_price: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    time: pd.Timestamp
    action: ActionType
    side: Side
    qty: float
    price: float
    fee: float
    # 若為 exit，標記原因
    exit_type: Optional[ExitType] = None


@dataclass(frozen=True)
class Trade:
    side: Side
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_type: ExitType
    pnl: float
    bars_held: int


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series  # index = time
