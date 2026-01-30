from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from .models import Position, Trade, Side, ExitType, Fill


@dataclass
class Portfolio:
    cash: float
    position: Position
    trades: List[Trade]

    def __init__(self, initial_cash: float) -> None:
        self.cash = float(initial_cash)
        self.position = Position()
        self.trades = []

    def equity(self, mark_price: float) -> float:
        if self.position.side is None or self.position.qty == 0:
            return self.cash
        # 未實現損益（單標的、無槓桿）
        if self.position.side == Side.LONG:
            unreal = (mark_price - self.position.avg_price) * self.position.qty
        else:
            unreal = (self.position.avg_price - mark_price) * self.position.qty
        return self.cash + unreal

    def apply_entry_fill(self, fill: Fill) -> None:
        assert fill.action.value == "entry"
        # 這裡簡化：一次只允許單一倉位（無加倉），加倉你之後可擴充
        self.position.side = fill.side
        self.position.qty = fill.qty
        self.position.avg_price = fill.price
        self.position.entry_time = fill.time
        self.position.entry_bar_i = fill.entry_bar_i
        self.cash -= fill.fee  # 扣單次手續費（不做保證金計算）

    def apply_exit_fill(self, fill: Fill, bars_held: int) -> None:
        assert fill.action.value == "exit"
        assert self.position.side is not None

        entry_price = self.position.avg_price
        qty = self.position.qty
        if self.position.side == Side.LONG:
            pnl = (fill.price - entry_price) * qty - fill.fee
        else:
            pnl = (entry_price - fill.price) * qty - fill.fee

        self.cash += pnl  # 將已實現損益回到現金（已扣 fee）
        self.trades.append(
            Trade(
                side=self.position.side,
                qty=qty,
                entry_time=self.position.entry_time or fill.time,
                entry_price=entry_price,
                sl_price=self.position.sl_price,
                tp_price=self.position.tp_price,
                exit_time=fill.time,
                exit_price=fill.price,
                exit_type=fill.exit_type or ExitType.MANUAL,
                pnl=pnl,
                bars_held=bars_held,
            )
        )

        # flat
        self.position = Position()
