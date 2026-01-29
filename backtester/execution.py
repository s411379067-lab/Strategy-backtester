from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .models import (
    BacktestConfig,
    Fill,
    Position,
    Side,
    ActionType,
    ExitType,
)


def _apply_slippage(price: float, side: Side, slippage_bps: float, is_entry_or_buy: bool) -> float:
    """
    保守滑價：買更貴、賣更便宜。
    - entry long = buy
    - exit long  = sell
    - entry short = sell
    - exit short  = buy
    """
    bps = slippage_bps / 10_000.0
    if is_entry_or_buy:
        return price * (1.0 + bps)
    return price * (1.0 - bps)


def _fee(notional: float, fee_rate: float) -> float:
    return abs(notional) * fee_rate


@dataclass
class ExecutionModel:
    config: BacktestConfig

    def fill_entry(self, time: pd.Timestamp, side: Side, qty: float, price: float) -> Fill:
        # entry long=buy, entry short=sell
        is_buy = (side == Side.LONG)
        fill_price = _apply_slippage(price, side, self.config.slippage_bps, is_entry_or_buy=is_buy)
        fee = _fee(notional=fill_price * qty, fee_rate=self.config.fee_rate)
        return Fill(time=time, action=ActionType.ENTRY, side=side, qty=qty, price=fill_price, fee=fee)

    def fill_exit(self, time: pd.Timestamp, side: Side, qty: float, price: float, exit_type: ExitType) -> Fill:
        # exit long=sell, exit short=buy
        is_buy = (side == Side.SHORT)
        fill_price = _apply_slippage(price, side, self.config.slippage_bps, is_entry_or_buy=is_buy)
        fee = _fee(notional=fill_price * qty, fee_rate=self.config.fee_rate)
        return Fill(time=time, action=ActionType.EXIT, side=side, qty=qty, price=fill_price, fee=fee, exit_type=exit_type)

    def conservative_exit_price(
        self,
        side: Side,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        tp: Optional[float],
        sl: Optional[float],
        be: Optional[float],
        time_exit: bool,
    ) -> tuple[Optional[ExitType], Optional[float]]:
        """
        回傳 (exit_type, exit_price)。
        保守規則：同一根 bar 同時觸發多條件，優先對你最不利。
        """
        hits = []

        if side == Side.LONG:
            if sl is not None and bar_low <= sl:
                hits.append((ExitType.SL, sl))
            if be is not None and bar_low <= be:
                hits.append((ExitType.BE, be))
            if tp is not None and bar_high >= tp:
                hits.append((ExitType.TP, tp))
        else:
            # SHORT：SL = 價格上去觸發；TP = 價格下去觸發
            if sl is not None and bar_high >= sl:
                hits.append((ExitType.SL, sl))
            if be is not None and bar_high >= be:
                hits.append((ExitType.BE, be))
            if tp is not None and bar_low <= tp:
                hits.append((ExitType.TP, tp))

        if hits:
            if not self.config.conservative_intrabar:
                # 非保守：用 priority（這裡簡化成 SL > BE > TP 仍然符合常識）
                priority = {ExitType.SL: 0, ExitType.BE: 1, ExitType.TP: 2}
                hits.sort(key=lambda x: priority[x[0]])
                return hits[0]
            # 保守：永遠先 SL（最不利），其次 BE，最後 TP
            priority = {ExitType.SL: 0, ExitType.BE: 1, ExitType.TP: 2}
            hits.sort(key=lambda x: priority[x[0]])
            return hits[0]

        if time_exit and self.config.time_exit_on_close:
            return (ExitType.TIME, bar_close)

        return (None, None)
