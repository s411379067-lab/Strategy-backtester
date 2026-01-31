from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd

from .models import OrderIntent, Position


@dataclass(frozen=True)
class StrategyContext:
    df: pd.DataFrame
    i: int
    time: pd.Timestamp
    position: Position
    init_equity: float
    now_equity: float
    indicators: Dict[str, Any]  # 已算好的指標/特徵（Series可用 .iat[i] 取值）


class Strategy(ABC):
    @abstractmethod
    def required_indicators(self) -> Dict[str, Any]:
        """回傳要計算的指標定義（MVP: 由 engine 負責放入 context.indicators）。"""
        raise NotImplementedError

    @abstractmethod
    def generate_intents(self, ctx: StrategyContext) -> List[OrderIntent]:
        """bar close 產生下一步意圖（entry/add/exit）。"""
        raise NotImplementedError
