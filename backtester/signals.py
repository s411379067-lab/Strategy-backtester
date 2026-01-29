from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd

class Condition(Protocol):
    def eval(self, df: pd.DataFrame) -> pd.Series: ...

@dataclass
class And:
    a: Condition
    b: Condition
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.a.eval(df) & self.b.eval(df)

@dataclass
class Or:
    a: Condition
    b: Condition
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.a.eval(df) | self.b.eval(df)

@dataclass
class Not:
    a: Condition
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return ~self.a.eval(df)
