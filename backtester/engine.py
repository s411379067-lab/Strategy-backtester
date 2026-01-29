from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

from .models import BacktestConfig, BacktestResult, Side, ActionType, OrderIntent, ExitType
from .indicators import IndicatorRegistry
from .strategy_base import Strategy, StrategyContext
from .execution import ExecutionModel
from .portfolio import Portfolio


@dataclass
class BacktestEngine:
    config: BacktestConfig

    def run(self, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        self._validate_df(df)

        indicator_registry = IndicatorRegistry()
        indicators = self._compute_indicators(df, strategy, indicator_registry)

        portfolio = Portfolio(initial_cash=self.config.initial_cash)
        exec_model = ExecutionModel(config=self.config)

        equity_points: List[float] = []
        equity_index: List[pd.Timestamp] = []

        # bars held 計數：用 index 差估算（MVP）
        entry_bar_i: int | None = None

        for i in range(len(df)):
            t = df.index[i]
            o = float(df["open"].iat[i])
            h = float(df["high"].iat[i])
            l = float(df["low"].iat[i])
            c = float(df["close"].iat[i])

            # 1) 先處理持倉的 intrabar exit（TP/SL/BE + 保守規則）
            if portfolio.position.side is not None and portfolio.position.qty > 0:
                side = portfolio.position.side
                exit_type, exit_price = exec_model.conservative_exit_price(
                    side=side,
                    bar_open=o,
                    bar_high=h,
                    bar_low=l,
                    bar_close=c,
                    tp=portfolio.position.tp_price,
                    sl=portfolio.position.sl_price,
                    be=portfolio.position.be_price,
                    time_exit=False,
                )

                if exit_type is not None and exit_price is not None:
                    fill = exec_model.fill_exit(
                        time=t,
                        side=side,
                        qty=portfolio.position.qty,
                        price=exit_price,
                        exit_type=exit_type,
                    )
                    bars_held = (i - entry_bar_i) if entry_bar_i is not None else 0
                    portfolio.apply_exit_fill(fill, bars_held=bars_held)
                    entry_bar_i = None

            # 2) time-exit（用 close 出場）
            if portfolio.position.side is not None and portfolio.position.qty > 0:
                # 這裡示範：策略若要 time-exit，透過 intent 來觸發
                pass

            # 3) 收盤產生 intents（entry / exit / 更新出場線）
            ctx = StrategyContext(
                df=df,
                i=i,
                time=t,
                position=portfolio.position,
                indicators=indicators,
            )
            intents = strategy.generate_intents(ctx)

            # 4) 套用 intents：MVP 只做
            #    - 若無倉：允許 entry
            #    - 若有倉：允許更新 tp/sl/be、或 time-exit（close 出）
            intents_sorted = sorted(intents, key=lambda x: x.priority)

            for it in intents_sorted:
                if it.action == ActionType.ENTRY:
                    if portfolio.position.side is None or portfolio.position.qty == 0:
                        fill = exec_model.fill_entry(time=t, side=it.side, qty=it.qty, price=c)
                        portfolio.apply_entry_fill(fill)
                        entry_bar_i = i
                        # 若 intent 有帶 tp/sl/be，直接寫入 position
                        portfolio.position.tp_price = it.tp_price
                        portfolio.position.sl_price = it.sl_price
                        portfolio.position.be_price = it.be_price

                elif it.action == ActionType.EXIT:
                    if portfolio.position.side is not None and portfolio.position.qty > 0:
                        # time-exit：用 close 出
                        if it.exit_type == ExitType.TIME:
                            fill = exec_model.fill_exit(
                                time=t,
                                side=portfolio.position.side,
                                qty=portfolio.position.qty,
                                price=c,
                                exit_type=ExitType.TIME,
                            )
                            bars_held = (i - entry_bar_i) if entry_bar_i is not None else 0
                            portfolio.apply_exit_fill(fill, bars_held=bars_held)
                            entry_bar_i = None

                # ADD 暫不實作（你後續要加倉時再擴充）

            # 5) 記錄 equity（用 close mark）
            equity_points.append(portfolio.equity(mark_price=c))
            equity_index.append(t)

        equity = pd.Series(equity_points, index=pd.Index(equity_index, name="time"), name="equity")
        return BacktestResult(trades=portfolio.trades, equity_curve=equity)

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {sorted(missing)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df.index must be a pandas.DatetimeIndex")

    @staticmethod
    def _compute_indicators(df: pd.DataFrame, strategy: Strategy, reg: IndicatorRegistry) -> Dict[str, Any]:
        req = strategy.required_indicators()
        indicators: Dict[str, Any] = {}
        for name, spec in req.items():
            # MVP: spec 用 tuple 表示 (fn, params...)
            fn = spec[0]
            params = spec[1:]
            if fn == "rolling_high":
                indicators[name] = reg.rolling_high(df, *params)
            elif fn == "rolling_low":
                indicators[name] = reg.rolling_low(df, *params)
            else:
                raise ValueError(f"Unknown indicator fn: {fn}")
        return indicators
