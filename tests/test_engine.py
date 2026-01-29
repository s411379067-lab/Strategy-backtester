import pandas as pd

from backtester.engine import BacktestEngine
from backtester.models import BacktestConfig
from backtester.strategies.xyz_strategy import XYZStrategy, XYZParams


def test_engine_runs_and_outputs_equity_curve():
    # 造一段簡單 OHLC
    idx = pd.date_range("2026-01-01", periods=50, freq="5min")
    df = pd.DataFrame(
        {
            "open":  [100 + i * 0.1 for i in range(50)],
            "high":  [100 + i * 0.1 + 0.5 for i in range(50)],
            "low":   [100 + i * 0.1 - 0.5 for i in range(50)],
            "close": [100 + i * 0.1 + 0.2 for i in range(50)],
        },
        index=idx,
    )

    cfg = BacktestConfig(initial_cash=10000, fee_rate=0.0, slippage_bps=0.0, conservative_intrabar=True)
    engine = BacktestEngine(cfg)
    strat = XYZStrategy(XYZParams(breakout_lookback=10, fixed_sl_pct=0.01, rr=2.0, qty=1.0))

    result = engine.run(df, strat)

    assert result.equity_curve.index.equals(df.index)
    assert len(result.equity_curve) == len(df)
