import pandas as pd
from pytest import approx

from backtester.models import BacktestConfig, Side
from backtester.execution import ExecutionModel


def test_conservative_intrabar_long_sl_first():
    cfg = BacktestConfig(conservative_intrabar=True) # 初始倉位、滑價、fee
    ex = ExecutionModel(cfg) #

    # LONG：同一根同時觸發 TP/SL -> 保守先 SL
    exit_type, exit_price = ex.conservative_exit_price(
        side=Side.LONG,
        bar_open=100,
        bar_high=120,   # hit TP=110
        bar_low=80,     # hit SL=90
        bar_close=105,
        tp=110,
        sl=90,
        be=None,
        time_exit=False,
    )
    assert exit_type.value == "sl"
    assert exit_price == 90


def test_fee_and_slippage_applied_on_entry_long():
    cfg = BacktestConfig(fee_rate=0.001, slippage_bps=10)  # 0.1% fee, 10bps slippage
    ex = ExecutionModel(cfg)

    fill = ex.fill_entry(time=pd.Timestamp("2026-01-01"), side=Side.LONG, qty=2, price=100)
    # buy with +10bps => 100.1
    assert fill.price == approx(100.1)
    # fee = notional * 0.001 = 100.1*2*0.001 = 0.2002
    assert fill.fee == approx(0.2002)
