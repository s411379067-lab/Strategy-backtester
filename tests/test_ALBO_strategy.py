# tests/test_albo_strategy.py
import pandas as pd
import pytest

from backtester.models import Position, Side, ActionType
from backtester.strategy_base import StrategyContext

from backtester.strategies.ALBO_strategy import ALBOStrategy, ALBOParams


def _make_df(n_rows: int = 10, side: str = "long") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n_rows, freq="5min")
    if side == "long":
        df = pd.DataFrame(
            {
                "open":  [100.0 + i for i in range(n_rows)],
                "high":  [101.0 + i for i in range(n_rows)],
                "low":   [99.0 + i for i in range(n_rows)],
                "close": [100.5 + i for i in range(n_rows)],
            },
            index=idx,
        )
        return df
    else:
        df = pd.DataFrame(
            {
                "open":  [100.0 - i for i in range(n_rows)],
                "high":  [101.0 - i for i in range(n_rows)],
                "low":   [99.0 - i for i in range(n_rows)],
                "close": [99.5 - i for i in range(n_rows)],
            },
            index=idx,
        )
        return df


def test_albo_required_indicators_keys():
    p = ALBOParams(
        break_out_series_n=3,
        break_out_n_bars=10,
        rr=1.0,
        qty=1.0,
    )
    strat = ALBOStrategy(p)
    req = strat.required_indicators()
    # LONG
    assert "strong_bar_series" in req
    assert "bar_series" in req
    assert "rocp_1" in req
    assert f"rocp_{p.break_out_series_n}" in req
    assert "hh" in req
    assert "atr" in req
    assert "ma" in req
    # SHORT
    assert "ll" in req



def test_albo_generate_intents_entry_when_LONG_conditions_met():
    # 目標：在滿足 cond1~cond4 時，必須產生一筆 LONG ENTRY intent，且 SL/TP 計算正確
    n = 3
    p = ALBOParams(
        break_out_series_n=n,
        break_out_n_bars=10,  # 這個單元測試不依賴 rolling_high 真實計算，直接餵 hh
        rr=2.0,
    )
    strat = ALBOStrategy(p)

    df = _make_df(10)
    i = 6  # 確保 i >= n-1，且 i-n+1 不越界
    t = df.index[i]

    # 準備 indicators（直接餵 Series，模擬 engine 計算後放進 ctx.indicators）
    idx = df.index

    # cond1: 最近 N 根 body 嚴格遞增 -> 在 i 位置給 True
    strong = pd.Series([False] * len(df), index=idx)
    strong.iat[i] = True

    # cond2: 最近 N 根都是 bull bar -> bar_side_sum == n
    bull_sum = pd.Series([0] * len(df), index=idx, dtype=float)
    bull_sum.iat[i] = float(n)

    # cond3: rocp_1 > atr/close
    rocp_1 = pd.Series([0.0] * len(df), index=idx)
    rocp_1.iat[i] = 0.02  # 2%

    atr = pd.Series([1.0] * len(df), index=idx)  # atr=1
    # close 大約 106.5，因此 atr/close ~ 0.009...，rocp_1=0.02 應成立

    # cond4: close > hh_prev
    hh = pd.Series([100.0] * len(df), index=idx)
    hh.iat[i - 1] = 105.0  # hh_prev=105
    # close[i] 會是 100.5 + i = 106.5 > 105

    # rocp_n 你目前程式雖取出但未使用；仍提供避免 KeyError
    rocp_n = pd.Series([0.0] * len(df), index=idx)

    indicators = {
        "strong_bar_series": strong,
        "bar_series": bull_sum,
        "rocp_1": rocp_1,
        f"rocp_{n}": rocp_n,
        "hh": hh,
        "ll": pd.Series([0.0]*len(df), index=idx),  # 不用的話隨便給一個 Series
        "atr": atr,
        "ma": pd.Series([100.0]*len(df), index=idx),  # 給一個會被突破的均線
    }

    pos = Position(side=None, qty=0.0)
    ctx = StrategyContext(df=df, i=i, time=t, position=pos, indicators=indicators, init_equity=10000.0, now_equity=10000.0)

    intents = strat.generate_intents(ctx)

    assert len(intents) == 1
    it = intents[0]
    assert it.action == ActionType.ENTRY
    assert it.side == Side.LONG
    # assert it.qty == p.qty

    # 依你策略定義：SL = 最近 N 根序列第一根的 open（i-n+1）
    sl_idx = i - n + 1
    expected_sl = float(df["open"].iat[sl_idx])

    expected_tp = float(df["close"].iat[i]) + (float(df["close"].iat[i]) - expected_sl) * p.rr

    assert it.sl_price == expected_sl
    assert it.tp_price == expected_tp

def test_albo_generate_intents_entry_when_SHORT_conditions_met():
    # 目標：在滿足 cond1~cond4 時，必須產生一筆 SHORT ENTRY intent，且 SL/TP 計算正確
    n = 3
    p = ALBOParams(
        break_out_series_n=n,
        break_out_n_bars=10,  # 這個單元測試不依賴 rolling_high 真實計算，直接餵 hh
        rr=2.0,
        allow_side=Side.SHORT,
    )
    strat = ALBOStrategy(p)

    df = _make_df(10, side="short")
    i = 6  # 確保 i >= n-1，且 i-n+1 不越界
    t = df.index[i]

    # 準備 indicators（直接餵 Series，模擬 engine 計算後放進 ctx.indicators）
    idx = df.index

    # cond1_short: 最近 N 根 body 嚴格遞增 -> 在 i 位置給 True
    strong = pd.Series([False] * len(df), index=idx)
    strong.iat[i] = True

    # cond2_short: 最近 N 根都是 bear bar -> bar_side_sum == -n
    short_sum = pd.Series([0] * len(df), index=idx, dtype=float)
    short_sum.iat[i] = float(-n)

    # cond3_short: rocp_1 > atr/close
    rocp_1 = pd.Series([0.0] * len(df), index=idx)
    rocp_1.iat[i] = -0.02  # 2%

    atr = pd.Series([1.0] * len(df), index=idx)  # atr=1
    # close 大約 96.5，因此 atr/close ~ 0.0103，rocp_1=-0.02 應成立

    # cond4_short: close > ll_prev
    ll = pd.Series([100.0] * len(df), index=idx)
    ll.iat[i - 1] = 95.0  # ll_prev=95
    # close[i] 會是 100.5 - i = 94.5 < 95

    # rocp_n 你目前程式雖取出但未使用；仍提供避免 KeyError
    rocp_n = pd.Series([0.0] * len(df), index=idx)

    indicators = {
        "strong_bar_series": strong,
        "bar_series": short_sum,
        "rocp_1": rocp_1,
        f"rocp_{n}": rocp_n,
        "hh": pd.Series([0.0]*len(df), index=idx),  # 不用的話隨便給一個 Series
        "ll": ll,
        "atr": atr,
        "ma": pd.Series([100.0]*len(df), index=idx),  # 給一個會被突破的均線

    }

    pos = Position(side=None, qty=0.0)
    ctx = StrategyContext(df=df, i=i, time=t, position=pos, indicators=indicators, init_equity=10000.0, now_equity=10000.0)

    intents = strat.generate_intents(ctx)

    assert len(intents) == 1
    it = intents[0]
    assert it.action == ActionType.ENTRY
    assert it.side == Side.SHORT
    # assert it.qty == p.qty

    # 依你策略定義：SL = 最近 N 根序列第一根的 open（i-n+1）
    sl_idx = i - n + 1
    expected_sl = float(df["open"].iat[sl_idx])

    expected_tp = float(df["close"].iat[i]) - (expected_sl - float(df["close"].iat[i])) * p.rr * 1

    # close: 99.5 98.5 97.5 96.5 95.5 94.5 93.5
    # open: 100.0 99.0 98.0 97.0 96.0 95.0 94.0
    # i=6 -> close=93.5
    # sl = open[i-n+1] = open[4] = 96.0
    # tp = 93.5 - (96.0 - 93.5) * 2 = 88.5
    assert it.sl_price == expected_sl
    assert it.tp_price == expected_tp



def test_albo_generate_intents_no_entry_when_in_position():
    # 有倉時應該不產生 entry intents（你目前的實作是直接 return []）
    p = ALBOParams(
        break_out_series_n=3,
        break_out_n_bars=10,
        rr=2.0,
        qty=1.0,
    )
    strat = ALBOStrategy(p)

    df = _make_df(10)
    i = 6
    t = df.index[i]

    # indicators 內容在這個 case 不重要，給空也可以（因為會提早 return）
    indicators = {}

    pos = Position(side=Side.LONG, qty=1.0, avg_price=100.0, entry_time=df.index[0])
    ctx = StrategyContext(df=df, i=i, time=t, position=pos, indicators=indicators, init_equity=10000.0, now_equity=10000.0)

    intents = strat.generate_intents(ctx)
    assert intents == []
