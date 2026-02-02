# tests/test_albo_strategy.py
import pandas as pd
import pytest
import numpy as np

from backtester.engine import BacktestEngine
from backtester.models import BacktestConfig, Position, Side, ActionType
from backtester.strategy_base import StrategyContext

from backtester.strategies.al_breakout_first_pullback_strategy import ALBreakoutFirstPullbackStrategy, ALBreakoutFirstPullbackParams
from backtester import indicators as ind


import pandas as pd
def load_crypto_parquet_data(coin_name: str, timeframe: str = "5m", nM: int = 54, section: str = "UTC") -> pd.DataFrame:
    df = pd.read_parquet(fr'C:\Users\User\Desktop\Crypto\{coin_name}_{timeframe}_{nM}M_{section}.parquet')
    return df
def generate_us_session_bars_info(df, include_holidays: bool = False):
    # 確保時間有時區資訊
    df['dt_ny'] = pd.to_datetime(df['dt_utc'], utc=True).dt.tz_convert('America/New_York')

    # 取日期（當地日曆）
    df['date'] = df['dt_ny'].dt.date
    df['weekday'] = df['dt_ny'].dt.day_name() 
    # 對每天依時間排序並編號
    df = df.sort_values(['date', 'dt_ny']).reset_index(drop=True)
    df['bar_index'] = df.groupby('date').cumcount() + 1  # 第幾根K線，從1開始
    if not include_holidays:
        weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    else:
        weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df = df.loc[df['weekday'].isin(weekday)]
    # 查看結果
    df.set_index('dt_utc', inplace=True)
    # print(df[['dt_ny', 'date', 'weekday', 'bar_index']].head(5))

    return df
def generate_allday_bars_info(df, include_holidays: bool = True):
# 確保時間有時區資訊
    df['dt_ny'] = pd.to_datetime(df['dt_utc'], utc=True).dt.tz_convert('America/New_York')

    # 取日期（當地日曆）
    df['date'] = df['dt_ny'].dt.date
    df['time'] = df['dt_ny'].dt.time
    df['weekday'] = df['dt_ny'].dt.day_name() 
    # 對每天依時間排序並編號
    df = df.sort_values(['date', 'dt_ny']).reset_index(drop=True)
    df['bar_index'] = df.groupby('date').cumcount() + 1  # 第幾根K線，從1開始
    if not include_holidays:
        weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    else:
        weekday = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df = df.loc[df['weekday'].isin(weekday)]
    # 查看結果
    df.set_index('dt_utc', inplace=True)
    # print(df[['dt_ny', 'date', 'weekday', 'bar_index']].head(5))

    return df

def _make_df(bo_rows: int = 10, pd_rows = 3, second_leg_rows = 10, side: str = "long") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=bo_rows+pd_rows+second_leg_rows, freq="5min")
    if side == "long":
        df = pd.DataFrame([])
        df_BO = pd.DataFrame(
            {
                "open":  [100.0 + i for i in range(bo_rows)],
                "high":  [101.0 + i for i in range(bo_rows)],
                "low":   [99.0 + i for i in range(bo_rows)],
                "close": [100.5 + i for i in range(bo_rows)],
            },
            index=idx[:bo_rows],
        )
        BO_last_close = df_BO["close"].iat[-1]
        df_PB = pd.DataFrame(
            {
                "open":  [BO_last_close - i for i in range(pd_rows)],
                "high":  [BO_last_close + 1 - i for i in range(pd_rows)],
                "low":   [BO_last_close - 1 - i for i in range(pd_rows)],
                "close": [BO_last_close - 0.5 - i for i in range(pd_rows)],
            },
            index=idx[bo_rows:bo_rows+pd_rows],
        )
        pb_last_close = df_PB["close"].iat[-1]
        df_second_leg = pd.DataFrame(
            {
                "open":  [pb_last_close + i for i in range(second_leg_rows)],
                "high":  [pb_last_close + 1 + i for i in range(second_leg_rows)],
                "low":   [pb_last_close - 1 + i for i in range(second_leg_rows)],
                "close": [pb_last_close + 0.5 + i for i in range(second_leg_rows)],
            },
            index=idx[bo_rows+pd_rows:bo_rows+pd_rows+second_leg_rows],
        )
        df = pd.concat([df_BO, df_PB, df_second_leg], ignore_index=True)
        return df
    else:
        df = pd.DataFrame([])
        df_BO = pd.DataFrame(
            {
                "open":  [100.0 - i for i in range(bo_rows)],
                "high":  [101.0 - i for i in range(bo_rows)],
                "low":   [99.0 - i for i in range(bo_rows)],
                "close": [99.5 - i for i in range(bo_rows)],
            },
            index=idx[:bo_rows],
        )
        BO_last_close = df_BO["close"].iat[-1]
        df_PB = pd.DataFrame(
            {
                "open":  [BO_last_close + i for i in range(pd_rows)],
                "high":  [BO_last_close + 1 + i for i in range(pd_rows)],
                "low":   [BO_last_close - 1 + i for i in range(pd_rows)],
                "close": [BO_last_close + 0.5 + i for i in range(pd_rows)],
            },
            index=idx[bo_rows:bo_rows+pd_rows],
        )
        pb_last_close = df_PB["close"].iat[-1]
        df_second_leg = pd.DataFrame(
            {
                "open":  [pb_last_close - i for i in range(second_leg_rows)],
                "high":  [pb_last_close + 1 - i for i in range(second_leg_rows)],
                "low":   [pb_last_close - 1 - i for i in range(second_leg_rows)],
                "close": [pb_last_close - 0.5 - i for i in range(second_leg_rows)],
            },
            index=idx[bo_rows+pd_rows:bo_rows+pd_rows+second_leg_rows],
        )
        df = pd.concat([df_BO, df_PB, df_second_leg], ignore_index=True)
        return df
    
def test_al_breakout_first_pullback_required_indicators_keys():
    p = ALBreakoutFirstPullbackParams(
        break_out_series_n=3,
        break_out_n_bars=5,
        rr=1.0,
    )
    strat = ALBreakoutFirstPullbackStrategy(p)
    req = strat.required_indicators()
    # LONG & SHORT
    #"rocp_1": ("rocp", 1),
    # f"rocp_{n}": ("rocp", n),
    # "hh": ("rolling_high", self.p.break_out_n_bars, "high"),
    # "ll": ("rolling_low", self.p.break_out_n_bars, "low"),
    # "atr": ("atr", 14),
    # "ma": ("ma", 20, "close", "EMA"),
    # "streak": ("bar_streak", "ignore"),
    assert "rocp_1" in req
    assert f"rocp_{p.break_out_series_n}" in req
    assert "hh" in req
    assert "ll" in req
    assert "atr" in req
    assert "ma" in req
    assert "streak" in req

def test_al_breakout_first_pullback_generate_intents_entry_when_LONG_conditions_met():
    p = ALBreakoutFirstPullbackParams(
        break_out_series_n=3,
        break_out_n_bars=5,  # 這個單元測試不依賴 rolling_high 真實計算，直接餵 hh
        rr=2.0,
    )
    strat = ALBreakoutFirstPullbackStrategy(p)
    df = _make_df(bo_rows=10, pd_rows=3, second_leg_rows=10, side="long")
    i = 11  # 確保 i >= 2，且 i-2 不越界
    t = df.index[i]
    ctx = StrategyContext(
        i=i,
        df=df,
        indicators={
            "rocp_1": pd.Series([0.0]*len(df)),
            "rocp_3": pd.Series([0.0]*len(df)),
            "hh": ind.rolling_high(df, length=p.break_out_n_bars, column="high"),
            "ll": pd.Series([float("nan")]*len(df)),
            "atr": pd.Series([1.0]*len(df)),
            "ma": pd.Series([100.0]*len(df)),
            "streak": pd.Series(np.arange(1,11).tolist() + [-1, -2, -3] + np.arange(1,11).tolist()),  # 模擬出現反向K線
        },
        position=Position(side=None, qty=0.0),
        time = t,
        init_equity=10000.0,
        now_equity=10000.0,
    )
    intents, debug_info = strat.generate_intents(ctx)
    cond1 = debug_info["cond1"]
    cond2 = debug_info["cond2"]
    cond3 = debug_info["cond3"]
    cond4 = debug_info["cond4"]
    cond5 = debug_info["cond5"]
    streak_prev2 = debug_info["streak_prev2"]
    streak_prev = debug_info["streak_prev"]
    streak_curr = debug_info["streak_curr"]
    high_prev2 = debug_info["high_prev2"]
    hh_prev3 = debug_info["hh_prev3"]
    assert high_prev2 == 110
    assert hh_prev3 == 109
    assert streak_prev2 == 10
    assert streak_prev == -1
    assert streak_curr == -2

    assert cond1
    assert cond2
    assert cond3
    assert cond4
    assert cond5

    # 應該產生一筆 LONG ENTRY intent
    assert len(intents) == 1
    intent = intents[0] 
    assert intent.action == ActionType.ENTRY
    assert intent.side == Side.LONG
    # SL 應該設在突破點下方（i-2 根 K 線的 open）
    expected_sl = float(df["open"].iat[i-11])
    assert intent.sl_price == expected_sl
    # TP 應該設在 SL 距離的 RR 倍
    expected_tp = float(df["low"].iat[i-1]) + (float(df["low"].iat[i-1]) - expected_sl) * p.rr
    assert intent.tp_price == expected_tp
    expected_qty = (10000.0 * p.max_notional_pct) / float(df["low"].iat[i-1])
    assert intent.qty <= max(p.min_qty, float(expected_qty))

def test_al_breakout_first_pullback_generate_intents_no_entry_when_no_pullback():
    p = ALBreakoutFirstPullbackParams(
        break_out_series_n=3,
        break_out_n_bars=5,
        rr=2.0,
    )
    strat = ALBreakoutFirstPullbackStrategy(p)
    df = _make_df(bo_rows=10, pd_rows=1, second_leg_rows=10, side="long")  # 沒有回調段
    i = 10  # 確保 i >= 2，且 i-2 不越界
    t = df.index[i]
    ctx = StrategyContext(
        i=i,
        df=df,
        indicators={
            "rocp_1": pd.Series([0.0]*len(df)),
            "rocp_3": pd.Series([0.0]*len(df)),
            "hh": pd.Series([float("nan")]* (i-2) + [101.5] + [float("nan")]*(len(df)-i-1)),
            "ll": pd.Series([float("nan")]*len(df)),
            "atr": pd.Series([1.0]*len(df)),
            "ma": pd.Series([100.0]*len(df)),
            "streak": pd.Series([0]*(i-2) + [1, 1, 1] + [0]*(len(df)-i-1)),  # 沒有反向K線
        },
        position=Position(side=None, qty=0.0),
        time = t,
        init_equity=10000.0,
        now_equity=10000.0,
    )
    intents = strat.generate_intents(ctx)
    # 不應該產生任何 intent
    assert len(intents) == 0

def test_max_position_size_limit():
    # 造一段簡單 OHLC
    coin_name = 'BTC'  # 可更改為 'ETH', 'SOL', 'BTC', 'ADA', 'PAXG' 等等
    nM = 48
    timeframe = "5m"
    section = "UTC"

    df = load_crypto_parquet_data(coin_name=coin_name, timeframe=timeframe, nM=nM, section=section)
    df = generate_allday_bars_info(df, include_holidays=False)

    cfg = BacktestConfig(initial_cash=10000, fee_rate=0.0, slippage_bps=0.0, conservative_intrabar=True)
    engine = BacktestEngine(cfg)
    strat = ALBreakoutFirstPullbackStrategy(ALBreakoutFirstPullbackParams(break_out_series_n=3, break_out_n_bars=5, BO_n_times_atr=0, max_notional_pct=1.6, min_qty=0.0001, sl_atr_like=0.0, rr=2, time_exit_bars=50))  # 故意下大單

    result = engine.run(df, strat)

    # 檢查沒有超過初始資金的持倉
    assert len(result.trades) > 0

    for trade in result.trades:
        sl_range = abs(trade.entry_price - trade.sl_price)
        max_notional_lose = cfg.initial_cash * (strat.p.max_notional_pct)
        max_qty = max_notional_lose / sl_range if sl_range > 0 else float('inf')
        assert trade.qty <= max_qty
        # assert -max_notional*1.1 > trade.pnl

def test_al_breakout_first_pullback_strategy_state():
    p = ALBreakoutFirstPullbackParams(
        break_out_series_n=3,
        break_out_n_bars=5,
        n_bar_pivot=2,
        rr=2.0,
    )
    strat = ALBreakoutFirstPullbackStrategy(p)

    df = _make_df(bo_rows=10, pd_rows=3, second_leg_rows=10, side="long")
    i = 17  # 確保 i >= 2，且 i-2 不越界
    t = df.index[i]
    ctx = StrategyContext(
        i=i,
        df=df,
        indicators={
            "rocp_1": pd.Series([0.0]*len(df)),
            "rocp_3": pd.Series([0.0]*len(df)),
            "hh": ind.rolling_high(df, length=p.break_out_n_bars, column="high"),
            "ll": pd.Series([float("nan")]*len(df)),
            "atr": pd.Series([1.0]*len(df)),
            "ma": pd.Series([100.0]*len(df)),
            "streak": pd.Series(np.arange(1,11).tolist() + [-1, -2, -3] + np.arange(1,11).tolist()),  # 模擬出現反向K線
            "ll_streak": ind.hh_ll_streak(df = df, side = "ll"),
            "hh_streak": ind.hh_ll_streak(df = df, side = "hh"),
            "pivot_low_mask": ind.pivot_mask(df = df, side = "low", k = 3),
        },
        position=Position(side=Side.LONG, qty=1.0),
        time = t,
        init_equity=10000.0,
        now_equity=10000.0,
    )
    strat.state = "UPDATE_SL"
    ctx.position.sl_price = 99  # 模擬已經有倉位和停損價
    strat.pb_start_i = 12  # 模擬 pullback 開始 index
    intents, debug_info = strat.generate_intents(ctx)
    # 模擬進場後，檢查 state 是否更新

    assert len(intents) == 0
    assert debug_info["state"] == "UPDATE_SL"
    # assert intents[0].sl_price == 106
    # assert debug_info["state"] == "PB"

    
