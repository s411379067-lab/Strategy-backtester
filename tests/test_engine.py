import pandas as pd

from backtester.engine import BacktestEngine
from backtester.models import BacktestConfig
from backtester.strategies.xyz_strategy import XYZStrategy, XYZParams
from backtester.strategies.ALBO_strategy import ALBOStrategy, ALBOParams

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

    # 對齊 index，且長度相同
    assert result.equity_curve.index.equals(df.index)
    assert len(result.equity_curve) == len(df)

    # 最終 equity 有變化（代表有交易）

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
    strat = ALBOStrategy(ALBOParams(break_out_series_n=3, break_out_n_bars=5, BO_n_times_atr=0, max_notional_pct=1.6, min_qty=0.0001, sl_atr_like=0.0, rr=2, time_exit_bars=50))  # 故意下大單

    result = engine.run(df, strat)

    # 檢查沒有超過初始資金的持倉
    for trade in result.trades:
        sl_range = abs(trade.entry_price - trade.sl_price)
        max_notional_lose = cfg.initial_cash * (strat.p.max_notional_pct / 100)
        max_qty = max_notional_lose / sl_range if sl_range > 0 else float('inf')
        assert trade.qty <= max_qty
        # assert -max_notional*1.1 > trade.pnl
    assert len(result.trades) > 0