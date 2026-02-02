from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .models import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def basic_metrics(result: BacktestResult) -> Dict[str, float]:
    trades = result.trades
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": max_drawdown(result.equity_curve),
        }
    
    win_trades = [t for t in trades if t.pnl > 0]
    loss_trades = [t for t in trades if t.pnl <= 0]

    wins_held_bars = np.mean([t.bars_held for t in win_trades]) if win_trades else 0.0
    losses_held_bars = np.mean([t.bars_held for t in loss_trades]) if loss_trades else 0.0

    avg_wins_pnl = np.mean([t.pnl for t in win_trades]) if win_trades else 0.0
    avg_losses_pnl = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0.0

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    pay_off_ratio = float(avg_wins_pnl / -avg_losses_pnl) if avg_losses_pnl < 0 else float("inf")
    std = pnls.std(ddof=1) if len(pnls) > 1 else 0.0

    daily_equity = result.equity_curve.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()


    sharpe_ratio_by_day = (daily_returns.mean() / daily_returns.std(ddof=1)) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
    sharpe_ratio = float(sharpe_ratio_by_day)

    total_days = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days
    epsilon = 1e-8
    pct_non_zero_days = float((daily_returns.abs() > epsilon).sum())

    total_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1

    return {
        "trades": float(len(trades)),
        "win_rate": float((pnls > 0).mean()),
        "avg_pnl": float(pnls.mean()),
        "profit_factor": pf,
        "pay_off_ratio": pay_off_ratio,
        "max_drawdown": max_drawdown(result.equity_curve),
        "sharpe_ratio": sharpe_ratio,
        "avg_win_pnl": avg_wins_pnl,
        "avg_loss_pnl": avg_losses_pnl,
        "avg_win_held_bars": wins_held_bars,
        "avg_loss_held_bars": losses_held_bars,
        "pct_non_zero_days": pct_non_zero_days,
        "total_days": float(total_days),
        "total_return": total_return,
    }
