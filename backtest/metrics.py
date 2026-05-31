from __future__ import annotations
import numpy as np
import pandas as pd

from backtest.trade import Trade, TradeState


def compute_metrics(trades: list[Trade], equity_curve: list[tuple]) -> dict:
    closed = [t for t in trades if t.state != TradeState.OPEN]
    if not closed:
        return {"total_trades": 0, "win_rate": 0.0, "expectancy": 0.0, "max_drawdown": 0.0}

    pnls = [t.pnl for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(closed)
    loss_rate = 1 - win_rate
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    entry_prices = [t.entry_price for t in closed]
    sl_prices = [t.sl_price for t in closed]
    risks = [abs(e - s) for e, s in zip(entry_prices, sl_prices)]
    rr_list = [p / r for p, r in zip(pnls, risks) if r > 0]
    avg_rr = float(np.mean(rr_list)) if rr_list else 0.0

    equity_series = equity_to_series(equity_curve)
    mdd = max_drawdown(equity_series)
    sharpe = sharpe_ratio(equity_series)

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "avg_rr": round(avg_rr, 4),
        "expectancy": round(expectancy, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(mdd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


def equity_to_series(equity_curve: list[tuple]) -> pd.Series:
    if not equity_curve:
        return pd.Series(dtype=float)
    timestamps, values = zip(*equity_curve)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    drawdowns = (equity - rolling_max) / rolling_max
    return float(abs(drawdowns.min()))


def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252 * 96) -> float:
    """
    periods_per_year: 252 trading days * 96 fifteen-minute bars/day ≈ 24192.
    Adjust if using different timeframe.
    """
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * (periods_per_year ** 0.5))


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print("=" * 50 + "\n")
