"""
Paper Trading Engine
====================
Runs two models in parallel on live Binance data.
Manages paper positions, calculates P&L, tracks 30-day performance.

No real money. No real orders. Zero execution risk.
"""
from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """An open paper trade."""
    symbol:     str
    direction:  str        # "long" or "short"
    entry_price: float
    sl:         float
    tp1:        float
    tp2:        Optional[float]
    size:       float      # base units (e.g. BTC quantity)
    dollar_risk: float     # $ risked
    entry_time: str        # ISO timestamp
    model_name: str
    confidence: float
    atr:        float = 0.0
    trail_atr_mult: float = 1.0
    meta:       dict = field(default_factory=dict)
    partial_exit_done: bool = False
    trailing_sl: Optional[float] = None   # None = not activated yet
    original_sl: float = 0.0              # stored at entry for R calculation


@dataclass
class ClosedTrade:
    """A completed paper trade."""
    symbol:     str
    direction:  str
    entry_price: float
    exit_price:  float
    sl:         float
    tp1:        float
    size:       float
    dollar_risk: float
    pnl:        float      # dollar P&L
    rr:         float      # R-multiple achieved
    outcome:    str        # "win_tp1" | "win_tp2" | "loss_sl" | "timeout"
    entry_time: str
    exit_time:  str
    model_name: str
    confidence: float
    meta:       dict = field(default_factory=dict)


class PaperAccount:
    """
    Tracks one model's paper trading account.
    Starting capital: $10,000.  Risk: 1% per trade.
    """

    def __init__(self, model_name: str, starting_capital: float, risk_pct: float,
                 trades_dir: str, max_dd_pct: float = 0.10):
        self.model_name      = model_name
        self.capital         = starting_capital
        self.starting_capital = starting_capital
        self.risk_pct        = risk_pct
        self.max_dd_pct      = max_dd_pct
        self.trades_dir      = trades_dir
        self.peak_capital    = starting_capital

        self.open_positions: list[PaperPosition] = []
        self.closed_trades:  list[ClosedTrade]   = []

        self.trades_file = os.path.join(trades_dir, f"{model_name}_trades.jsonl")
        self._load_trades()

        self.kill_switch_active = False

    # ── Persistence ────────────────────────────────────────────────────────────
    def _load_trades(self):
        """Load existing closed trades from disk on startup."""
        if os.path.exists(self.trades_file):
            with open(self.trades_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            self.closed_trades.append(ClosedTrade(**d))
                        except Exception:
                            pass
            # Recalculate capital from closed trades
            self.capital = self.starting_capital + sum(t.pnl for t in self.closed_trades)
            self.peak_capital = max(self.starting_capital,
                                    *[self.starting_capital + sum(t.pnl for t in self.closed_trades[:i+1])
                                      for i in range(len(self.closed_trades))]) if self.closed_trades else self.starting_capital
            logger.info("[%s] Loaded %d closed trades. Capital: $%.2f",
                        self.model_name, len(self.closed_trades), self.capital)

    def _save_trade(self, trade: ClosedTrade):
        os.makedirs(os.path.dirname(self.trades_file), exist_ok=True)
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(asdict(trade)) + "\n")

    # ── Signal Processing ──────────────────────────────────────────────────────
    def process_signal(self, setup) -> bool:
        """
        Process a new trade setup from the strategy.
        Returns True if a paper order was placed.
        """
        if self.kill_switch_active:
            logger.warning("[%s] Kill switch active — no new trades", self.model_name)
            return False

        # Check drawdown kill switch
        if self.capital < self.starting_capital * (1 - self.max_dd_pct):
            self.kill_switch_active = True
            logger.warning("[%s] Kill switch fired! Capital %.1f%% below start",
                           self.model_name,
                           (self.starting_capital - self.capital) / self.starting_capital * 100)
            return False

        # No duplicate positions on same symbol+direction
        for pos in self.open_positions:
            if pos.symbol == setup.symbol and pos.direction == setup.direction:
                return False

        # Position sizing: 1% of current capital
        dollar_risk = self.capital * self.risk_pct
        price_risk  = abs(setup.entry - setup.sl)
        if price_risk <= 0:
            return False

        size = dollar_risk / price_risk   # units of base asset

        pos = PaperPosition(
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=setup.entry,
            sl=setup.sl,
            tp1=setup.tp1,
            tp2=setup.tp2,
            size=size,
            dollar_risk=dollar_risk,
            entry_time=datetime.now(timezone.utc).isoformat(),
            model_name=self.model_name,
            confidence=setup.confidence,
            atr=setup.atr,
            trail_atr_mult=getattr(setup, "trail_atr_mult", 1.0),
            original_sl=setup.sl,
            meta=setup.meta or {},
        )
        self.open_positions.append(pos)
        logger.info("[%s] 📥 PAPER %s %s @ %.6f  SL=%.6f  TP1=%.6f  Risk=$%.2f",
                    self.model_name, setup.direction.upper(), setup.symbol,
                    setup.entry, setup.sl, setup.tp1, dollar_risk)
        return True

    # ── Position Update ─────────────────────────────────────────────────────────
    def update_positions(self, current_prices: dict[str, float]):
        """
        Full position lifecycle — matches backtest engine exactly:
          1. Hard SL or trailing SL hit → close
          2. TP1 hit → 50% partial exit, SL → breakeven, trailing activates
          3. Trailing SL updates (only moves in profit direction)
          4. TP2 hit → close runner
        """
        to_close = []

        for pos in self.open_positions:
            price = current_prices.get(pos.symbol, 0.0)
            if price <= 0:
                continue

            atr = pos.atr if pos.atr > 0 else abs(pos.entry_price - pos.sl) * 0.5
            sl_active = pos.trailing_sl if pos.trailing_sl is not None else pos.sl

            if pos.direction == "long":
                # Hard SL or trailing SL hit
                if price <= sl_active:
                    outcome = "trail_stop" if pos.trailing_sl is not None else "loss_sl"
                    if pos.partial_exit_done:
                        runner_pnl = (sl_active - pos.entry_price) * pos.size * 0.5
                        rr = (sl_active - pos.entry_price) / max(pos.entry_price - (pos.original_sl or pos.sl), 1e-9)
                        to_close.append((pos, sl_active, outcome, runner_pnl, rr))
                    else:
                        to_close.append((pos, sl_active, "loss_sl", -pos.dollar_risk, -1.0))
                    continue

                # TP1 partial exit
                if not pos.partial_exit_done and price >= pos.tp1:
                    raw_pnl = (pos.tp1 - pos.entry_price) * pos.size * 0.5
                    rr = (pos.tp1 - pos.entry_price) / max(pos.entry_price - pos.sl, 1e-9)
                    pos.partial_exit_done = True
                    pos.sl           = pos.entry_price
                    pos.trailing_sl  = pos.entry_price
                    self.capital += raw_pnl
                    if self.capital > self.peak_capital: self.peak_capital = self.capital
                    logger.info("[%s] 📤 TP1 PARTIAL LONG %s  +$%.2f  R=+%.2fR  [Trail: BE]",
                                self.model_name, pos.symbol, raw_pnl, rr)

                # Update trailing SL (moves UP only)
                if pos.partial_exit_done and pos.trailing_sl is not None:
                    new_trail = price - pos.trail_atr_mult * atr
                    if new_trail > pos.trailing_sl:
                        pos.trailing_sl = new_trail

                # TP2 runner exit
                if pos.tp2 and price >= pos.tp2 and pos.partial_exit_done:
                    runner_pnl = (pos.tp2 - pos.entry_price) * pos.size * 0.5
                    rr = (pos.tp2 - pos.entry_price) / max(pos.entry_price - (pos.original_sl or pos.sl), 1e-9)
                    to_close.append((pos, pos.tp2, "win_tp2", runner_pnl, rr))

            else:  # short
                if price >= sl_active:
                    outcome = "trail_stop" if pos.trailing_sl is not None else "loss_sl"
                    if pos.partial_exit_done:
                        runner_pnl = (pos.entry_price - sl_active) * pos.size * 0.5
                        rr = (pos.entry_price - sl_active) / max((pos.original_sl or pos.sl) - pos.entry_price, 1e-9)
                        to_close.append((pos, sl_active, outcome, runner_pnl, rr))
                    else:
                        to_close.append((pos, sl_active, "loss_sl", -pos.dollar_risk, -1.0))
                    continue

                if not pos.partial_exit_done and price <= pos.tp1:
                    raw_pnl = (pos.entry_price - pos.tp1) * pos.size * 0.5
                    rr = (pos.entry_price - pos.tp1) / max(pos.sl - pos.entry_price, 1e-9)
                    pos.partial_exit_done = True
                    pos.sl           = pos.entry_price
                    pos.trailing_sl  = pos.entry_price
                    self.capital += raw_pnl
                    if self.capital > self.peak_capital: self.peak_capital = self.capital
                    logger.info("[%s] 📤 TP1 PARTIAL SHORT %s  +$%.2f  R=+%.2fR  [Trail: BE]",
                                self.model_name, pos.symbol, raw_pnl, rr)

                if pos.partial_exit_done and pos.trailing_sl is not None:
                    new_trail = price + pos.trail_atr_mult * atr
                    if new_trail < pos.trailing_sl:
                        pos.trailing_sl = new_trail

                if pos.tp2 and price <= pos.tp2 and pos.partial_exit_done:
                    runner_pnl = (pos.entry_price - pos.tp2) * pos.size * 0.5
                    rr = (pos.entry_price - pos.tp2) / max((pos.original_sl or pos.sl) - pos.entry_price, 1e-9)
                    to_close.append((pos, pos.tp2, "win_tp2", runner_pnl, rr))

        # ── Close all flagged positions ────────────────────────────────────────
        for pos, exit_price, outcome, pnl, rr in to_close:
            if pos in self.open_positions:
                self.open_positions.remove(pos)
            self.capital += pnl
            if self.capital > self.peak_capital: self.peak_capital = self.capital

            trade = ClosedTrade(
                symbol=pos.symbol, direction=pos.direction,
                entry_price=pos.entry_price, exit_price=exit_price,
                sl=pos.sl, tp1=pos.tp1, size=pos.size, dollar_risk=pos.dollar_risk,
                pnl=round(pnl, 4), rr=round(rr, 3), outcome=outcome,
                entry_time=pos.entry_time,
                exit_time=datetime.now(timezone.utc).isoformat(),
                model_name=pos.model_name, confidence=pos.confidence, meta=pos.meta,
            )
            self.closed_trades.append(trade)
            self._save_trade(trade)

            icon = "✅" if pnl >= 0 else "❌"
            logger.info("[%s] %s CLOSED %s %s @ %.6g  PnL=$%.2f  R=%.2f  [%s]",
                        self.model_name, icon, pos.direction.upper(), pos.symbol,
                        exit_price, pnl, rr, outcome)

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Compute current performance statistics."""
        trades = self.closed_trades
        if not trades:
            return {
                "capital": self.capital,
                "return_pct": 0.0,
                "total_trades": 0,
                "open_trades": len(self.open_positions),
                "win_rate": 0.0,
                "avg_rr": 0.0,
                "max_dd": 0.0,
                "total_pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "today_trades": 0,
                "today_pnl": 0.0,
                "kill_switch": self.kill_switch_active,
            }

        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        today = datetime.now(timezone.utc).date().isoformat()
        today_trades = [t for t in trades if t.exit_time[:10] == today]

        # Running max drawdown
        running_capital = self.starting_capital
        peak = self.starting_capital
        max_dd = 0.0
        for t in trades:
            running_capital += t.pnl
            if running_capital > peak:
                peak = running_capital
            dd = (peak - running_capital) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "capital":       round(self.capital, 2),
            "return_pct":    round((self.capital / self.starting_capital - 1) * 100, 2),
            "total_trades":  len(trades),
            "open_trades":   len(self.open_positions),
            "win_rate":      round(len(wins) / len(trades) * 100, 1),
            "avg_rr":        round(sum(t.rr for t in trades) / len(trades), 2),
            "max_dd":        round(max_dd * 100, 2),
            "total_pnl":     round(sum(t.pnl for t in trades), 2),
            "wins":          len(wins),
            "losses":        len(losses),
            "today_trades":  len(today_trades),
            "today_pnl":     round(sum(t.pnl for t in today_trades), 2),
            "kill_switch":   self.kill_switch_active,
        }
