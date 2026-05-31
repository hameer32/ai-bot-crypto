from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from risk.drawdown_guard import DrawdownGuard
from strategies import get_strategy
from strategies.base import TradeSetup

logger = logging.getLogger(__name__)

POLL_INTERVAL = 60  # seconds between candle polls


class PaperTrader:
    """
    Paper trading loop using the BaseStrategy framework.
    Strategy is selected via cfg.active_strategy (e.g. plan_a_swing, plan_b_scalp).
    Logs all trade events to JSONL. No real orders placed.
    """

    def __init__(self, cfg, log_path: str = "logs/paper_trades.jsonl"):
        self.cfg = cfg
        self.log_path = log_path
        self.capital = cfg.risk.initial_capital
        self.guard = DrawdownGuard(self.capital, cfg.risk.max_drawdown_pct)

        strategy_name = getattr(cfg, "active_strategy", "original_smc")
        self.strategy = get_strategy(strategy_name, cfg)

        # Open position per symbol: {symbol: {"setup": TradeSetup, "entry_time": ts, "sl": float, ...}}
        self.open_positions: dict[str, dict] = {}
        self._trade_id = 0
        self._all_data: dict[tuple[str, str], pd.DataFrame] = {}
        self._prepared = False

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    # ── Data ingestion ────────────────────────────────────────────────────────

    def update_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Update the data buffer for one (symbol, timeframe) pair."""
        self._all_data[(symbol, timeframe)] = df
        self._prepared = False  # re-prepare on next tick

    def _ensure_prepared(self) -> None:
        if not self._prepared and self._all_data:
            self.strategy.prepare(self._all_data)
            self._prepared = True

    # ── Per-bar logic ─────────────────────────────────────────────────────────

    def tick(self, symbol: str, ts: pd.Timestamp, row: pd.Series) -> None:
        """Process one closed candle on the strategy's primary (LTF) timeframe."""
        if not self.guard.update(self.capital):
            logger.warning("Kill switch active — not trading %s", symbol)
            return

        self._ensure_prepared()

        # Update any open position with current bar prices
        self._update_open_position(symbol, ts, row)

        # If already in a position for this symbol, skip new signals
        if symbol in self.open_positions:
            return

        setups: list[TradeSetup] = self.strategy.on_bar(symbol, ts, row, self.capital)
        if not setups:
            return

        # Take the highest-confidence setup
        setup = max(setups, key=lambda s: s.confidence)
        self._open_position(symbol, ts, setup)

    def _update_open_position(self, symbol: str, ts: pd.Timestamp, row: pd.Series) -> None:
        pos = self.open_positions.get(symbol)
        if pos is None:
            return

        hi = row.get("high", row.get("close"))
        lo = row.get("low", row.get("close"))
        setup: TradeSetup = pos["setup"]

        hit_sl = (setup.direction == "long" and lo <= pos["sl"]) or \
                 (setup.direction == "short" and hi >= pos["sl"])
        hit_tp1 = (setup.direction == "long" and hi >= setup.tp1) or \
                  (setup.direction == "short" and lo <= setup.tp1)  # tp1 is now 4R
        hit_tp2 = setup.tp2 is not None and (
            (setup.direction == "long" and hi >= setup.tp2) or
            (setup.direction == "short" and lo <= setup.tp2)
        )

        if hit_sl:
            pnl = pos["size"] * (pos["sl"] - setup.entry) * (1 if setup.direction == "long" else -1)
            self._close_position(symbol, ts, pos["sl"], pnl, "sl")
        elif hit_tp2:
            pnl = pos["size"] * (setup.tp2 - setup.entry) * (1 if setup.direction == "long" else -1)
            self._close_position(symbol, ts, setup.tp2, pnl, "tp2")
        elif hit_tp1 and not pos.get("partial_taken"):
            # 50% exit at TP1, trail SL to breakeven
            half_size = pos["size"] / 2
            pnl = half_size * (setup.tp1 - setup.entry) * (1 if setup.direction == "long" else -1)
            self.capital += pnl
            pos["size"] = half_size
            pos["sl"] = setup.entry  # trail to breakeven
            pos["partial_taken"] = True
            self._log_event("tp1_partial", symbol, ts, setup, pnl)
            logger.info("PAPER TP1 (partial) %s %s pnl=%.2f", symbol, setup.direction, pnl)

    def _open_position(self, symbol: str, ts: pd.Timestamp, setup: TradeSetup) -> None:
        risk_per_unit = abs(setup.entry - setup.sl)
        if risk_per_unit <= 0:
            return

        max_risk = self.capital * self.cfg.risk.risk_pct_per_trade
        size = max_risk / risk_per_unit

        self._trade_id += 1
        self.open_positions[symbol] = {
            "id": self._trade_id,
            "setup": setup,
            "entry_time": ts,
            "size": size,
            "sl": setup.sl,
            "partial_taken": False,
        }

        self._log_event("open", symbol, ts, setup, size=size)
        logger.info(
            "PAPER OPEN  %s %-5s @ %.4f  SL=%.4f  TP1=%.4f  conf=%.2f",
            symbol, setup.direction, setup.entry, setup.sl, setup.tp1, setup.confidence,
        )

    def _close_position(self, symbol: str, ts: pd.Timestamp, price: float, pnl: float, reason: str) -> None:
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return
        self.capital += pnl
        self.guard.update(self.capital)
        self._log_event("close", symbol, ts, pos["setup"], pnl=pnl, exit_price=price, reason=reason)
        logger.info("PAPER CLOSE %s %s pnl=%.2f cap=%.2f [%s]",
                    symbol, pos["setup"].direction, pnl, self.capital, reason)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, symbol: str, ts: pd.Timestamp,
                   setup: TradeSetup, pnl: float = 0.0, size: float = 0.0,
                   exit_price: float = 0.0, reason: str = "") -> None:
        entry = {
            "type": event_type,
            "timestamp": str(ts),
            "symbol": symbol,
            "direction": setup.direction,
            "entry": setup.entry,
            "sl": setup.sl,
            "tp1": setup.tp1,
            "tp2": setup.tp2,
            "confidence": setup.confidence,
            "atr": setup.atr,
            "pnl": round(pnl, 4),
            "size": round(size, 6),
            "exit_price": exit_price,
            "reason": reason,
            "capital": round(self.capital, 2),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "strategy": self.strategy.name,
            "capital": round(self.capital, 2),
            "open_positions": {k: v["setup"].direction for k, v in self.open_positions.items()},
            "drawdown": round(self.guard.current_drawdown, 4),
            "kill_switch_active": not self.guard.active,
        }
