"""
BaseStrategy — every strategy must implement this interface.

A strategy receives pre-loaded OHLCV DataFrames for all required timeframes,
prepares its internal state (indicators, zone maps), then answers one question
per bar: what trades should be opened right now?

The backtest runner, paper trader, and optimizer all work through this interface,
so swapping strategies requires only changing the active_strategy config key.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
import pandas as pd


@dataclass
class TradeSetup:
    """A fully specified trade proposal returned by a strategy."""
    symbol: str
    direction: Literal["long", "short"]
    entry: float
    sl: float
    tp1: float                         # first partial-exit target (4R default)
    tp2: float | None = None           # optional runner target (HTF liquidity)
    confidence: float = 0.5            # 0–1 fusion score
    atr: float = 0.0
    trail_atr_mult: float = 1.0        # trailing SL = 1×ATR below/above low/high after TP1
    meta: dict = field(default_factory=dict)  # tier scores, zone info, etc.


class BaseStrategy(ABC):
    """
    Abstract base for all strategies.

    Lifecycle
    ---------
    1. runner calls  strategy.prepare(all_data)          — once per backtest
    2. runner calls  strategy.on_bar(symbol, ts, row)    — once per LTF bar
       └─ returns list[TradeSetup] (empty = no signal)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier matching the folder name and config key."""

    @property
    @abstractmethod
    def required_timeframes(self) -> dict[str, list[str]]:
        """
        Map of symbol → list of timeframes this strategy needs.
        Example: {"BTCUSDT": ["1w", "1d", "4h", "1h", "15m", "5m"]}
        """

    @abstractmethod
    def prepare(self, all_data: dict[tuple[str, str], pd.DataFrame]) -> None:
        """
        Called once before the bar loop.
        Build indicators, zone maps, HTF signal series here.
        all_data keys are (symbol, timeframe).
        """

    @abstractmethod
    def on_bar(
        self,
        symbol: str,
        ts: pd.Timestamp,
        row: pd.Series,
        capital: float,
    ) -> list[TradeSetup]:
        """
        Called on each closed LTF bar.
        Returns zero or more TradeSetup objects.
        Must NOT look ahead — only use data up to and including ts.
        """
