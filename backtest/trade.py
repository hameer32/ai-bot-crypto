from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class TradeState(Enum):
    OPEN = "open"
    CLOSED_TP1 = "closed_tp1"
    CLOSED_SL = "closed_sl"
    CLOSED_TRAIL = "closed_trail"
    CLOSED_EOD = "closed_eod"


@dataclass
class Trade:
    id: int
    symbol: str
    direction: str          # 'long' or 'short'
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp1_price: float
    trail_activation: float
    size: float
    confidence: float
    atr_at_entry: float
    rr_target: float = 2.0

    # mutable state
    state: TradeState = TradeState.OPEN
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    partial_exit_done: bool = False
    trailing_sl: Optional[float] = None
    pnl: float = 0.0
    remaining_size: float = 0.0

    def __post_init__(self):
        self.remaining_size = self.size

    def risk(self) -> float:
        return abs(self.entry_price - self.sl_price) * self.size

    def is_open(self) -> bool:
        return self.state == TradeState.OPEN

    def pnl_per_unit(self, price: float) -> float:
        if self.direction == "long":
            return price - self.entry_price
        return self.entry_price - price

    def unrealized_pnl(self, price: float) -> float:
        return self.pnl_per_unit(price) * self.remaining_size
