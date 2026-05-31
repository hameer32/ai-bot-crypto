from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TradeSetup:
    timestamp: pd.Timestamp
    symbol: str
    direction: str          # 'long' or 'short'
    entry_price: float
    sl_price: float
    tp1_price: float        # partial exit at 2R
    trail_activation: float # price at which trailing SL activates (1R)
    position_size: float
    confidence: float
    atr_at_entry: float
    rr_target: float = 2.0


def compute_sl(row: pd.Series, direction: str, atr_k: float) -> float:
    atr_val = row.get("atr", 0)
    if direction == "long":
        level = row.get("sweep_low_level", row.get("roll_low", row["low"]))
        if pd.isna(level):
            level = row["low"]
        return float(level) - atr_k * float(atr_val)
    else:
        level = row.get("sweep_high_level", row.get("roll_high", row["high"]))
        if pd.isna(level):
            level = row["high"]
        return float(level) + atr_k * float(atr_val)


def compute_tp(entry: float, sl: float, rr: float, direction: str) -> float:
    risk = abs(entry - sl)
    if direction == "long":
        return entry + risk * rr
    return entry - risk * rr


def dynamic_rr(vol_regime: str, structure: str) -> float:
    if vol_regime == "high" and structure in ("bullish", "bearish"):
        return 3.0
    if vol_regime == "low":
        return 1.5
    return 2.0


def build_trade_setup(
    row: pd.Series,
    direction: str,
    capital: float,
    risk_pct: float,
    atr_k: float,
    partial_exit_rr: float,
    symbol: str,
    confidence: float,
) -> TradeSetup | None:
    entry = float(row["close"])
    sl = compute_sl(row, direction, atr_k)
    risk_per_unit = abs(entry - sl)

    if risk_per_unit <= 0:
        return None

    rr = dynamic_rr(
        str(row.get("vol_regime", "normal")),
        str(row.get("htf_structure", "neutral")),
    )
    tp1 = compute_tp(entry, sl, partial_exit_rr, direction)
    trail_activation = compute_tp(entry, sl, 1.0, direction)
    size = (capital * risk_pct) / risk_per_unit

    return TradeSetup(
        timestamp=row.name if hasattr(row, "name") else row.get("timestamp"),
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        trail_activation=trail_activation,
        position_size=size,
        confidence=confidence,
        atr_at_entry=float(row.get("atr", 0)),
        rr_target=rr,
    )
