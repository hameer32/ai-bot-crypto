from __future__ import annotations


def position_size(
    capital: float,
    risk_pct: float,
    entry: float,
    sl: float,
) -> float:
    risk_amount = capital * risk_pct
    risk_per_unit = abs(entry - sl)
    if risk_per_unit == 0:
        return 0.0
    return risk_amount / risk_per_unit
