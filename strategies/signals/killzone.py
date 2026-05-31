"""
ICT Killzones — Session-Based Time Filter
------------------------------------------
Institutional algo activity peaks at specific UTC windows.
Only signals inside these windows have historically high accuracy.

Global Sessions (UTC) — crypto, forex, commodities:
  Asia          : 00:00 – 04:00  (relevant for BTC/crypto)
  London Open   : 07:00 – 10:00  ← highest accuracy
  NY AM Open    : 12:00 – 15:00  ← second highest
  NY PM / Close : 18:00 – 20:00  ← lower reliability
  Silver Bullet : 10:00–11:00 and 15:00–16:00 UTC

Indian Market Sessions (UTC = IST − 5:30):
  IST Opening   : 03:45 – 05:30 UTC (9:15 – 11:00 IST) — highest volatility
  IST Midday    : 06:00 – 08:00 UTC (11:30 – 13:30 IST) — breakout continuation
  IST Pre-Close : 08:30 – 10:00 UTC (14:00 – 15:30 IST) — institutional closing

Use build_killzone(df, market="india") for NSE stocks/indices.
Default market="global" for crypto/forex/commodities.
"""
from __future__ import annotations
import pandas as pd

# (start_hour_utc, end_hour_utc, name, weight)
KILLZONES_GLOBAL = [
    (0,  4,  "asia",          0.70),
    (7,  10, "london",        1.00),
    (10, 11, "silver_bullet", 1.00),
    (12, 15, "ny_am",         0.90),
    (15, 16, "silver_bullet", 1.00),
    (18, 20, "ny_pm",         0.50),
]

# Indian market killzones (UTC times)
# NSE trades Mon–Fri 9:15–15:30 IST = 3:45–10:00 UTC
KILLZONES_INDIA = [
    (3,  6,  "ist_open",      1.00),   # 9:15–11:30 IST — opening momentum
    (6,  8,  "ist_midday",    0.75),   # 11:30–13:30 IST — continuation
    (8,  10, "ist_preclose",  0.90),   # 14:00–15:30 IST — institutional activity
]

_MARKET_MAP = {
    "global":    KILLZONES_GLOBAL,
    "crypto":    KILLZONES_GLOBAL,
    "forex":     KILLZONES_GLOBAL,
    "commodity": KILLZONES_GLOBAL,
    "india":     KILLZONES_INDIA,
    "india_equity": KILLZONES_INDIA,
    "india_index":  KILLZONES_INDIA,
}


def _get_hours(index: pd.DatetimeIndex) -> pd.Series:
    if hasattr(index, "tz") and index.tz is not None:
        return pd.Series(index.tz_convert("UTC").hour, index=index)
    return pd.Series(index.hour, index=index)


def in_killzone(index: pd.DatetimeIndex, market: str = "global") -> pd.Series:
    """Return bool Series — True if timestamp is inside any killzone."""
    hours = _get_hours(index)
    kzs   = _MARKET_MAP.get(market, KILLZONES_GLOBAL)
    result = pd.Series(False, index=index)
    for start, end, _, _ in kzs:
        result |= (hours >= start) & (hours < end)
    return result


def killzone_name(index: pd.DatetimeIndex, market: str = "global") -> pd.Series:
    """Return string Series with killzone name or 'none'."""
    hours = _get_hours(index)
    kzs   = _MARKET_MAP.get(market, KILLZONES_GLOBAL)
    result = pd.Series("none", index=index)
    for start, end, name, _ in kzs:
        result[((hours >= start) & (hours < end))] = name
    return result


def killzone_weight(index: pd.DatetimeIndex, market: str = "global") -> pd.Series:
    """Return float Series with confidence weight for current killzone."""
    hours = _get_hours(index)
    kzs   = _MARKET_MAP.get(market, KILLZONES_GLOBAL)
    result = pd.Series(0.0, index=index)
    for start, end, _, weight in kzs:
        result[((hours >= start) & (hours < end))] = weight
    return result


def build_killzone(df: pd.DataFrame, market: str = "global") -> pd.DataFrame:
    """Attach killzone columns to df. market='india' for NSE stocks/indices."""
    df = df.copy()
    df["in_killzone"]    = in_killzone(df.index, market)
    df["session"]        = killzone_name(df.index, market)
    df["session_weight"] = killzone_weight(df.index, market)
    return df
