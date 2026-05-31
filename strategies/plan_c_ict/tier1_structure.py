"""
Plan C ICT — Tier 1: Daily + Weekly Structure
=============================================
Determines macro bias using ICT's BOS/CHoCH model on Daily and Weekly data.
Also identifies the nearest Daily/Weekly liquidity pools (BSL/SSL) as TP targets.

Outputs merged onto LTF (15m) index:
  t1_bias         : 'bullish' | 'bearish' | 'neutral'
  t1_strength     : float [0,1]  — how decisive the bias is
  t1_choch        : bool         — CHoCH occurred in last 10 Daily bars
  t1_structure_d  : str          — Daily structure label
  t1_structure_w  : str          — Weekly structure label
  t1_in_discount  : bool         — price below Daily 50% equilibrium (valid for longs)
  t1_in_premium   : bool         — price above Daily 50% equilibrium (valid for shorts)
  t1_in_ote       : bool         — price in Daily OTE zone (61.8-79%)
  t1_target_long  : float        — nearest Daily BSL above price (TP destination)
  t1_target_short : float        — nearest Daily SSL below price (TP destination)
  t1_pd_zone      : str          — 'premium'|'discount'|'equilibrium'
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.signals.bos_choch import detect_bos_choch
from strategies.signals.premium_discount import build_premium_discount
from strategies.signals.inducement import build_inducement


def build_tier1(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    swing_window_d: int = 3,
    swing_window_w: int = 2,
    choch_lookback: int = 10,
    target_lookback: int = 50,
) -> pd.DataFrame:
    """
    Build T1 bias and merge onto ltf_df.
    """
    # ── Daily analysis ────────────────────────────────────────────────────────
    d = detect_bos_choch(daily_df, swing_window=swing_window_d)
    d = build_premium_discount(d, swing_window=swing_window_d, lookback=60)
    d = build_inducement(d, swing_window=swing_window_d, sweep_lookback=5)

    # ── Weekly analysis ───────────────────────────────────────────────────────
    w = detect_bos_choch(weekly_df, swing_window=swing_window_w)

    # ── Bias strength ─────────────────────────────────────────────────────────
    # Daily and Weekly have different calendar indices — reindex weekly onto
    # daily before combining (forward-fill: weekly state persists until next week).
    d_struct = d["structure"].map({"bullish": 1, "bearish": -1, "neutral": 0}).fillna(0)
    w_struct_raw = w["structure"].map({"bullish": 1, "bearish": -1, "neutral": 0}).fillna(0)

    # Strip tz for reindex alignment, then restore
    d_idx_tz = d.index.tz
    d_idx_naive = d.index.tz_localize(None) if d_idx_tz is not None else d.index
    w_idx_naive = w.index.tz_localize(None) if (w.index.tz is not None) else w.index

    w_struct_naive = pd.Series(w_struct_raw.values, index=w_idx_naive)
    w_struct_daily = w_struct_naive.reindex(d_idx_naive, method="ffill").fillna(0)
    w_struct = pd.Series(w_struct_daily.values, index=d.index)

    # Recent CHoCH on Daily
    d_choch_bull = d["choch_bull"].rolling(choch_lookback, min_periods=1).max().astype(bool)
    d_choch_bear = d["choch_bear"].rolling(choch_lookback, min_periods=1).max().astype(bool)
    choch_vote = d_choch_bull.astype(int) - d_choch_bear.astype(int)

    # Combine into bias score [-3, +3] — now all on daily index, no NaN from misalignment
    raw_score = (d_struct + w_struct + choch_vote).fillna(0)

    # Build a clean T1 signal DataFrame on Daily index
    # Threshold: >=1 vote = directional (daily bullish alone is sufficient).
    # >=2 votes → higher-conviction label but too few signals on weekly NaN data.
    t1_daily = pd.DataFrame(index=d.index)
    t1_daily["t1_bias"]        = np.where(raw_score >= 1, "bullish",
                                  np.where(raw_score <= -1, "bearish", "neutral"))
    t1_daily["t1_strength"]    = (raw_score.abs() / 3.0).clip(0, 1)
    t1_daily["t1_choch"]       = d_choch_bull | d_choch_bear
    t1_daily["t1_structure_d"] = d["structure"]
    t1_daily["t1_in_discount"] = d["pd_zone"] == "discount"
    t1_daily["t1_in_premium"]  = d["pd_zone"] == "premium"
    t1_daily["t1_in_ote"]      = d["in_ote"]
    t1_daily["t1_pd_zone"]     = d["pd_zone"]
    t1_daily["t1_equilibrium"] = d["equilibrium"]

    # Nearest BSL/SSL targets from Daily
    closes_d = d["close"].values
    highs_d  = d["high"].values
    lows_d   = d["low"].values

    # Build rolling BSL (recent swing highs above price) and SSL (swing lows below price)
    last_sh = d["last_sh"].values
    last_sl = d["last_sl"].values

    t1_daily["t1_target_long"]  = last_sh   # nearest swing high = BSL target
    t1_daily["t1_target_short"] = last_sl   # nearest swing low  = SSL target

    # Weekly structure merge
    w_sig = pd.DataFrame({"t1_structure_w": w["structure"]}, index=w.index)

    # Strip tz from all indices before merging
    def strip_tz(idx):
        if hasattr(idx, "tz") and idx.tz is not None:
            return idx.tz_localize(None)
        return idx

    ltf_idx   = strip_tz(ltf_df.index)
    daily_idx = strip_tz(t1_daily.index)
    weekly_idx = strip_tz(w_sig.index)

    t1_daily.index = daily_idx
    w_sig.index    = weekly_idx

    # Backward merge onto LTF
    ltf = ltf_df.copy()
    ltf.index = ltf_idx

    merged = pd.merge_asof(
        ltf.reset_index(),
        t1_daily.reset_index(),
        left_on="timestamp",
        right_on="timestamp",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged,
        w_sig.reset_index(),
        left_on="timestamp",
        right_on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp")
    merged.index = ltf_df.index  # restore original index (with tz if any)

    # Fill missing T1 columns with defaults
    for col, default in [
        ("t1_bias", "neutral"), ("t1_strength", 0.0), ("t1_choch", False),
        ("t1_structure_d", "neutral"), ("t1_structure_w", "neutral"),
        ("t1_in_discount", False), ("t1_in_premium", False), ("t1_in_ote", False),
        ("t1_pd_zone", "neutral"),
    ]:
        if col not in merged.columns:
            merged[col] = default

    return merged
