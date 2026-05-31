from __future__ import annotations
import pandas as pd

from signals.liquidity import liquidity_sweep
from signals.trap import trap_detection
from signals.order_flow import order_flow_proxy
from signals.market_structure import market_structure
from signals.volatility_regime import volatility_regime


def build_htf_signals(htf_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach market structure and volatility regime to HTF df."""
    df = htf_df.copy()
    df = market_structure(df)
    df = volatility_regime(df)
    return df


def align_htf_to_ltf(ltf_df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge HTF signals onto LTF df using backward merge (most recent past HTF candle).
    Fully lookahead-safe: only past HTF candles are used.
    """
    htf_signals = htf_df[["structure", "vol_regime"]].copy()
    htf_signals = htf_signals.rename(
        columns={"structure": "htf_structure", "vol_regime": "htf_vol_regime"}
    )
    htf_signals = htf_signals.reset_index()

    ltf_reset = ltf_df.reset_index()

    merged = pd.merge_asof(
        ltf_reset.sort_values("timestamp"),
        htf_signals.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp")
    return merged


def build_ltf_signals(ltf_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply all LTF signal functions."""
    r = cfg.rules
    df = ltf_df.copy()
    df.attrs["n_period"] = r.n_period
    df = liquidity_sweep(df, vol_pct_thresh=r.vol_pct_thresh)
    df = trap_detection(df, body_ratio_thresh=r.body_ratio_thresh)
    df = order_flow_proxy(df, vol_pct_thresh=r.vol_pct_thresh)
    df = market_structure(df)
    df = volatility_regime(df)
    return df


def generate_signals(ltf_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Combine entry conditions into boolean signal columns.

    SMC logic: a liquidity sweep sets up the trade; the reversal candle (trap)
    is the entry trigger. Sweep must have occurred within the last `sweep_lookback`
    bars; the trap (or order-flow absorption) confirms on the current bar.

    Long entry (all required):
      1. HTF bias = bullish
      2. Liquidity sweep of lows within last N bars  (sweep_bull.rolling)
      3. Trap confirmation OR strong order-flow absorption on current bar
      4. Vol regime = normal or high

    Short entry: inverse.
    """
    sweep_lookback = max(cfg.rules.n_period // 5, 3)  # e.g. 6 bars
    df = ltf_df.copy()
    df["htf_structure"] = df.get("htf_structure", pd.Series("neutral", index=df.index))

    # Recent sweep: was there a sweep within the last sweep_lookback bars?
    recent_sweep_bull = df["sweep_bull"].rolling(sweep_lookback, min_periods=1).max().astype(bool)
    recent_sweep_bear = df["sweep_bear"].rolling(sweep_lookback, min_periods=1).max().astype(bool)

    # Entry trigger: trap OR strong order-flow absorption (either is sufficient)
    entry_trigger_bull = df["trap_bull"] | df["of_bull"]
    entry_trigger_bear = df["trap_bear"] | df["of_bear"]

    long_cond = (
        (df["htf_structure"] == "bullish")
        & recent_sweep_bull
        & entry_trigger_bull
        & df["vol_regime"].isin(["normal", "high"])
    )

    short_cond = (
        (df["htf_structure"] == "bearish")
        & recent_sweep_bear
        & entry_trigger_bear
        & df["vol_regime"].isin(["normal", "high"])
    )

    df["long_signal"] = long_cond
    df["short_signal"] = short_cond
    return df
