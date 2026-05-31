"""
ICT Signal Dataset Builder — Institutional Edition
====================================================
Collects ALL qualifying ICT signal bars across multiple symbols and labels each
via forward price simulation:

  label = 1  if TP1 (4R) hit before SL within MAX_BARS 5m bars
  label = 0  if SL hit first or timeout

New in institutional upgrade:
  • Volume context  : relative volume, volume percentile
  • Price momentum  : 12-bar (1h), 48-bar (4h), 96-bar (8h) returns
  • Candle quality  : body ratio, ATR percentile rank
  • OB geometry     : zone width in ATR, entry depth in zone
  • Quant scores    : Bollinger mean-reversion, time-series momentum
  • Time features   : day of week, session progress
  • Tier sub-scores : extracted T1/T2/T3 component factors
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MAX_FORWARD_BARS = 576   # 48h on 5m (1h assets: 576 1h bars = 24 days)
# MIN_BARS_BETWEEN is set per-TF in collect_signal_dataset:
#   5m  → 6 bars  = 30 min between signals
#   15m → 4 bars  = 1 hour
#   1h  → 1 bar   = 1 hour minimum (allow more signals from 1h assets)
_MIN_BARS_BY_TF = {"5m": 6, "15m": 4, "30m": 2, "1h": 1, "4h": 1, "1d": 1}


# ── Context features (pre-computed from LTF rolling history) ─────────────────

def _add_context_cols(ltf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach rolling context columns to the LTF DataFrame.
    All computations are backward-looking — zero lookahead.
    """
    df = ltf_df.copy()

    # Volume
    vol_ma = df["volume"].rolling(20, min_periods=1).mean().replace(0, np.nan)
    df["ctx_rvol"]    = (df["volume"] / vol_ma).fillna(1.0)
    df["ctx_vol_pct"] = df["volume"].rolling(100, min_periods=20).rank(pct=True).fillna(0.5)

    # Price momentum (returns over lookback windows)
    df["ctx_mom_12b"]  = df["close"].pct_change(12).fillna(0.0)   # ~1h
    df["ctx_mom_48b"]  = df["close"].pct_change(48).fillna(0.0)   # ~4h
    df["ctx_mom_96b"]  = df["close"].pct_change(96).fillna(0.0)   # ~8h

    # Candle quality
    spread = (df["high"] - df["low"]).clip(lower=1e-9)
    df["ctx_body_ratio"] = (df["close"] - df["open"]).abs() / spread

    # ATR percentile rank (where is current volatility vs history)
    if "t3_atr" in df.columns:
        df["ctx_atr_rank"] = df["t3_atr"].rolling(200, min_periods=50).rank(pct=True).fillna(0.5)
    else:
        df["ctx_atr_rank"] = 0.5

    # Bollinger-band mean reversion score (using 20-bar close)
    roll_mean = df["close"].rolling(20, min_periods=5).mean()
    roll_std  = df["close"].rolling(20, min_periods=5).std().clip(lower=1e-9)
    zscore = (df["close"] - roll_mean) / roll_std
    # Map z-score to [0,1]: 1.0 = near lower band (bullish MR), 0.0 = near upper band
    df["ctx_bb_score"] = (1.0 / (1.0 + np.exp(zscore))).fillna(0.5)

    # Time-series momentum (Moskowitz): sign of 96-bar return normalized
    tsm_raw = df["close"].pct_change(120).fillna(0.0)
    df["ctx_tsm"]  = (1.0 / (1.0 + np.exp(-tsm_raw * 100))).fillna(0.5)

    # Day of week (0=Monday, 6=Sunday)
    df["ctx_day_of_week"] = df.index.dayofweek.astype(float)

    return df


# ── Feature extraction ────────────────────────────────────────────────────────

_BIAS_MAP    = {"bullish": 1, "bearish": -1, "neutral": 0}
_PD_MAP      = {"discount": 1, "premium": -1, "neutral": 0, "equilibrium": 0}
_SESSION_MAP = {"london": 0, "ny_am": 1, "ny_pm": 2, "none": -1, "asia": -2}


def extract_features(bar: pd.Series, direction: str,
                     entry: float, sl: float, tp1: float) -> dict:
    """All features known at signal time — zero lookahead."""
    # Prefer 5m ATR (t4_atr — execution TF) over 15m ATR (t3_atr — confirmation TF)
    t3_atr  = float(bar.get("t4_atr", bar.get("t3_atr", entry * 0.002))) or entry * 0.002
    risk    = abs(entry - sl)
    reward  = abs(tp1 - entry)
    is_long = direction == "long"

    # OB zone geometry
    if is_long:
        ob_top = float(bar.get("t2_ob_bull_top", np.nan))
        ob_bot = float(bar.get("t2_ob_bull_bot", np.nan))
    else:
        ob_top = float(bar.get("t2_ob_bear_top", np.nan))
        ob_bot = float(bar.get("t2_ob_bear_bot", np.nan))

    ob_height_atr = (ob_top - ob_bot) / t3_atr if (np.isfinite(ob_top) and np.isfinite(ob_bot) and ob_top > ob_bot) else 1.0
    ob_range      = ob_top - ob_bot if (np.isfinite(ob_top) and np.isfinite(ob_bot) and ob_top > ob_bot) else 1e-9
    # Entry depth: 0 = at OB bottom (best), 1 = at OB top (worst) for longs
    entry_depth   = (entry - ob_bot) / ob_range if ob_range > 0 and np.isfinite(ob_bot) else 0.5
    if not is_long:
        entry_depth = 1.0 - entry_depth   # flip for shorts

    return {
        # Direction
        "direction":        1 if is_long else 0,

        # Tier 1 — Daily/Weekly bias
        "t1_bias":          _BIAS_MAP.get(str(bar.get("t1_bias", "neutral")), 0),
        "t1_strength":      float(bar.get("t1_strength", 0.0)),
        "t1_pd_zone":       _PD_MAP.get(str(bar.get("t1_pd_zone", "neutral")), 0),

        # Tier 2 — Zone quality (all zone types)
        "t2_in_4h_ob":      int(bool(bar.get("t2_in_ob_bull"      if is_long else "t2_in_ob_bear",      False))),
        "t2_in_1h_ob":      int(bool(bar.get("t2h1_in_ob_bull"    if is_long else "t2h1_in_ob_bear",    False))),
        "t2_in_fvg":        int(bool(bar.get("t2_in_fvg_bull"     if is_long else "t2_in_fvg_bear",     False))),
        "t2_in_brk":        int(bool(bar.get("t2_in_brk_bull"     if is_long else "t2_in_brk_bear",     False)
                                  or bar.get("t2h1_in_brk_bull"   if is_long else "t2h1_in_brk_bear",   False))),
        "t2_in_rej":        int(bool(bar.get("t2h1_in_rej_bull"   if is_long else "t2h1_in_rej_bear",   False))),
        "t2_in_mit":        int(bool(bar.get("t2h1_in_mit_bull"   if is_long else "t2h1_in_mit_bear",   False))),
        "t2_in_vi":         int(bool(bar.get("t2_in_vi_bull"      if is_long else "t2_in_vi_bear",      False)
                                  or bar.get("t2h1_in_vi_bull"    if is_long else "t2h1_in_vi_bear",    False))),
        "zone_quality":     float(max(bar.get("t2_zone_quality_bull"    if is_long else "t2_zone_quality_bear",    0.0),
                                      bar.get("t2h1_zone_quality_bull"  if is_long else "t2h1_zone_quality_bear",  0.0),
                                      bar.get("t2m30_zone_quality_bull" if is_long else "t2m30_zone_quality_bear", 0.0))),
        "t2_confluence":    float(bar.get("t2_confluence", 0.0)),

        # T2.5 — 30m intermediate zone layer (0 when 30m data not in cache)
        "t2m30_in_ob":      int(bool(bar.get("t2m30_in_ob_bull"   if is_long else "t2m30_in_ob_bear",   False))),
        "t2m30_in_fvg":     int(bool(bar.get("t2m30_in_fvg_bull"  if is_long else "t2m30_in_fvg_bear",  False))),
        "t2m30_in_brk":     int(bool(bar.get("t2m30_in_brk_bull"  if is_long else "t2m30_in_brk_bear",  False))),
        "t2m30_in_rej":     int(bool(bar.get("t2m30_in_rej_bull"  if is_long else "t2m30_in_rej_bear",  False))),
        "t2m30_choch":      int(bool(bar.get("t2m30_choch_bull"   if is_long else "t2m30_choch_bear",   False))),
        "t2m30_zone_qual":  float(bar.get("t2m30_zone_quality_bull" if is_long else "t2m30_zone_quality_bear", 0.0)),

        # T4 — 5m execution OB active (1 = tighter 5m OB entry available right now)
        "t4_in_5m_ob":      int(bool(bar.get("t4_signal_long" if is_long else "t4_signal_short", False))),

        # OB geometry — how wide and where in zone
        "ob_height_atr":    ob_height_atr,
        "entry_depth_ob":   float(np.clip(entry_depth, 0, 1)),

        # Tier 3 — Precision
        "session":          _SESSION_MAP.get(str(bar.get("t3_session", "none")), -1),
        "t3_in_ote":        int(bool(bar.get("t3_in_ote", False))),
        "t3_choch":         int(bool(bar.get("t3_choch_bull" if is_long else "t3_choch_bear", False))),
        "swept":            int(bool(bar.get("t3_ssl_swept" if is_long else "t3_bsl_swept", False))),
        "conf_score":       float(bar.get("conf_long" if is_long else "conf_short", 0.0)),

        # Risk geometry
        "sl_atr_ratio":     risk / t3_atr,
        "rr_potential":     reward / max(risk, 1e-9),
        "atr_pct_of_price": t3_atr / max(entry, 1e-9),

        # HTF target distance in R-multiples
        "tp2_distance_r":   _tp2_dist(bar, entry, sl, is_long),

        # Volume context (from ctx_ columns pre-computed on LTF)
        "rvol":             float(bar.get("ctx_rvol", 1.0)),
        "vol_pct":          float(bar.get("ctx_vol_pct", 0.5)),

        # Price momentum
        "mom_12b":          float(bar.get("ctx_mom_12b", 0.0)),
        "mom_48b":          float(bar.get("ctx_mom_48b", 0.0)),
        "mom_96b":          float(bar.get("ctx_mom_96b", 0.0)),

        # Candle & volatility quality
        "body_ratio":       float(bar.get("ctx_body_ratio", 0.5)),
        "atr_rank":         float(bar.get("ctx_atr_rank", 0.5)),

        # Quant layer scores
        "bb_score":         float(bar.get("ctx_bb_score", 0.5)),
        "tsm_score":        float(bar.get("ctx_tsm", 0.5)),

        # Time
        "day_of_week":      float(bar.get("ctx_day_of_week", 0.0)),
    }


def _tp2_dist(bar: pd.Series, entry: float, sl: float, is_long: bool) -> float:
    key = "t1_target_long" if is_long else "t1_target_short"
    try:
        val = float(bar.get(key, np.nan))
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(val):
        return 0.0
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    dist = (val - entry) / risk if is_long else (entry - val) / risk
    return max(dist, 0.0)


# ── Forward simulation ────────────────────────────────────────────────────────

def simulate_outcome(ltf_df: pd.DataFrame, bar_pos: int,
                     entry: float, sl: float, tp1: float,
                     direction: str, max_bars: int = MAX_FORWARD_BARS) -> int:
    future = ltf_df.iloc[bar_pos + 1 : bar_pos + 1 + max_bars]
    for _, frow in future.iterrows():
        hi, lo = float(frow["high"]), float(frow["low"])
        if direction == "long":
            if lo <= sl:  return 0
            if hi >= tp1: return 1
        else:
            if hi >= sl:  return 0
            if lo <= tp1: return 1
    return 0   # timeout = loss


# ── Main dataset collector ────────────────────────────────────────────────────

def collect_signal_dataset(
    strategy,
    all_data: dict,
    cfg,
    ltf_name: str = "5m",
    risk_pct: float = 0.01,
    capital: float = 10_000.0,
    training_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Scans every qualifying ICT signal bar for each symbol (not limited by
    single-position backtest constraint). Returns labeled feature DataFrame.

    training_symbols: symbols to collect from (can differ from trading symbols).
                      Defaults to strategy's configured symbols.
    """
    import yaml

    with open("config.yaml") as f:
        raw = yaml.safe_load(f)

    def _cfg_get(obj, key, default):
        if obj is None:
            return default
        if hasattr(obj, key):
            return getattr(obj, key)
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return default

    pcfg      = getattr(cfg, "plan_c_ict", None)
    threshold = float(_cfg_get(pcfg, "confidence_threshold", 0.45))
    zone_gate = str(_cfg_get(pcfg, "zone_gate", "ob_only"))
    # Min bars between signals — TF-aware: 5m=6bars(30min), 1h=1bar(1h)
    MIN_BARS_BETWEEN = _MIN_BARS_BY_TF.get(ltf_name, 6)

    if training_symbols is not None:
        symbols = training_symbols
    else:
        pcfg_syms = _cfg_get(pcfg, "symbols", None)
        symbols   = list(pcfg_syms) if pcfg_syms else (cfg.symbols if hasattr(cfg, "symbols") else ["BTCUSDT"])

    strategy.prepare(all_data)

    # Force-build signals for any training symbol not populated by prepare()
    # (prepare() only processes plan_c_ict.symbols, which may be BTC-only)
    for sym in symbols:
        if sym not in strategy._signals:
            try:
                strategy._signals[sym] = strategy._build_symbol(sym, all_data)
                print(f"  [ML] Built signals for {sym} (training-only)")
            except Exception as e:
                print(f"  [ML] {sym} build failed — {e}")

    records = []

    for sym in symbols:
        ltf_df = all_data.get((sym, ltf_name))
        if ltf_df is None or len(ltf_df) < 100:
            print(f"  [ML] No LTF data for {sym} {ltf_name} — skipping")
            continue

        sig_df = strategy._signals.get(sym)
        if sig_df is None:
            print(f"  [ML] No signal DataFrame for {sym} — skipping")
            continue

        # Attach rolling context features to sig_df (aligned on LTF index)
        sig_with_ctx = _merge_context(sig_df, ltf_df)

        bar_positions = {ts: i for i, ts in enumerate(ltf_df.index)}
        last_long_bar  = -9999
        last_short_bar = -9999

        n_sig = 0
        for ts, bar in sig_with_ctx.iterrows():
            if ts not in bar_positions:
                continue
            bar_pos = bar_positions[ts]

            if str(bar.get("t3_session", "none")) == "asia":
                continue
            if not bool(bar.get("t3_in_killzone", False)):
                continue

            close = float(bar.get("close", 0.0))
            if close <= 0:
                continue

            # Use 5m ATR (t4_atr) for tight, accurate feature computation
            t4_atr = float(bar.get("t4_atr", bar.get("t3_atr", close * 0.002))) or close * 0.002

            # ── Long signal ───────────────────────────────────────────────────
            conf_long     = float(bar.get("conf_long", 0.0))
            ssl_swept     = bool(bar.get("t3_ssl_swept", False))

            in_4h_ob_long  = bool(bar.get("t2_in_ob_bull",       False))
            in_1h_ob_long  = bool(bar.get("t2h1_in_ob_bull",     False))
            in_4h_fvg_long = bool(bar.get("t2_in_fvg_bull",      False))
            in_4h_brk_long = bool(bar.get("t2_in_brk_bull",      False))
            in_1h_brk_long = bool(bar.get("t2h1_in_brk_bull",    False))
            in_1h_rej_long = bool(bar.get("t2h1_in_rej_bull",    False))
            in_1h_mit_long = bool(bar.get("t2h1_in_mit_bull",    False))
            in_4h_vi_long  = bool(bar.get("t2_in_vi_bull",       False))
            in_1h_vi_long  = bool(bar.get("t2h1_in_vi_bull",     False))

            # zone_gate: "ob_only" (crypto — strict, highest quality)
            #            "all_zones" (forex/commodity/india — more signals)
            if zone_gate == "ob_only":
                in_any_long = (in_4h_ob_long or in_1h_ob_long)
            else:
                in_any_long = (in_4h_ob_long or in_1h_ob_long or in_4h_brk_long or
                               in_1h_brk_long or in_1h_rej_long or in_1h_mit_long or
                               in_4h_vi_long  or in_1h_vi_long  or in_4h_fvg_long)

            if (conf_long >= threshold and in_any_long
                    and ssl_swept and bar_pos - last_long_bar >= MIN_BARS_BETWEEN):

                entry   = close
                t4_long = bool(bar.get("t4_signal_long", False))

                if t4_long:
                    sl  = float(bar.get("t4_sl_long",  entry - 2.0 * t4_atr))
                    tp1 = float(bar.get("t4_tp1_long", entry + 4.0 * abs(entry - sl)))
                elif in_4h_ob_long:
                    t2_sl = bar.get("t2_ob_sl_long", np.nan)
                    sl = float(t2_sl) if (_ok_sl(t2_sl) and float(t2_sl) < entry) else entry - 2.0 * t4_atr
                    r = entry - sl; tp1 = entry + 4.0 * r
                elif in_1h_ob_long:
                    t2h1_sl = bar.get("t2h1_ob_sl_long", np.nan)
                    sl = float(t2h1_sl) if (_ok_sl(t2h1_sl) and float(t2h1_sl) < entry) else entry - 2.0 * t4_atr
                    r = entry - sl; tp1 = entry + 4.0 * r
                elif in_4h_brk_long:
                    sl_raw = bar.get("t2_brk_sl_long", np.nan)
                    sl = float(sl_raw) if (_ok_sl(sl_raw) and float(sl_raw) < entry) else entry - 2.0 * t4_atr
                    r = entry - sl; tp1 = entry + 4.0 * r
                else:
                    sl  = entry - 2.0 * t4_atr
                    tp1 = entry + 4.0 * 2.0 * t4_atr

                if sl >= entry or tp1 <= entry:
                    continue

                label = simulate_outcome(ltf_df, bar_pos, entry, sl, tp1, "long")
                feats = extract_features(bar, "long", entry, sl, tp1)
                records.append({**feats, "label": label, "sym": sym, "ts": ts,
                                 "entry": entry, "sl": sl, "tp1": tp1})
                last_long_bar = bar_pos
                n_sig += 1

            # ── Short signal ──────────────────────────────────────────────────
            conf_short     = float(bar.get("conf_short", 0.0))
            bsl_swept      = bool(bar.get("t3_bsl_swept", False))

            in_4h_ob_short  = bool(bar.get("t2_in_ob_bear",       False))
            in_1h_ob_short  = bool(bar.get("t2h1_in_ob_bear",     False))
            in_4h_fvg_short = bool(bar.get("t2_in_fvg_bear",      False))
            in_4h_brk_short = bool(bar.get("t2_in_brk_bear",      False))
            in_1h_brk_short = bool(bar.get("t2h1_in_brk_bear",    False))
            in_1h_rej_short = bool(bar.get("t2h1_in_rej_bear",    False))
            in_1h_mit_short = bool(bar.get("t2h1_in_mit_bear",    False))
            in_4h_vi_short  = bool(bar.get("t2_in_vi_bear",       False))
            in_1h_vi_short  = bool(bar.get("t2h1_in_vi_bear",     False))
            if zone_gate == "ob_only":
                in_any_short = (in_4h_ob_short or in_1h_ob_short)
            else:
                in_any_short = (in_4h_ob_short or in_1h_ob_short or in_4h_brk_short or
                                in_1h_brk_short or in_1h_rej_short or in_1h_mit_short or
                                in_4h_vi_short  or in_1h_vi_short  or in_4h_fvg_short)

            if (conf_short >= threshold and in_any_short
                    and bsl_swept and bar_pos - last_short_bar >= MIN_BARS_BETWEEN):

                entry    = close
                t4_short = bool(bar.get("t4_signal_short", False))

                if t4_short:
                    sl  = float(bar.get("t4_sl_short",  entry + 2.0 * t4_atr))
                    tp1 = float(bar.get("t4_tp1_short", entry - 4.0 * abs(sl - entry)))
                elif in_4h_ob_short:
                    t2_sl = bar.get("t2_ob_sl_short", np.nan)
                    sl = float(t2_sl) if (_ok_sl(t2_sl) and float(t2_sl) > entry) else entry + 2.0 * t4_atr
                    r = sl - entry; tp1 = entry - 4.0 * r
                elif in_1h_ob_short:
                    t2h1_sl = bar.get("t2h1_ob_sl_short", np.nan)
                    sl = float(t2h1_sl) if (_ok_sl(t2h1_sl) and float(t2h1_sl) > entry) else entry + 2.0 * t4_atr
                    r = sl - entry; tp1 = entry - 4.0 * r
                elif in_4h_brk_short:
                    sl_raw = bar.get("t2_brk_sl_short", np.nan)
                    sl = float(sl_raw) if (_ok_sl(sl_raw) and float(sl_raw) > entry) else entry + 2.0 * t4_atr
                    r = sl - entry; tp1 = entry - 4.0 * r
                else:
                    sl  = entry + 2.0 * t4_atr
                    tp1 = entry - 4.0 * 2.0 * t4_atr

                if sl <= entry or tp1 >= entry:
                    continue

                label = simulate_outcome(ltf_df, bar_pos, entry, sl, tp1, "short")
                feats = extract_features(bar, "short", entry, sl, tp1)
                records.append({**feats, "label": label, "sym": sym, "ts": ts,
                                 "entry": entry, "sl": sl, "tp1": tp1})
                last_short_bar = bar_pos
                n_sig += 1

        print(f"  [ML] {sym}: {n_sig} signals collected")

    df = pd.DataFrame(records)
    if len(df):
        df = df.sort_values("ts").reset_index(drop=True)
    return df


def _merge_context(sig_df: pd.DataFrame, ltf_df: pd.DataFrame) -> pd.DataFrame:
    """Attach rolling context columns computed from raw LTF data into sig_df."""
    ltf_ctx = _add_context_cols(ltf_df)
    ctx_cols = [c for c in ltf_ctx.columns if c.startswith("ctx_")]

    def _strip(idx):
        return idx.tz_localize(None) if (hasattr(idx, "tz") and idx.tz is not None) else idx

    sig_idx = _strip(sig_df.index)
    ltf_idx = _strip(ltf_ctx.index)

    ctx_sub = ltf_ctx[ctx_cols].copy()
    ctx_sub.index = pd.DatetimeIndex(ltf_idx)
    ctx_sub.index.name = "_ts_ctx"

    sig_reset = sig_df.copy()
    sig_reset.index = pd.DatetimeIndex(sig_idx)
    sig_reset.index.name = "_ts"
    sig_reset = sig_reset.reset_index()  # now has column "_ts"

    merged = pd.merge_asof(
        sig_reset,
        ctx_sub.reset_index(),   # column "_ts_ctx"
        left_on="_ts",
        right_on="_ts_ctx",
        direction="backward",
    )

    for col in ctx_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    merged.index = sig_df.index
    return merged


def _ok_sl(x) -> bool:
    try:
        return not np.isnan(float(x))
    except (TypeError, ValueError):
        return False


FEATURE_COLS = [
    # Direction
    "direction",
    # T1
    "t1_bias", "t1_strength", "t1_pd_zone",
    # T2 — 4H + 1H zone types
    "t2_in_4h_ob", "t2_in_1h_ob", "t2_in_fvg",
    "t2_in_brk", "t2_in_rej", "t2_in_mit", "t2_in_vi",
    "zone_quality", "t2_confluence",
    # T2.5 — 30m intermediate zones (0 when not in cache — backward-compatible)
    "t2m30_in_ob", "t2m30_in_fvg", "t2m30_in_brk", "t2m30_in_rej",
    "t2m30_choch", "t2m30_zone_qual",
    # T4 — 5m execution OB active (0 = not yet, 1 = precise 5m entry available)
    "t4_in_5m_ob",
    # OB geometry
    "ob_height_atr", "entry_depth_ob",
    # T3
    "session", "t3_in_ote", "t3_choch", "swept", "conf_score",
    # Risk
    "sl_atr_ratio", "rr_potential", "atr_pct_of_price", "tp2_distance_r",
    # Volume
    "rvol", "vol_pct",
    # Momentum
    "mom_12b", "mom_48b", "mom_96b",
    # Candle / volatility quality
    "body_ratio", "atr_rank",
    # Quant
    "bb_score", "tsm_score",
    # Time
    "day_of_week",
]
