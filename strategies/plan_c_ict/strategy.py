"""
Plan C — ICT Smart Money Concepts Strategy
==========================================
Full 4-tier ICT cascade:
  Tier 1 (bias)        : Weekly + Daily BOS/CHoCH → macro direction
  Tier 2 (zones)       : 4H + 1H OBs + FVGs in discount/premium zone
                         Optional T2.5: 30m intermediate zones
  Tier 3 (confirmation): 15m — CHoCH, liquidity sweep, OTE, killzone
  Tier 4 (execution)   : 5m — enter at 5m OB inside T3-confirmed zone
                         Tighter SL (5m OB wick vs 15m OB wick) = better R:R

Required timeframes: 1w, 1d, 4h, 1h, 15m, 5m
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, TradeSetup
from strategies.plan_c_ict.tier1_structure  import build_tier1
from strategies.plan_c_ict.tier2_ict_zones  import build_tier2
from strategies.plan_c_ict.tier3_precision  import build_tier3
from strategies.plan_c_ict.tier4_execution  import build_tier4
from strategies.plan_c_ict.confluence       import (
    compute_ict_confluence_series,
    CONFIDENCE_THRESHOLD,
)
from indicators.atr import atr as compute_atr

_ML_MODEL_PATH = "models/ict_predictor.pkl"


def _default_cfg():
    return {
        "swing_window_d":   3,
        "swing_window_w":   2,
        "swing_window_h4":  3,
        "swing_window_h1":  3,
        "swing_window_m30": 4,   # 30m T2.5 intermediate zones
        "swing_window_m15": 3,
        "choch_lookback":   10,
        "sweep_lookback":   5,
        "atr_period":       14,
        "impulse_mult":     1.5,
        "min_fvg_atr":      0.25,
        "tp1_rr":           4.0,   # TP1 risk:reward — minimum 1:4
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


class PlanCICTStrategy(BaseStrategy):

    name = "plan_c_ict"

    @property
    def required_timeframes(self) -> dict[str, list[str]]:
        pcfg_syms = _cfg_get(self._pcfg, "symbols", None)
        if pcfg_syms:
            syms = list(pcfg_syms)
        else:
            syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else ["BTCUSDT", "ETHUSDT"]
        ltf = _cfg_get(self._pcfg, "ltf", "5m")
        if ltf == "1h":
            # 1h-LTF segments (India/commodity/forex): only need 1w, 1d, 4h, 1h
            return {s: ["1w", "1d", "4h", "1h"] for s in syms}
        else:
            # 5m-LTF (crypto): need 15m for T3 confirmation + 5m for T4 execution
            return {s: ["1w", "1d", "4h", "1h", "15m", "5m"] for s in syms}

    def __init__(self, cfg):
        self._cfg   = cfg
        self._pcfg  = getattr(cfg, "plan_c_ict", None)
        self._signals: dict[str, pd.DataFrame] = {}
        self._ml_model    = None
        self._ml_threshold = float(_cfg_get(self._pcfg, "ml_threshold", 0.0))
        self._try_load_ml()

    # ── ML model loader ───────────────────────────────────────────────────────
    def _try_load_ml(self) -> None:
        import os
        if not os.path.exists(_ML_MODEL_PATH):
            return
        try:
            from ml.supervised import ICTTradePredictor
            predictor = ICTTradePredictor()
            if predictor.load(_ML_MODEL_PATH):
                self._ml_model = predictor
                thr = self._ml_threshold
                print(f"  [PlanC] ML model loaded (threshold={thr:.2f})")
        except Exception as e:
            print(f"  [PlanC] ML model load failed — {e}")

    # ── prepare ───────────────────────────────────────────────────────────────
    def prepare(self, all_data: dict[tuple[str, str], pd.DataFrame]) -> None:
        # plan_c_ict can override the global symbol list — BTC OB setups are
        # considerably more reliable than ETH (different liquidity structure)
        pcfg_syms = _cfg_get(self._pcfg, "symbols", None)
        if pcfg_syms:
            syms = list(pcfg_syms)
        elif hasattr(self._cfg, "symbols"):
            syms = self._cfg.symbols
        else:
            syms = list({s for s, _ in all_data})
        for sym in syms:
            try:
                self._signals[sym] = self._build_symbol(sym, all_data)
            except Exception as e:
                print(f"  [PlanC] WARNING: {sym} prepare failed — {e}")
                import traceback
                traceback.print_exc()

    def _build_symbol(
        self,
        sym: str,
        all_data: dict[tuple[str, str], pd.DataFrame],
    ) -> pd.DataFrame:
        def get(tf):
            df = all_data.get((sym, tf))
            if df is None or len(df) < 50:
                raise ValueError(f"Missing or too-short data: {sym} {tf}")
            return df.copy()

        weekly = get("1w")
        daily  = get("1d")
        h4     = get("4h")
        h1     = get("1h")

        ltf_name = _cfg_get(self._pcfg, "ltf", "5m")
        market   = _cfg_get(self._pcfg, "market", "global")

        # For 1h LTF (Indian/commodity/forex): T3 runs on 1h, no 5m T4 layer
        # For 5m LTF (crypto): T3 runs on 15m, T4 runs on 5m
        is_1h_ltf = (ltf_name == "1h")

        if is_1h_ltf:
            m15_df = h1       # T3 confirmation on 1h
            m5_df  = h1       # T4 execution also on 1h (no 5m available long-term)
        else:
            m15_df = get("15m")
            m5_df  = get("5m")

        # 30m is optional T2.5 — skip if using 1h LTF (30m < 1h doesn't make sense)
        m30_df = None if is_1h_ltf else all_data.get((sym, "30m"))
        if m30_df is not None and len(m30_df) < 50:
            m30_df = None

        c = _default_cfg()
        if self._pcfg:
            for k in c:
                c[k] = _cfg_get(self._pcfg, k, c[k])

        # ── Tier 1: Weekly + Daily bias → merged onto confirmation TF ───────────
        df_15m = build_tier1(
            daily_df=daily,
            weekly_df=weekly,
            ltf_df=m15_df,
            swing_window_d=c["swing_window_d"],
            swing_window_w=c["swing_window_w"],
            choch_lookback=c["choch_lookback"],
        )

        # ── Tier 2: zone sources depend on LTF ───────────────────────────────
        # 5m LTF (crypto):  T2 = 4H + 1H zones
        # 1h LTF (others):  T2 = Daily + 4H zones (shifted up one level)
        if is_1h_ltf:
            t2_h4_src = daily  # Daily acts as "slower" T2 source
            t2_h1_src = h4     # 4H acts as "faster" T2 source
        else:
            t2_h4_src = h4
            t2_h1_src = h1

        df_15m = build_tier2(
            h4_df=t2_h4_src,
            h1_df=t2_h1_src,
            ltf_df=df_15m,
            swing_window_h4=c["swing_window_h4"],
            swing_window_h1=c["swing_window_h1"],
            atr_period=c["atr_period"],
            impulse_mult=c["impulse_mult"],
            min_fvg_atr=c["min_fvg_atr"],
            m30_df=m30_df,
            swing_window_m30=c.get("swing_window_m30", 4),
        )

        # ── Tier 3: CHoCH + sweep + OTE + killzone (market-aware) ─────────────
        t3_15m = build_tier3(
            m15_df=df_15m,
            atr_period=c["atr_period"],
            swing_window=c["swing_window_m15"],
            choch_lookback=c["choch_lookback"],
            sweep_lookback=c["sweep_lookback"],
            tp1_rr=c["tp1_rr"],
            market=market,
        )
        t3_15m["t3_tp2_long"]  = t3_15m.get("t2_htf_bsl", pd.Series(np.nan, index=t3_15m.index))
        t3_15m["t3_tp2_short"] = t3_15m.get("t2_htf_ssl", pd.Series(np.nan, index=t3_15m.index))

        # Confluence computed on 15m (includes T1/T2/T3 weights)
        t3_15m["conf_long"]  = compute_ict_confluence_series(t3_15m, "long")
        t3_15m["conf_short"] = compute_ict_confluence_series(t3_15m, "short")

        # ── Tier 1+2 on 5m (execution context) ───────────────────────────────
        df_5m = build_tier1(
            daily_df=daily,
            weekly_df=weekly,
            ltf_df=m5_df,
            swing_window_d=c["swing_window_d"],
            swing_window_w=c["swing_window_w"],
            choch_lookback=c["choch_lookback"],
        )
        df_5m = build_tier2(
            h4_df=h4,
            h1_df=h1,
            ltf_df=df_5m,
            swing_window_h4=c["swing_window_h4"],
            swing_window_h1=c["swing_window_h1"],
            atr_period=c["atr_period"],
            impulse_mult=c["impulse_mult"],
            min_fvg_atr=c["min_fvg_atr"],
            # no 30m T2.5 on 5m — 30m > 5m makes it a zone source, not T2.5
        )

        # ── Merge T3 confirmation from 15m → 5m (backward fill) ──────────────
        T3_CONF_COLS = [
            "t3_signal_long", "t3_signal_short",
            "t3_choch_bull",  "t3_choch_bear",
            "t3_in_killzone", "t3_session",     "t3_session_wt",
            "t3_ssl_swept",   "t3_bsl_swept",
            "t3_in_ote",      "t3_confidence",
            "t3_atr",         # 15m ATR — used as feature baseline by feature_builder
            "conf_long",      "conf_short",
            "t3_tp2_long",    "t3_tp2_short",
            "t1_bias",        "t1_strength",    "t1_pd_zone",
            "t1_target_long", "t1_target_short",
        ]
        t3_cols_avail = [c_ for c_ in T3_CONF_COLS if c_ in t3_15m.columns]

        def _strip_tz(idx):
            if hasattr(idx, "tz") and idx.tz is not None:
                return idx.tz_localize(None)
            return idx

        t3_for_merge = t3_15m[t3_cols_avail].copy()
        t3_for_merge.index = _strip_tz(t3_15m.index)

        df_5m_stripped = df_5m.copy()
        df_5m_stripped.index = _strip_tz(df_5m.index)

        merged_5m = pd.merge_asof(
            df_5m_stripped.reset_index(),
            t3_for_merge.reset_index(),
            on="timestamp",
            direction="backward",
        ).set_index("timestamp")
        merged_5m.index = df_5m.index   # restore tz-aware index

        # Default-fill any missing T3 columns
        for col in t3_cols_avail:
            if col not in merged_5m.columns:
                merged_5m[col] = False if "signal" in col or "swept" in col or "choch" in col else 0.0

        # ── Tier 4: 5m execution — enter at 5m OB inside T3-confirmed zone ───
        t4 = build_tier4(
            m5_df=merged_5m,
            atr_period=c["atr_period"],
            tp1_rr=c["tp1_rr"],
            impulse_mult=c["impulse_mult"],
        )

        # ── ML context features ────────────────────────────────────────────────
        try:
            from ml.feature_builder import _add_context_cols
            t4 = _add_context_cols(t4)
        except Exception:
            pass

        return t4

    # ── on_bar ────────────────────────────────────────────────────────────────
    def on_bar(
        self,
        symbol: str,
        ts: pd.Timestamp,
        row: pd.Series,
        capital: float = 10_000.0,
    ) -> list[TradeSetup]:

        df = self._signals.get(symbol)
        if df is None or ts not in df.index:
            return []

        bar = df.loc[ts]
        setups: list[TradeSetup] = []

        cfg  = _default_cfg()
        if self._pcfg:
            for k in cfg:
                cfg[k] = _cfg_get(self._pcfg, k, cfg[k])

        threshold = cfg["confidence_threshold"]

        # Asia session excluded — ICT Power of 3 manipulation phase unreliable
        session = str(bar.get("t3_session", "none"))
        if session == "asia":
            return []

        close = float(bar.get("close", 0.0))
        t4_atr = float(bar.get("t4_atr", bar.get("t3_atr", close * 0.002)))

        # ── Long: T2 zone + T3 confirmation + T4 SL refinement ───────────────
        # Gate: T2 institutional zone + T3 15m CHoCH/sweep/killzone + confidence
        # SL:   T4 5m OB wick when available (tighter) — else T2 zone SL
        # This matches the training signal collection exactly.
        conf_long     = float(bar.get("conf_long", 0.0))
        in_kz         = bool(bar.get("t3_in_killzone", False))
        ssl_swept     = bool(bar.get("t3_ssl_swept", False))

        in_4h_ob_long  = bool(bar.get("t2_in_ob_bull",       False))
        in_1h_ob_long  = bool(bar.get("t2h1_in_ob_bull",     False))
        in_4h_brk_long = bool(bar.get("t2_in_brk_bull",      False))
        in_1h_brk_long = bool(bar.get("t2h1_in_brk_bull",    False))
        in_1h_rej_long = bool(bar.get("t2h1_in_rej_bull",    False))
        in_1h_mit_long = bool(bar.get("t2h1_in_mit_bull",    False))
        in_4h_vi_long  = bool(bar.get("t2_in_vi_bull",       False))
        in_1h_vi_long  = bool(bar.get("t2h1_in_vi_bull",     False))
        in_any_long    = (in_4h_ob_long or in_1h_ob_long or in_4h_brk_long or
                          in_1h_brk_long or in_1h_rej_long or in_1h_mit_long or
                          in_4h_vi_long or in_1h_vi_long)

        if conf_long >= threshold and in_kz and in_any_long and ssl_swept:
            # Use T4 SL (5m OB wick) when available — tighter stop, better RR
            t4_long = bool(bar.get("t4_signal_long", False))
            if t4_long:
                sl  = float(bar.get("t4_sl_long",  close - 2.0 * t4_atr))
                tp1 = float(bar.get("t4_tp1_long", close + cfg["tp1_rr"] * abs(close - sl)))
            else:
                sl  = _best_sl_long(bar, close, t4_atr,
                                    in_4h_ob_long, in_1h_ob_long,
                                    in_4h_brk_long, in_1h_brk_long, in_1h_rej_long)
                r_long = close - sl
                tp1 = close + cfg["tp1_rr"] * max(r_long, 0.0001)

            tp2_raw = bar.get("t3_tp2_long", bar.get("t1_target_long", np.nan))
            tp2 = float(tp2_raw) if not _isnan(tp2_raw) and float(tp2_raw) > tp1 else None

            zone = ("4h_ob" if in_4h_ob_long else "1h_ob" if in_1h_ob_long else
                    "4h_brk" if in_4h_brk_long else "1h_brk" if in_1h_brk_long else
                    "1h_rej" if in_1h_rej_long else "1h_mit" if in_1h_mit_long else
                    "4h_vi" if in_4h_vi_long else "1h_vi")
            zq = max(float(bar.get("t2_zone_quality_bull",   0.0)),
                     float(bar.get("t2h1_zone_quality_bull", 0.0)))

            if sl < close < tp1:
                setups.append(TradeSetup(
                    symbol=symbol, direction="long",
                    entry=close, sl=sl, tp1=tp1, tp2=tp2,
                    confidence=conf_long, atr=t4_atr,
                    meta={
                        "t1_bias":      bar.get("t1_bias",      "neutral"),
                        "t1_strength":  bar.get("t1_strength",  0.0),
                        "t1_pd_zone":   bar.get("t1_pd_zone",   "neutral"),
                        "t2_in_ob":     bar.get("t2_in_ob_bull", False),
                        "t2_in_fvg":    bar.get("t2_in_fvg_bull", False),
                        "t3_session":   bar.get("t3_session",   "none"),
                        "t3_choch":     bar.get("t3_choch_bull", False),
                        "t3_swept":     bar.get("t3_ssl_swept",  False),
                        "t3_in_ote":    bar.get("t3_in_ote",    False),
                        "t4_in_5m_ob":  t4_long,
                        "zone_type":    zone,
                        "zone_quality": round(zq, 3),
                    },
                ))

        # ── Short: same pattern ───────────────────────────────────────────────
        conf_short     = float(bar.get("conf_short", 0.0))
        bsl_swept      = bool(bar.get("t3_bsl_swept", False))

        in_4h_ob_short  = bool(bar.get("t2_in_ob_bear",       False))
        in_1h_ob_short  = bool(bar.get("t2h1_in_ob_bear",     False))
        in_4h_brk_short = bool(bar.get("t2_in_brk_bear",      False))
        in_1h_brk_short = bool(bar.get("t2h1_in_brk_bear",    False))
        in_1h_rej_short = bool(bar.get("t2h1_in_rej_bear",    False))
        in_1h_mit_short = bool(bar.get("t2h1_in_mit_bear",    False))
        in_4h_vi_short  = bool(bar.get("t2_in_vi_bear",       False))
        in_1h_vi_short  = bool(bar.get("t2h1_in_vi_bear",     False))
        in_any_short    = (in_4h_ob_short or in_1h_ob_short or in_4h_brk_short or
                           in_1h_brk_short or in_1h_rej_short or in_1h_mit_short or
                           in_4h_vi_short or in_1h_vi_short)

        if conf_short >= threshold and in_kz and in_any_short and bsl_swept:
            t4_short = bool(bar.get("t4_signal_short", False))
            if t4_short:
                sl  = float(bar.get("t4_sl_short",  close + 2.0 * t4_atr))
                tp1 = float(bar.get("t4_tp1_short", close - cfg["tp1_rr"] * abs(sl - close)))
            else:
                sl  = _best_sl_short(bar, close, t4_atr,
                                     in_4h_ob_short, in_1h_ob_short,
                                     in_4h_brk_short, in_1h_brk_short, in_1h_rej_short)
                r_short = sl - close
                tp1 = close - cfg["tp1_rr"] * max(r_short, 0.0001)

            tp2_raw = bar.get("t3_tp2_short", bar.get("t1_target_short", np.nan))
            tp2 = float(tp2_raw) if not _isnan(tp2_raw) and float(tp2_raw) < tp1 else None

            zone = ("4h_ob" if in_4h_ob_short else "1h_ob" if in_1h_ob_short else
                    "4h_brk" if in_4h_brk_short else "1h_brk" if in_1h_brk_short else
                    "1h_rej" if in_1h_rej_short else "1h_mit" if in_1h_mit_short else
                    "4h_vi" if in_4h_vi_short else "1h_vi")
            zq = max(float(bar.get("t2_zone_quality_bear",   0.0)),
                     float(bar.get("t2h1_zone_quality_bear", 0.0)))

            if sl > close > tp1:
                setups.append(TradeSetup(
                    symbol=symbol, direction="short",
                    entry=close, sl=sl, tp1=tp1, tp2=tp2,
                    confidence=conf_short, atr=t4_atr,
                    meta={
                        "t1_bias":      bar.get("t1_bias",      "neutral"),
                        "t1_strength":  bar.get("t1_strength",  0.0),
                        "t1_pd_zone":   bar.get("t1_pd_zone",   "neutral"),
                        "t2_in_ob":     bar.get("t2_in_ob_bear", False),
                        "t2_in_fvg":    bar.get("t2_in_fvg_bear", False),
                        "t3_session":   bar.get("t3_session",   "none"),
                        "t3_choch":     bar.get("t3_choch_bear", False),
                        "t3_swept":     bar.get("t3_bsl_swept",  False),
                        "t3_in_ote":    bar.get("t3_in_ote",    False),
                        "t4_in_5m_ob":  t4_short,
                        "zone_type":    zone,
                        "zone_quality": round(zq, 3),
                    },
                ))

        return self._apply_ml_filter(setups, bar)

    # ── ML filter ─────────────────────────────────────────────────────────────
    def _apply_ml_filter(
        self, setups: list[TradeSetup], bar: pd.Series
    ) -> list[TradeSetup]:
        if not setups or self._ml_model is None or self._ml_threshold <= 0.0:
            return setups
        from ml.feature_builder import extract_features
        filtered = []
        for s in setups:
            feats  = extract_features(bar, s.direction, s.entry, s.sl, s.tp1)
            p_win  = self._ml_model.predict_proba_single(feats)
            s.meta["ml_proba"] = round(p_win, 3)
            if p_win >= self._ml_threshold:
                filtered.append(s)
        return filtered


def _isnan(x) -> bool:
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _best_sl_long(bar, entry, t3_atr,
                  in_4h_ob, in_1h_ob, in_4h_brk, in_1h_brk, in_1h_rej) -> float:
    """
    SL priority for long entries (tightest valid SL wins):
      1. 4H OB bottom − 0.2×ATR   (most common, widest zone)
      2. 1H OB bottom − 0.15×ATR  (tighter precision)
      3. 4H Breaker bottom − 0.2×ATR
      4. 1H Breaker bottom − 0.15×ATR
      5. 1H Rejection block bottom − 0.1×ATR  (tightest — body is the zone)
      6. 2×ATR fallback
    """
    def _try(col, mult):
        v = bar.get(col, np.nan)
        if not _isnan(v):
            sl = float(v) - mult * t3_atr
            if sl < entry:
                return sl
        return None

    if in_4h_ob:
        sl = _try("t2_ob_sl_long", 0.0)   # already includes 0.2×ATR buffer
        if sl is not None: return sl
    if in_1h_ob:
        sl = _try("t2h1_ob_sl_long", 0.0)
        if sl is not None: return sl
    if in_4h_brk:
        sl = _try("t2_brk_sl_long", 0.0)
        if sl is not None: return sl
    if in_1h_brk:
        sl = _try("t2h1_brk_sl_long", 0.0)
        if sl is not None: return sl
    if in_1h_rej:
        sl = _try("t2h1_rej_sl_long", 0.0)
        if sl is not None: return sl
    return entry - 2.0 * t3_atr


def _best_sl_short(bar, entry, t3_atr,
                   in_4h_ob, in_1h_ob, in_4h_brk, in_1h_brk, in_1h_rej) -> float:
    """SL priority for short entries — mirror of _best_sl_long."""
    def _try(col, mult):
        v = bar.get(col, np.nan)
        if not _isnan(v):
            sl = float(v) + mult * t3_atr
            if sl > entry:
                return sl
        return None

    if in_4h_ob:
        sl = _try("t2_ob_sl_short", 0.0)
        if sl is not None: return sl
    if in_1h_ob:
        sl = _try("t2h1_ob_sl_short", 0.0)
        if sl is not None: return sl
    if in_4h_brk:
        sl = _try("t2_brk_sl_short", 0.0)
        if sl is not None: return sl
    if in_1h_brk:
        sl = _try("t2h1_brk_sl_short", 0.0)
        if sl is not None: return sl
    if in_1h_rej:
        sl = _try("t2h1_rej_sl_short", 0.0)
        if sl is not None: return sl
    return entry + 2.0 * t3_atr
