"""
Plan D — ICT + Ichimoku + Indicators Hybrid Strategy
=====================================================
Separate from Plan C (ICT + ML). Do NOT modify Plan C.

Architecture:
  T1 (Macro Bias)  : Ichimoku Cloud (4H/Daily) + EMA 200 trend
                     Replaces Plan C's BOS/CHoCH weekly/daily bias
                     More flexible — Ichimoku gives continuous bias, not binary
  T2 (Zones)       : ICT Order Blocks + FVGs (same as Plan C)
                     PLUS: Ichimoku Kumo (cloud) as an additional zone
  T3 (Confirm)     : 15m CHoCH (same as Plan C)
                     PLUS: RSI divergence + MACD histogram direction
  T4 (Execute)     : 5m OB entry (same as Plan C)

Key difference from Plan C:
  - NOT hardbound on ICT. If Ichimoku gives strong bias but no OB,
    a significant RSI+MACD confluence can still trigger a signal.
  - Lower confidence thresholds by default.
  - More signals per instrument → more training data for ML.

Required timeframes: 1w, 1d, 4h, 1h, 15m, 5m

Models saved to:  models/plan_d_hybrid_{segment}.pkl
Results saved to: results/plan_d_*
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, TradeSetup
from strategies.plan_c_ict.tier1_structure import build_tier1
from strategies.plan_c_ict.tier2_ict_zones import build_tier2
from strategies.plan_c_ict.tier3_precision import build_tier3
from strategies.plan_c_ict.tier4_execution import build_tier4
from strategies.plan_c_ict.strategy import _isnan
from indicators.ichimoku import ichimoku
from indicators.extra_indicators import rsi, macd, bollinger_bands, ema_trend, volume_signals
from indicators.atr import atr as compute_atr

_ML_MODEL_PATH = "models/plan_d_hybrid.pkl"


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _default_cfg():
    return {
        # ICT parameters (more relaxed than Plan C)
        "swing_window_d":   3,
        "swing_window_w":   2,
        "swing_window_h4":  3,
        "swing_window_h1":  3,
        "swing_window_m30": 4,
        "swing_window_m15": 3,
        "choch_lookback":   30,    # more flexible than Plan C's 45
        "sweep_lookback":   12,    # more flexible
        "atr_period":       14,
        "impulse_mult":     1.2,   # lower than Plan C's 1.8 → more OBs detected
        "min_fvg_atr":      0.20,  # lower → smaller FVGs included
        "tp1_rr":           4.0,   # 1:4 minimum
        "confidence_threshold": 0.35,  # lower than Plan C's 0.45

        # Ichimoku parameters
        "ichi_tenkan":     9,
        "ichi_kijun":      26,
        "ichi_senkou_b":   52,
        "ichi_signal_lb":  3,

        # Indicator parameters
        "rsi_period":      14,
        "macd_fast":       12,
        "macd_slow":       26,
        "macd_signal":     9,
        "bb_period":       20,
        "ema_fast":        21,
        "ema_slow":        50,
        "ema_long":        200,

        # Weighting for hybrid confidence score
        "ichi_weight":     0.35,   # Ichimoku bias contribution
        "ict_weight":      0.40,   # ICT zone quality contribution
        "indicator_weight":0.25,   # RSI/MACD/BB contribution
    }


class PlanDHybridStrategy(BaseStrategy):
    """ICT + Ichimoku + Indicators hybrid. Separate from Plan C."""

    name = "plan_d_hybrid"

    @property
    def required_timeframes(self) -> dict[str, list[str]]:
        pcfg = getattr(self._cfg, "plan_d_hybrid", None)
        ltf  = _cfg_get(pcfg, "ltf", "5m")
        syms = _cfg_get(pcfg, "symbols", None)
        if not syms:
            syms = getattr(self._cfg, "symbols", ["BTCUSDT"])
        if ltf == "1h":
            return {s: ["1w","1d","4h","1h"] for s in syms}
        return {s: ["1w","1d","4h","1h","15m","5m"] for s in syms}

    def __init__(self, cfg):
        self._cfg  = cfg
        self._pcfg = getattr(cfg, "plan_d_hybrid", None)
        self._signals: dict[str, pd.DataFrame] = {}
        self._ml_model     = None
        self._ml_threshold = float(_cfg_get(self._pcfg, "ml_threshold", 0.0))
        self._ml_path      = _cfg_get(self._pcfg, "model_path", _ML_MODEL_PATH)
        self._try_load_ml()

    def _try_load_ml(self):
        import os
        if not os.path.exists(self._ml_path):
            return
        try:
            from ml.supervised import ICTTradePredictor
            pred = ICTTradePredictor()
            if pred.load(self._ml_path):
                self._ml_model = pred
                print(f"  [PlanD] ML model loaded ({self._ml_path}, thr={self._ml_threshold})")
        except Exception as e:
            print(f"  [PlanD] ML load failed: {e}")

    def prepare(self, all_data):
        pcfg = getattr(self._cfg, "plan_d_hybrid", None)
        syms = _cfg_get(pcfg, "symbols", None)
        if not syms:
            syms = getattr(self._cfg, "symbols", list({s for s, _ in all_data}))
        for sym in syms:
            try:
                self._signals[sym] = self._build_symbol(sym, all_data)
            except Exception as e:
                import traceback
                print(f"  [PlanD] WARNING: {sym} prepare failed — {e}")

    def _build_symbol(self, sym, all_data):
        def get(tf):
            df = all_data.get((sym, tf))
            if df is None or len(df) < 50:
                raise ValueError(f"Missing or too-short data: {sym} {tf}")
            return df.copy()

        c = _default_cfg()
        pcfg = getattr(self._cfg, "plan_d_hybrid", None)
        if pcfg:
            for k in c:
                c[k] = _cfg_get(pcfg, k, c[k])

        weekly = get("1w")
        daily  = get("1d")
        h4     = get("4h")
        h1     = get("1h")
        ltf_name = _cfg_get(pcfg, "ltf", "5m")
        is_1h_ltf = (ltf_name == "1h")

        if is_1h_ltf:
            m15_df = h1; m5_df = h1
            t2_h4_src, t2_h1_src = daily, h4
        else:
            m15_df = get("15m"); m5_df = get("5m")
            t2_h4_src, t2_h1_src = h4, h1

        # ── Add Ichimoku to 4H (for T1 bias replacement) ────────────────────
        h4_ichi = ichimoku(h4,
                           tenkan_period=c["ichi_tenkan"],
                           kijun_period=c["ichi_kijun"],
                           senkou_b_period=c["ichi_senkou_b"],
                           signal_lookback=c["ichi_signal_lb"])

        # ── Add Ichimoku to daily (macro context) ────────────────────────────
        daily_ichi = ichimoku(daily,
                              tenkan_period=c["ichi_tenkan"],
                              kijun_period=c["ichi_kijun"],
                              senkou_b_period=c["ichi_senkou_b"])
        daily_ichi = ema_trend(daily_ichi, c["ema_fast"], c["ema_slow"], c["ema_long"])

        # ── T1: standard ICT weekly/daily bias ──────────────────────────────
        df_15m = build_tier1(
            daily_df=daily,
            weekly_df=weekly,
            ltf_df=m15_df,
            swing_window_d=c["swing_window_d"],
            swing_window_w=c["swing_window_w"],
            choch_lookback=c["choch_lookback"],
        )

        # ── T2: ICT zones with relaxed impulse_mult ──────────────────────────
        m30_df = all_data.get((sym, "30m"))
        if m30_df is not None and len(m30_df) < 50:
            m30_df = None

        df_15m = build_tier2(
            h4_df=t2_h4_src,
            h1_df=t2_h1_src,
            ltf_df=df_15m,
            swing_window_h4=c["swing_window_h4"],
            swing_window_h1=c["swing_window_h1"],
            atr_period=c["atr_period"],
            impulse_mult=c["impulse_mult"],
            min_fvg_atr=c["min_fvg_atr"],
            m30_df=None if is_1h_ltf else m30_df,
            swing_window_m30=c["swing_window_m30"],
        )

        # ── Add Ichimoku to the 15m/1h confirmation frame ───────────────────
        df_15m = ichimoku(df_15m,
                          tenkan_period=c["ichi_tenkan"],
                          kijun_period=c["ichi_kijun"])
        df_15m = rsi(df_15m, c["rsi_period"])
        df_15m = macd(df_15m, c["macd_fast"], c["macd_slow"], c["macd_signal"])
        df_15m = bollinger_bands(df_15m, c["bb_period"])
        df_15m = volume_signals(df_15m)

        # Merge 4H Ichimoku scores to 15m (backward fill — macro context)
        ichi_4h_cols = ["ichi_bull_score","ichi_bear_score","ichi_above_cloud",
                        "ichi_below_cloud","ichi_cloud_bullish","ichi_tk_bull",
                        "ichi_tk_cross_bull","ichi_tk_cross_bear"]
        h4_ichi_sub = h4_ichi[[c_ for c_ in ichi_4h_cols if c_ in h4_ichi.columns]].copy()
        h4_ichi_sub.columns = [f"h4_{c_}" for c_ in h4_ichi_sub.columns]

        def _strip_tz(idx):
            if hasattr(idx, "tz") and idx.tz is not None:
                return idx.tz_localize(None)
            return idx

        orig_idx = df_15m.index
        df_15m_s = df_15m.copy(); df_15m_s.index = _strip_tz(df_15m.index)
        h4_ichi_sub.index = _strip_tz(h4_ichi_sub.index)

        df_15m_merged = pd.merge_asof(
            df_15m_s.reset_index(), h4_ichi_sub.reset_index(),
            on="timestamp", direction="backward"
        ).set_index("timestamp")
        df_15m_merged.index = orig_idx

        for col in h4_ichi_sub.columns:
            if col not in df_15m_merged.columns:
                df_15m_merged[col] = 0.0

        # ── T3: 15m CHoCH confirmation (same as Plan C) ──────────────────────
        market = _cfg_get(pcfg, "market", "global")
        t3_15m = build_tier3(
            m15_df=df_15m_merged,
            atr_period=c["atr_period"],
            swing_window=c["swing_window_m15"],
            choch_lookback=c["choch_lookback"],
            sweep_lookback=c["sweep_lookback"],
            tp1_rr=c["tp1_rr"],
            market=market,
        )
        t3_15m["t3_tp2_long"]  = t3_15m.get("t2_htf_bsl", pd.Series(np.nan, index=t3_15m.index))
        t3_15m["t3_tp2_short"] = t3_15m.get("t2_htf_ssl", pd.Series(np.nan, index=t3_15m.index))

        # ── Plan D Confluence Score (Ichimoku + ICT + Indicators) ────────────
        ichi_wt = c["ichi_weight"]
        ict_wt  = c["ict_weight"]
        ind_wt  = c["indicator_weight"]

        ichi_bull = t3_15m.get("h4_ichi_bull_score", pd.Series(0.0, index=t3_15m.index))
        ichi_bear = t3_15m.get("h4_ichi_bear_score", pd.Series(0.0, index=t3_15m.index))

        ict_bull  = t3_15m.get("t2_zone_quality_bull",   pd.Series(0.0, index=t3_15m.index)).fillna(0)
        ict_bear  = t3_15m.get("t2_zone_quality_bear",   pd.Series(0.0, index=t3_15m.index)).fillna(0)

        rsi_bull  = (1 - t3_15m["rsi_mid"].fillna(0.5)).clip(0, 1)  # low RSI = good for long
        rsi_bear  = t3_15m["rsi_mid"].fillna(0.5).clip(0, 1)

        macd_bull = t3_15m["macd_bull"].fillna(0)
        macd_bear = (1 - t3_15m["macd_bull"]).fillna(0)

        ind_bull  = (rsi_bull * 0.6 + macd_bull * 0.4).clip(0, 1)
        ind_bear  = (rsi_bear * 0.6 + macd_bear * 0.4).clip(0, 1)

        t3_15m["d_conf_long"]  = (ichi_wt * ichi_bull + ict_wt * ict_bull + ind_wt * ind_bull).clip(0, 1)
        t3_15m["d_conf_short"] = (ichi_wt * ichi_bear + ict_wt * ict_bear + ind_wt * ind_bear).clip(0, 1)

        # ── Merge T3 to 5m and build T4 ──────────────────────────────────────
        T3_COLS = [
            "t3_signal_long","t3_signal_short","t3_choch_bull","t3_choch_bear",
            "t3_in_killzone","t3_session","t3_session_wt",
            "t3_ssl_swept","t3_bsl_swept","t3_in_ote","t3_confidence","t3_atr",
            "conf_long","conf_short","d_conf_long","d_conf_short",
            "t3_tp2_long","t3_tp2_short","t1_bias","t1_strength","t1_pd_zone",
            "t1_target_long","t1_target_short",
            # Pass Ichimoku scores to 5m
            "h4_ichi_bull_score","h4_ichi_bear_score","h4_ichi_above_cloud",
            "h4_ichi_cloud_bullish","h4_ichi_tk_cross_bull","h4_ichi_tk_cross_bear",
            # Indicators
            "rsi_mid","rsi_bull_div","rsi_bear_div","macd_bull","macd_hist_norm",
            "bb_pct","bb_squeeze","ema_bull_align","ema_bear_align",
        ]
        t3_avail = [c_ for c_ in T3_COLS if c_ in t3_15m.columns]

        df_5m = build_tier1(daily_df=daily, weekly_df=weekly, ltf_df=m5_df,
                            swing_window_d=c["swing_window_d"], swing_window_w=c["swing_window_w"],
                            choch_lookback=c["choch_lookback"])
        df_5m = build_tier2(h4_df=t2_h4_src, h1_df=t2_h1_src, ltf_df=df_5m,
                            swing_window_h4=c["swing_window_h4"], swing_window_h1=c["swing_window_h1"],
                            atr_period=c["atr_period"], impulse_mult=c["impulse_mult"],
                            min_fvg_atr=c["min_fvg_atr"])

        def _strip_tz2(idx):
            if hasattr(idx, "tz") and idx.tz is not None:
                return idx.tz_localize(None)
            return idx

        t3_merge = t3_15m[t3_avail].copy()
        t3_merge.index = _strip_tz2(t3_15m.index)
        df5s = df_5m.copy(); df5s.index = _strip_tz2(df_5m.index)

        merged_5m = pd.merge_asof(
            df5s.reset_index(), t3_merge.reset_index(),
            on="timestamp", direction="backward"
        ).set_index("timestamp")
        merged_5m.index = df_5m.index

        for col in t3_avail:
            if col not in merged_5m.columns:
                merged_5m[col] = False if "signal" in col else 0.0

        t4 = build_tier4(m5_df=merged_5m, atr_period=c["atr_period"], tp1_rr=c["tp1_rr"],
                         impulse_mult=c["impulse_mult"])

        # Use Plan D hybrid confidence (Ichimoku + ICT + Indicators) as conf_long
        # conf_long is what the signal collector and on_bar() use for thresholding
        if "d_conf_long" in t4.columns:
            t4["conf_long"]  = t4["d_conf_long"].fillna(0.0)
            t4["conf_short"] = t4["d_conf_short"].fillna(0.0)
        else:
            from strategies.plan_c_ict.confluence import compute_ict_confluence_series
            t4["conf_long"]  = compute_ict_confluence_series(t4, "long")
            t4["conf_short"] = compute_ict_confluence_series(t4, "short")

        try:
            from ml.feature_builder import _add_context_cols
            t4 = _add_context_cols(t4)
        except Exception:
            pass

        return t4

    def on_bar(self, symbol, ts, row, capital=10_000.0):
        df = self._signals.get(symbol)
        if df is None or ts not in df.index:
            return []

        bar = df.loc[ts]
        setups = []
        c = _default_cfg()
        pcfg = getattr(self._cfg, "plan_d_hybrid", None)
        if pcfg:
            for k in c:
                c[k] = _cfg_get(pcfg, k, c[k])

        threshold = c["confidence_threshold"]
        session   = str(bar.get("t3_session", "none"))
        if session == "asia":
            return []

        close  = float(bar.get("close", 0.0))
        t4_atr = float(bar.get("t4_atr", bar.get("t3_atr", close * 0.002))) or close * 0.002

        # ── Long ─────────────────────────────────────────────────────────────
        # Plan D: use Plan D hybrid confidence (Ichimoku + ICT + Indicators)
        # OR fall back to Plan C ICT confidence — whichever is available
        d_conf_long = float(bar.get("d_conf_long", bar.get("conf_long", 0.0)))
        ssl_swept   = bool(bar.get("t3_ssl_swept", False))
        in_kz       = bool(bar.get("t3_in_killzone", False))

        in_ob_long  = (bool(bar.get("t2_in_ob_bull",    False)) or
                       bool(bar.get("t2h1_in_ob_bull",  False)) or
                       bool(bar.get("t2_in_fvg_bull",   False)) or
                       bool(bar.get("t2_in_brk_bull",   False)) or
                       bool(bar.get("t2h1_in_brk_bull", False)) or
                       bool(bar.get("t2h1_in_rej_bull", False)))

        ichi_bull   = float(bar.get("h4_ichi_bull_score", 0.0)) >= 0.40

        # Plan D gate: EITHER (ICT zone + T3 confirm) OR (Ichimoku strong + RSI/MACD align)
        ict_gate  = in_ob_long and ssl_swept and in_kz
        ichi_gate = ichi_bull and bool(bar.get("macd_bull", False)) and in_kz

        if (ict_gate or ichi_gate) and d_conf_long >= threshold:
            t4_long = bool(bar.get("t4_signal_long", False))
            if t4_long:
                sl  = float(bar.get("t4_sl_long",  close - 2.0 * t4_atr))
                tp1 = float(bar.get("t4_tp1_long", close + c["tp1_rr"] * abs(close - sl)))
            else:
                sl  = close - 2.0 * t4_atr
                tp1 = close + c["tp1_rr"] * 2.0 * t4_atr

            tp2_raw = bar.get("t3_tp2_long", bar.get("t1_target_long", np.nan))
            tp2 = float(tp2_raw) if not _isnan(tp2_raw) and float(tp2_raw) > tp1 else None

            if sl < close < tp1:
                setups.append(TradeSetup(
                    symbol=symbol, direction="long",
                    entry=close, sl=sl, tp1=tp1, tp2=tp2,
                    confidence=d_conf_long, atr=t4_atr,
                    meta={
                        "t1_bias":          bar.get("t1_bias","neutral"),
                        "d_conf_long":      round(d_conf_long, 3),
                        "ichi_bull_score":  round(float(bar.get("h4_ichi_bull_score",0)),3),
                        "ict_gate":         ict_gate,
                        "ichi_gate":        ichi_gate,
                        "rsi":              round(float(bar.get("rsi_mid",0.5))*100, 1),
                        "macd_bull":        bool(bar.get("macd_bull",False)),
                        "t3_session":       bar.get("t3_session","none"),
                    }
                ))

        # ── Short ────────────────────────────────────────────────────────────
        d_conf_short = float(bar.get("d_conf_short", bar.get("conf_short", 0.0)))
        bsl_swept    = bool(bar.get("t3_bsl_swept", False))

        in_ob_short  = (bool(bar.get("t2_in_ob_bear",    False)) or
                        bool(bar.get("t2h1_in_ob_bear",  False)) or
                        bool(bar.get("t2_in_fvg_bear",   False)) or
                        bool(bar.get("t2_in_brk_bear",   False)) or
                        bool(bar.get("t2h1_in_brk_bear", False)) or
                        bool(bar.get("t2h1_in_rej_bear", False)))

        ichi_bear    = float(bar.get("h4_ichi_bear_score", 0.0)) >= 0.40
        ict_gate_s   = in_ob_short and bsl_swept and in_kz
        ichi_gate_s  = ichi_bear and not bool(bar.get("macd_bull", True)) and in_kz

        if (ict_gate_s or ichi_gate_s) and d_conf_short >= threshold:
            t4_short = bool(bar.get("t4_signal_short", False))
            if t4_short:
                sl  = float(bar.get("t4_sl_short",  close + 2.0 * t4_atr))
                tp1 = float(bar.get("t4_tp1_short", close - c["tp1_rr"] * abs(sl - close)))
            else:
                sl  = close + 2.0 * t4_atr
                tp1 = close - c["tp1_rr"] * 2.0 * t4_atr

            tp2_raw = bar.get("t3_tp2_short", bar.get("t1_target_short", np.nan))
            tp2 = float(tp2_raw) if not _isnan(tp2_raw) and float(tp2_raw) < tp1 else None

            if sl > close > tp1:
                setups.append(TradeSetup(
                    symbol=symbol, direction="short",
                    entry=close, sl=sl, tp1=tp1, tp2=tp2,
                    confidence=d_conf_short, atr=t4_atr,
                    meta={
                        "t1_bias":          bar.get("t1_bias","neutral"),
                        "d_conf_short":     round(d_conf_short, 3),
                        "ichi_bear_score":  round(float(bar.get("h4_ichi_bear_score",0)),3),
                        "ict_gate":         ict_gate_s,
                        "ichi_gate":        ichi_gate_s,
                        "rsi":              round(float(bar.get("rsi_mid",0.5))*100, 1),
                        "macd_bull":        bool(bar.get("macd_bull",False)),
                        "t3_session":       bar.get("t3_session","none"),
                    }
                ))

        return self._apply_ml_filter(setups, bar)

    def _apply_ml_filter(self, setups, bar):
        if not setups or self._ml_model is None or self._ml_threshold <= 0:
            return setups
        from ml.feature_builder import extract_features
        filtered = []
        for s in setups:
            feats = extract_features(bar, s.direction, s.entry, s.sl, s.tp1)
            # Add Plan D specific features
            feats["ichi_bull_score"] = float(bar.get("h4_ichi_bull_score", 0.0))
            feats["ichi_bear_score"] = float(bar.get("h4_ichi_bear_score", 0.0))
            feats["rsi_div"]         = float(bar.get("rsi_bull_div" if s.direction=="long" else "rsi_bear_div", 0.0))
            feats["macd_hist_norm"]  = float(bar.get("macd_hist_norm", 0.0))
            feats["bb_pct"]          = float(bar.get("bb_pct", 0.5))
            p_win = self._ml_model.predict_proba_single(feats)
            s.meta["ml_proba"] = round(p_win, 3)
            if p_win >= self._ml_threshold:
                filtered.append(s)
        return filtered
