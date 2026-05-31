"""
Plan B — Scalp Strategy Orchestrator
======================================
Timeframe hierarchy:
  Tier 1 (bias)  : 4H + 1H
  Tier 2 (zones) : 30m + 15m
  Tier 3 (entry) : 5m + 1m  ← LTF execution timeframe

Usage:
  strategy = PlanBScalpStrategy(cfg)
  strategy.prepare(all_data)
  setups = strategy.on_bar(sym, ts, row, capital)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, TradeSetup
from strategies.plan_b_scalp.tier1_bias  import build_tier1
from strategies.plan_b_scalp.tier2_zones import build_tier2
from strategies.plan_b_scalp.tier3_entry import build_tier3
from strategies.plan_a_swing.confluence  import compute_confluence


class PlanBScalpStrategy(BaseStrategy):

    name = "plan_b_scalp"

    @property
    def required_timeframes(self) -> dict[str, list[str]]:
        syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else ["BTCUSDT", "ETHUSDT"]
        return {s: ["4h", "1h", "30m", "15m", "5m", "1m"] for s in syms}

    def __init__(self, cfg):
        self._cfg  = cfg
        self._pcfg = getattr(cfg, "plan_b_scalp", None)
        self._signals: dict[str, pd.DataFrame] = {}

    def prepare(self, all_data: dict[tuple[str, str], pd.DataFrame]) -> None:
        syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else list(
            {s for s, _ in all_data}
        )
        for sym in syms:
            try:
                self._signals[sym] = self._build_symbol(sym, all_data)
            except Exception as e:
                print(f"  [PlanB] WARNING: {sym} prepare failed — {e}")

    def _build_symbol(self, sym, all_data):
        def get(tf):
            df = all_data.get((sym, tf))
            if df is None or len(df) < 50:
                raise ValueError(f"Missing or too-short data: {sym} {tf}")
            return df.copy()

        h4  = get("4h")
        h1  = get("1h")
        m30 = get("30m")
        m15 = get("15m")
        m5  = get("5m")
        m1  = get("1m")

        cfg = self._pcfg or _default_cfg()

        # Tier 1: 4H + 1H bias → merge onto 5m
        df = build_tier1(
            h4, h1, m5,
            ema_period=cfg.get("ema_period", 50),
            swing_window=cfg.get("swing_window", 5),
        )

        # Tier 2: 30m + 15m zones → merge onto 5m
        df = build_tier2(
            m30, m15, df,
            ob_impulse_mult=cfg.get("ob_impulse_mult", 1.2),
            fvg_min_atr=cfg.get("fvg_min_atr", 0.2),
            proximity_atr=cfg.get("proximity_atr", 1.5),
            swing_window=cfg.get("swing_window", 4),
        )

        # Tier 3: 5m entry signals, 1m for finer confirmation
        df = build_tier3(
            m5, m1, df,
            n_period=cfg.get("n_period", 15),
            vol_pct_thresh=cfg.get("vol_pct_thresh", 75.0),
            body_ratio_thresh=cfg.get("body_ratio_thresh", 0.3),
            sweep_lookback=cfg.get("sweep_lookback", 4),
            atr_k=cfg.get("atr_k", 0.8),
        )

        # Confluence (use Plan A scorer with scalp threshold)
        conf_thresh = cfg.get("conf_threshold", 0.55)
        df = compute_confluence(df, "long",  conf_threshold=conf_thresh)
        df = compute_confluence(df, "short", conf_threshold=conf_thresh)

        return df

    def on_bar(self, symbol, ts, row, capital) -> list[TradeSetup]:
        if symbol not in self._signals:
            return []

        df  = self._signals[symbol]
        cfg = self._pcfg or _default_cfg()
        threshold = cfg.get("conf_threshold", 0.55)

        if ts not in df.index:
            return []

        bar = df.loc[ts]
        setups = []

        for direction in ("long", "short"):
            if not bar.get(f"t3_{direction}_signal", False):
                continue
            if bar.get(f"conf_{direction}", 0.0) < threshold:
                continue

            ep  = float(bar["close"])
            atr = float(bar.get("atr", ep * 0.001))

            sl  = float(bar.get(f"t3_sl_{direction}",
                                  ep - atr if direction == "long" else ep + atr))
            tp1 = float(bar.get(f"t3_tp1_{direction}",
                                  ep + 1.5 * abs(ep - sl) if direction == "long"
                                  else ep - 1.5 * abs(ep - sl)))
            tp2_raw = bar.get(f"t3_tp2_{direction}", np.nan)
            tp2 = float(tp2_raw) if not pd.isna(tp2_raw) else None

            if abs(ep - sl) <= 0:
                continue

            setups.append(TradeSetup(
                symbol=symbol,
                direction=direction,
                entry=ep,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                confidence=float(bar.get(f"conf_{direction}", 0.5)),
                atr=atr,
                meta={
                    "tier1_bias":  bar.get("t1_bias", "neutral"),
                    "tier1_str":   bar.get("t1_strength", 0.0),
                    "tier2_score": bar.get("t2_zone_score", 0.0),
                    "tier3_score": bar.get(f"t3_entry_score_{direction}", 0.0),
                    "near_ob":     bar.get("t2_near_ob", False),
                    "near_fvg":    bar.get("t2_near_fvg", False),
                    "rr":          bar.get(f"rr_{direction}", 1.5),
                    "strategy":    "plan_b_scalp",
                },
            ))

        return setups


def _default_cfg() -> dict:
    return {
        "ema_period": 50, "swing_window": 5, "ob_impulse_mult": 1.2,
        "fvg_min_atr": 0.2, "proximity_atr": 1.5, "n_period": 15,
        "vol_pct_thresh": 75.0, "body_ratio_thresh": 0.3,
        "sweep_lookback": 4, "atr_k": 0.8, "conf_threshold": 0.55,
    }
