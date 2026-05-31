"""
Plan A — Swing Strategy Orchestrator
======================================
Timeframe hierarchy:
  Tier 1 (bias)  : Weekly + Daily
  Tier 2 (zones) : 4H + 1H
  Tier 3 (entry) : 15m + 5m  ← LTF execution timeframe

Usage:
  strategy = PlanASwingStrategy(cfg)
  strategy.prepare(all_data)          # call once
  setups = strategy.on_bar(sym, ts, row, capital)  # call per LTF bar
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, TradeSetup
from strategies.plan_a_swing.tier1_bias  import build_tier1
from strategies.plan_a_swing.tier2_zones import build_tier2
from strategies.plan_a_swing.tier3_entry import build_tier3
from strategies.plan_a_swing.confluence  import compute_confluence, above_threshold


class PlanASwingStrategy(BaseStrategy):

    name = "plan_a_swing"

    @property
    def required_timeframes(self) -> dict[str, list[str]]:
        syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else ["BTCUSDT", "ETHUSDT"]
        return {s: ["1w", "1d", "4h", "1h", "15m", "5m"] for s in syms}

    def __init__(self, cfg):
        self._cfg  = cfg
        self._pcfg = getattr(cfg, "plan_a_swing", None)
        self._signals: dict[str, pd.DataFrame] = {}  # symbol → prepared LTF df

    # ── prepare ───────────────────────────────────────────────────────────────
    def prepare(self, all_data: dict[tuple[str, str], pd.DataFrame]) -> None:
        """
        Build full indicator/zone/signal DataFrame for each symbol.
        Called once before the bar loop.
        """
        syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else list(
            {s for s, _ in all_data}
        )
        for sym in syms:
            try:
                self._signals[sym] = self._build_symbol(sym, all_data)
            except Exception as e:
                print(f"  [PlanA] WARNING: {sym} prepare failed — {e}")

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
        m15    = get("15m")
        m5     = get("5m")

        cfg = self._pcfg or _default_cfg()

        # Tier 1: weekly+daily bias → merge onto 15m
        df = build_tier1(
            weekly, daily, m15,
            ema_period=cfg.get("ema_period", 50),
            swing_window=cfg.get("swing_window", 3),
        )

        # Tier 2: 4H+1H zones → merge onto 15m
        df = build_tier2(
            h4, h1, df,
            ob_impulse_mult=cfg.get("ob_impulse_mult", 1.5),
            fvg_min_atr=cfg.get("fvg_min_atr", 0.3),
            proximity_atr=cfg.get("proximity_atr", 2.0),
            swing_window=cfg.get("swing_window", 5),
        )

        # Tier 3: 15m entry signals + SL/TP
        df = build_tier3(
            m15, m5, df,
            n_period=cfg.get("n_period", 20),
            vol_pct_thresh=cfg.get("vol_pct_thresh", 80.0),
            body_ratio_thresh=cfg.get("body_ratio_thresh", 0.4),
            sweep_lookback=cfg.get("sweep_lookback", 6),
            atr_k=cfg.get("atr_k", 1.0),
        )

        # Confluence scores
        conf_thresh = cfg.get("conf_threshold", 0.60)
        df = compute_confluence(df, "long",  conf_threshold=conf_thresh)
        df = compute_confluence(df, "short", conf_threshold=conf_thresh)

        return df

    # ── on_bar ────────────────────────────────────────────────────────────────
    def on_bar(
        self,
        symbol: str,
        ts: pd.Timestamp,
        row: pd.Series,
        capital: float,
    ) -> list[TradeSetup]:
        if symbol not in self._signals:
            return []

        df  = self._signals[symbol]
        cfg = self._pcfg or _default_cfg()
        threshold = cfg.get("conf_threshold", 0.60)

        if ts not in df.index:
            return []

        bar = df.loc[ts]
        setups = []

        for direction in ("long", "short"):
            sig_col = f"t3_{direction}_signal"
            if not bar.get(sig_col, False):
                continue
            if bar.get(f"conf_{direction}", 0.0) < threshold:
                continue

            ep  = float(bar["close"])
            atr = float(bar.get("atr", ep * 0.002))

            sl_col  = f"t3_sl_{direction}"
            tp1_col = f"t3_tp1_{direction}"
            tp2_col = f"t3_tp2_{direction}"

            sl  = float(bar.get(sl_col,  ep - atr if direction == "long" else ep + atr))
            tp1 = float(bar.get(tp1_col, ep + 2 * abs(ep - sl) if direction == "long"
                                          else ep - 2 * abs(ep - sl)))
            tp2_raw = bar.get(tp2_col, np.nan)
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
                    "tier1_bias":    bar.get("t1_bias", "neutral"),
                    "tier1_str":     bar.get("t1_strength", 0.0),
                    "tier2_score":   bar.get("t2_zone_score", 0.0),
                    "tier3_score":   bar.get(f"t3_entry_score_{direction}", 0.0),
                    "near_ob":       bar.get("t2_near_ob", False),
                    "near_fvg":      bar.get("t2_near_fvg", False),
                    "near_liq":      bar.get("t2_near_liq", False),
                    "rr":            bar.get(f"rr_{direction}", 2.0),
                    "strategy":      "plan_a_swing",
                },
            ))

        return setups


def _default_cfg() -> dict:
    return {
        "ema_period": 50, "swing_window": 3, "ob_impulse_mult": 1.5,
        "fvg_min_atr": 0.3, "proximity_atr": 2.0, "n_period": 20,
        "vol_pct_thresh": 80.0, "body_ratio_thresh": 0.4,
        "sweep_lookback": 6, "atr_k": 1.0, "conf_threshold": 0.60,
    }
