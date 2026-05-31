"""
Original SMC Strategy — wrapped in BaseStrategy interface.
Delegates to the existing download_and_run pipeline unchanged.
Used as baseline for comparison.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, TradeSetup
from scripts.download_and_run import (
    build_all_indicators, build_signals, build_htf_signals,
    align_htf, generate_signals, HTF_MAP,
)


class OriginalSMCStrategy(BaseStrategy):

    name = "original_smc"

    @property
    def required_timeframes(self) -> dict[str, list[str]]:
        syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else ["BTCUSDT", "ETHUSDT"]
        ltf = getattr(self._cfg, "ltf", "15m")
        htf = HTF_MAP.get(ltf, "4h")
        return {s: [ltf, htf] for s in syms}

    def __init__(self, cfg):
        self._cfg     = cfg
        self._signals: dict[str, pd.DataFrame] = {}

    def prepare(self, all_data):
        ltf = getattr(self._cfg, "ltf", "15m")
        htf = HTF_MAP.get(ltf, "4h")
        r   = self._cfg.rules if hasattr(self._cfg, "rules") else type("r", (), {
            "n_period": 20, "vol_pct_thresh": 80.0,
            "body_ratio_thresh": 0.4, "atr_period": 14,
        })()

        syms = self._cfg.symbols if hasattr(self._cfg, "symbols") else list(
            {s for s, _ in all_data}
        )
        for sym in syms:
            ltf_df = all_data.get((sym, ltf))
            htf_df = all_data.get((sym, htf))
            if ltf_df is None or htf_df is None:
                continue
            df = build_all_indicators(
                ltf_df.copy(),
                n=r.n_period,
                atr_p=r.atr_period,
                vol_w=20,
            )
            df = build_signals(
                df,
                vol_thresh=r.vol_pct_thresh,
                body_thresh=r.body_ratio_thresh,
                n=r.n_period,
            )
            htf_full = build_all_indicators(htf_df.copy(), n=r.n_period, vol_w=20)
            htf_s = build_htf_signals(htf_full)
            df    = align_htf(df, htf_s)
            df    = generate_signals(df, sweep_lb=max(r.n_period // 5, 3))
            df["conf_long"]  = 0.6
            df["conf_short"] = 0.6
            self._signals[sym] = df

    def on_bar(self, symbol, ts, row, capital) -> list[TradeSetup]:
        if symbol not in self._signals:
            return []
        df = self._signals[symbol]
        if ts not in df.index:
            return []
        bar = df.loc[ts]
        r   = self._cfg.rules if hasattr(self._cfg, "rules") else type("r", (), {"atr_k": 1.0})()
        setups = []

        for direction, sig_col in [("long", "long_signal"), ("short", "short_signal")]:
            if not bar.get(sig_col, False):
                continue
            ep  = float(bar["close"])
            atr = float(bar.get("atr", ep * 0.002))
            if direction == "long":
                level = float(bar["sweep_low_level"]) if not pd.isna(bar.get("sweep_low_level", np.nan)) else float(bar.get("roll_low", ep - atr))
                sl  = level - r.atr_k * atr
                tp1 = ep + 2 * abs(ep - sl)
            else:
                level = float(bar["sweep_high_level"]) if not pd.isna(bar.get("sweep_high_level", np.nan)) else float(bar.get("roll_high", ep + atr))
                sl  = level + r.atr_k * atr
                tp1 = ep - 2 * abs(ep - sl)

            if abs(ep - sl) <= 0:
                continue

            setups.append(TradeSetup(
                symbol=symbol, direction=direction,
                entry=ep, sl=sl, tp1=tp1, tp2=None,
                confidence=float(bar.get(f"conf_{direction}", 0.6)),
                atr=atr, meta={"strategy": "original_smc"},
            ))
        return setups
