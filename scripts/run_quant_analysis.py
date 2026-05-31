#!/usr/bin/env python3
"""Standalone quant model diagnostics — visualize regime, microstructure, momentum."""
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import load_config
from data.cache import cache_path, load
from indicators import build_indicators
from quant.mean_reversion import zscore, bollinger_signal, mean_reversion_halflife
from quant.regime import HMMRegimeDetector
from quant.microstructure import microstructure_score
from quant.momentum import tsm_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quant model diagnostics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = cache_path(cfg.data.cache_dir, args.symbol, cfg.ltf)

    if not os.path.exists(path):
        print(f"Cache missing. Run: python scripts/run_fetch.py --symbol {args.symbol}")
        sys.exit(1)

    df = load(path)
    df = build_indicators(df, cfg)

    print(f"\n{'='*50}")
    print(f"  Quant Analysis: {args.symbol} {cfg.ltf}")
    print(f"{'='*50}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")

    # Mean reversion
    z = zscore(df, cfg.quant.mean_reversion.zscore_window)
    bb = bollinger_signal(df, cfg.quant.mean_reversion.bb_period, cfg.quant.mean_reversion.bb_std)
    hl = mean_reversion_halflife(df["close"].dropna())
    print(f"\n  Mean Reversion:")
    print(f"    Z-score now: {z.iloc[-1]:.2f}")
    print(f"    BB signal (0=short,1=long): {bb.iloc[-1]:.2f}")
    print(f"    Half-life (bars): {hl:.1f}")

    # Regime
    if cfg.quant.regime.enabled:
        print(f"\n  HMM Regime Detection:")
        detector = HMMRegimeDetector(cfg.quant.regime.n_states)
        detector.fit(df, cfg.quant.regime.lookback)
        states = detector.predict_state(df)
        state_counts = states.value_counts()
        print(f"    State distribution: {state_counts.to_dict()}")
        print(f"    Current regime: {states.iloc[-1]}")

    # Microstructure
    ms = microstructure_score(df, cfg.quant.microstructure.kyle_window, cfg.quant.microstructure.amihud_window)
    print(f"\n  Microstructure Score (current): {ms.iloc[-1]:.4f}")
    print(f"  Microstructure Score (mean): {ms.mean():.4f}")

    # Momentum
    tsm = tsm_score(df, cfg.quant.momentum.tsm_lookback, cfg.quant.momentum.tsm_short_window)
    print(f"\n  TSM Score (current): {tsm.iloc[-1]:.4f}  (>0.5=long bias, <0.5=short)")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
