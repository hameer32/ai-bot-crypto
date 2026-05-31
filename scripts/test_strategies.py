#!/usr/bin/env python3
"""Quick smoke test for Plan A and Plan B strategies."""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.run_strategy import load_cache, run_backtest, compute_metrics
from config.config import load_config
from strategies.registry import get_strategy
import yaml

def main():
    cfg = load_config("config.yaml")
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    for key in ("plan_a_swing", "plan_b_scalp"):
        if key in raw:
            setattr(cfg, key, raw[key])

    all_data = load_cache()
    print(f"Loaded {len(all_data)} datasets\n")

    results = {}

    for strat_name, ltf in [
        ("original_smc", "15m"),
        ("plan_a_swing",  "15m"),
        ("plan_b_scalp",  "5m"),
    ]:
        print(f"{'='*60}")
        print(f"  {strat_name}  (LTF: {ltf})")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            strat  = get_strategy(strat_name, cfg)
            trades, equity, final = run_backtest(
                strat, all_data, cfg.symbols, ltf,
                capital=10000, risk_pct=0.01, max_dd=0.10,
            )
            m = compute_metrics(trades, equity, 10000)
            elapsed = time.time() - t0
            print(f"  Trades:     {m['trades']}")
            print(f"  Win Rate:   {m['win_rate']:.0%}")
            print(f"  Avg RR:     {m['avg_rr']:+.2f}")
            print(f"  Expectancy: {m['expectancy']:+.1f}")
            print(f"  Max DD:     {m['max_dd']:.1%}")
            print(f"  Return:     {m['ret_pct']:+.1f}%")
            print(f"  Time:       {elapsed:.0f}s")
            results[strat_name] = m
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        print()

    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Strategy':<20} {'Trades':>7} {'WinR':>6} {'Expect':>8} {'Return':>8}")
    print("  " + "-"*55)
    for name, m in results.items():
        print(f"  {name:<20} {m['trades']:>7} {m['win_rate']:>5.0%} "
              f"{m['expectancy']:>+8.1f} {m['ret_pct']:>+7.1f}%")

if __name__ == "__main__":
    main()
