#!/usr/bin/env python3
"""Run grid search optimization on train split."""
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import load_config
from data.cache import cache_path, load
from optimization.grid_search import grid_search, rank_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Grid search optimization")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--train-end", required=True, help="Train/test split date (YYYY-MM-DD)")
    parser.add_argument("--metric", default="expectancy", help="Metric to optimize")
    parser.add_argument("--output", default="results/grid_search.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ltf_path = cache_path(cfg.data.cache_dir, args.symbol, cfg.ltf)
    htf_path = cache_path(cfg.data.cache_dir, args.symbol, cfg.htf)

    if not os.path.exists(ltf_path):
        print(f"Cache missing. Run: python scripts/run_fetch.py --symbol {args.symbol}")
        sys.exit(1)

    ltf_raw = load(ltf_path)
    htf_raw = load(htf_path)

    results = grid_search(
        ltf_raw, htf_raw,
        base_cfg=cfg,
        metric=args.metric,
        train_end=args.train_end,
        symbol=args.symbol,
    )

    if results.empty:
        print("No results. Check signal counts and data quality.")
        sys.exit(1)

    top = rank_results(results)
    print("\nTop 10 parameter combinations:")
    print(top.to_string(index=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
