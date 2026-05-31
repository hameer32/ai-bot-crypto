#!/usr/bin/env python3
"""Run a full backtest on cached OHLCV data."""
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import load_config
from data.cache import cache_path, load
from indicators import build_indicators
from strategy.mtf_strategy import build_ltf_signals, build_htf_signals, align_htf_to_ltf, generate_signals
from strategy.confidence import filter_by_confidence
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ltf_path = cache_path(cfg.data.cache_dir, args.symbol, cfg.ltf)
    htf_path = cache_path(cfg.data.cache_dir, args.symbol, cfg.htf)

    if not os.path.exists(ltf_path) or not os.path.exists(htf_path):
        print(f"Cache missing. Run: python scripts/run_fetch.py --symbol {args.symbol}")
        sys.exit(1)

    logger.info("Loading data ...")
    ltf_raw = load(ltf_path)
    htf_raw = load(htf_path)

    if args.start:
        ltf_raw = ltf_raw[ltf_raw.index >= args.start]
        htf_raw = htf_raw[htf_raw.index >= args.start]

    # Build indicators + signals
    logger.info("Building indicators and signals ...")
    ltf = build_indicators(ltf_raw, cfg)
    ltf = build_ltf_signals(ltf, cfg)
    htf = build_indicators(htf_raw, cfg)
    htf = build_htf_signals(htf, cfg)
    ltf = align_htf_to_ltf(ltf, htf)
    ltf = generate_signals(ltf, cfg)
    ltf = filter_by_confidence(ltf, cfg.rules.confidence_threshold)

    # Find second symbol for correlation (optional)
    df_other = None
    other_symbols = [s for s in cfg.symbols if s != args.symbol]
    if other_symbols and cfg.quant.correlation.enabled:
        other_path = cache_path(cfg.data.cache_dir, other_symbols[0], cfg.ltf)
        if os.path.exists(other_path):
            df_other = load(other_path)
            if args.start:
                df_other = df_other[df_other.index >= args.start]

    logger.info("Running backtest ...")
    engine = BacktestEngine(cfg)
    summary = engine.run(ltf, symbol=args.symbol, train_end=args.end, df_other=df_other)

    metrics = compute_metrics(summary["trades"], summary["equity_curve"])
    print_report(metrics)

    logger.info("Final capital: $%.2f (started $%.2f)", summary["final_capital"], cfg.risk.initial_capital)


if __name__ == "__main__":
    main()
