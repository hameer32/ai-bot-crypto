#!/usr/bin/env python3
"""Fetch and cache OHLCV data for all configured symbols and timeframes."""
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import load_config
from data.fetcher import build_exchange, fetch_ohlcv
from data.validator import validate
from data.cache import cache_path, save, is_stale
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache OHLCV data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", help="Override symbol (e.g. BTCUSDT)")
    parser.add_argument("--timeframe", help="Override timeframe (e.g. 15m)")
    parser.add_argument("--days", type=int, help="Override lookback days")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache is fresh")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = [args.symbol] if args.symbol else cfg.symbols
    timeframes = [args.timeframe] if args.timeframe else [cfg.ltf, cfg.htf]
    lookback = args.days or cfg.data.lookback_days

    exchange = build_exchange(cfg.exchange)
    until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback)).timestamp() * 1000)

    for symbol in symbols:
        for tf in timeframes:
            path = cache_path(cfg.data.cache_dir, symbol, tf)
            if not args.force and not is_stale(path, max_age_hours=1):
                logger.info("Cache fresh: %s", path)
                continue
            logger.info("Fetching %s %s ...", symbol, tf)
            df = fetch_ohlcv(exchange, symbol, tf, since_ms, until_ms)
            if df.empty:
                logger.warning("No data returned for %s %s", symbol, tf)
                continue
            df = validate(df, tf, cfg.data.outlier_z_threshold)
            save(df, path)
            logger.info("Saved %d rows → %s", len(df), path)


if __name__ == "__main__":
    main()
