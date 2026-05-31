#!/usr/bin/env python3
"""Start the paper trading loop (no real orders placed)."""
import sys
import os
import logging
import argparse
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.config import load_config
from data.fetcher import build_exchange, fetch_ohlcv
from data.validator import validate
from live.paper_trader import PaperTrader
from strategies import list_strategies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
    "1w": 604_800_000,
}

CANDLES_PER_TF = {
    "1m": 200, "3m": 200, "5m": 300, "15m": 300,
    "30m": 200, "1h": 200, "2h": 200, "4h": 200,
    "6h": 200, "8h": 200, "12h": 200, "1d": 200,
    "1w": 100,
}


def fetch_candles(exchange, symbol: str, timeframe: str, outlier_z: float = 5.0) -> pd.DataFrame:
    from datetime import datetime, timezone
    limit = CANDLES_PER_TF.get(timeframe, 200)
    tf_ms = TF_MS.get(timeframe, 900_000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - limit * tf_ms
    try:
        df = fetch_ohlcv(exchange, symbol, timeframe, since_ms, now_ms)
        if df.empty:
            return df
        df = validate(df, timeframe, outlier_z)
        return df.iloc[:-1]  # drop forming candle
    except Exception as exc:
        logger.error("Fetch error %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Paper trading loop")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", default=None,
                        help="Strategy name (overrides active_strategy in config)")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--log", default="logs/paper_trades.jsonl")
    parser.add_argument("--list", action="store_true", help="List available strategies and exit")
    args = parser.parse_args()

    if args.list:
        for name in list_strategies():
            print(f"  {name}")
        return

    cfg = load_config(args.config)
    if args.strategy:
        cfg.active_strategy = args.strategy

    trader = PaperTrader(cfg, log_path=args.log)
    exchange = build_exchange(cfg.exchange)

    required_tfs = trader.strategy.required_timeframes  # {symbol: [tf1, tf2, ...]}
    strategy_name = getattr(cfg, "active_strategy", "original_smc")

    logger.info("Strategy: %s", strategy_name)
    logger.info("Required TFs: %s", {k: v for k, v in required_tfs.items()})
    logger.info("Poll interval: %ds | Log: %s", args.interval, args.log)
    logger.info("Press Ctrl+C to stop.")

    # Determine the entry (primary) TF per symbol — finest granularity
    def primary_tf(symbol: str) -> str:
        tfs = required_tfs.get(symbol, [cfg.ltf])
        return sorted(tfs, key=lambda t: TF_MS.get(t, 999_999_999))[0]

    backoff = args.interval
    try:
        while True:
            for symbol in cfg.symbols:
                tfs = required_tfs.get(symbol, [cfg.ltf, cfg.htf])
                ptf = primary_tf(symbol)

                for tf in tfs:
                    df = fetch_candles(exchange, symbol, tf, cfg.data.outlier_z_threshold)
                    if not df.empty:
                        trader.update_data(symbol, tf, df)

                # Tick on primary TF
                df_primary = trader._all_data.get((symbol, ptf))
                if df_primary is not None and not df_primary.empty:
                    ts = df_primary.index[-1]
                    row = df_primary.iloc[-1]
                    trader.tick(symbol, ts, row)

            logger.debug("Status: %s", trader.status())
            time.sleep(backoff)

    except KeyboardInterrupt:
        logger.info("Stopped. Final status: %s", trader.status())


if __name__ == "__main__":
    main()
