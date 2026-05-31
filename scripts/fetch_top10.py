#!/usr/bin/env python3
"""
Institutional Data Fetcher — Top-10 Crypto
===========================================
1. Queries Binance 24h ticker to find top N liquid non-stablecoin USDT pairs
2. Fetches all ICT-required timeframes: 1w, 1d, 4h, 1h, 5m (+ 15m)
3. Validates and caches to cache/

Usage:
  python scripts/fetch_top10.py
  python scripts/fetch_top10.py --n 10 --days 730 --force
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import time
from datetime import datetime, timedelta, timezone

from data.fetcher import build_exchange, fetch_ohlcv
from data.validator import validate
from data.cache import cache_path, save, is_stale

# All timeframes required by the ICT strategy pipeline
ICT_TIMEFRAMES = ["1w", "1d", "4h", "1h", "5m", "15m"]

# Stablecoins and wrapped assets to exclude
EXCLUDE_SYMBOLS = {
    # Stablecoins
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT", "DAIUSDT",
    "USD1USDT", "USDSUSDT", "EURUSDT", "GBPUSDT", "PYUSDUSDT", "SUSDEUSDT",
    # Wrapped / derivative tokens
    "WBTCUSDT", "STETHUSDT", "WBETHUSDT", "LDOUBUSDT", "WEETHUSDT",
}


def get_top_symbols(exchange, n: int = 10) -> list[str]:
    """Rank all USDT pairs by 24h quote volume, return top N non-stablecoin symbols."""
    logger.info("Fetching Binance 24h tickers...")
    tickers = exchange.fetch_tickers()

    candidates = []
    for symbol, t in tickers.items():
        if not symbol.endswith("/USDT"):
            continue
        ccxt_sym = symbol  # e.g. "BTC/USDT"
        binance_sym = symbol.replace("/", "")  # "BTCUSDT"
        if binance_sym in EXCLUDE_SYMBOLS:
            continue
        vol = t.get("quoteVolume") or 0
        candidates.append((binance_sym, vol))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in candidates[:n]]

    print(f"Top {n} by 24h USD volume:")
    for i, (sym, vol) in enumerate(candidates[:n], 1):
        print(f"  {i:2d}. {sym:<14s}  ${vol:>18,.0f}")

    return top


def fetch_symbol(exchange, symbol: str, timeframes: list[str],
                 since_ms: int, until_ms: int,
                 cache_dir: str, force: bool) -> int:
    """Fetch all timeframes for one symbol. Returns count of TFs fetched."""
    fetched = 0
    for tf in timeframes:
        path = cache_path(cache_dir, symbol, tf)
        if not force and not is_stale(path, max_age_hours=1):
            logger.info("  Cache fresh: %s %s", symbol, tf)
            continue
        logger.info("  Fetching %s %s ...", symbol, tf)
        df = fetch_ohlcv(exchange, symbol, tf, since_ms, until_ms)
        if df.empty:
            logger.warning("  No data: %s %s", symbol, tf)
            continue
        df = validate(df, tf, z_thresh=5.0)
        save(df, path)
        logger.info("  Saved %d rows → %s", len(df), path)
        fetched += 1
        time.sleep(0.3)  # be polite
    return fetched


def main():
    parser = argparse.ArgumentParser(description="Fetch top-N crypto for ICT ML training")
    parser.add_argument("--n",     type=int, default=10, help="Number of top symbols to fetch")
    parser.add_argument("--days",  type=int, default=730, help="Lookback days")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache is fresh")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    parser.add_argument("--symbols", default=None,
                        help="Override: comma-separated symbols (skip auto-detect)")
    args = parser.parse_args()

    exchange = build_exchange("binance")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info("Using provided symbols: %s", symbols)
    else:
        symbols = get_top_symbols(exchange, n=args.n)

    until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    os.makedirs(args.cache_dir, exist_ok=True)

    total_fetched = 0
    print(f"\n  Fetching {len(symbols)} symbols × {len(ICT_TIMEFRAMES)} timeframes")
    print(f"  Lookback: {args.days} days  |  Cache: {args.cache_dir}/\n")

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        n = fetch_symbol(exchange, sym, ICT_TIMEFRAMES, since_ms, until_ms,
                         args.cache_dir, args.force)
        total_fetched += n

    print(f"\n  Done. Fetched/refreshed {total_fetched} datasets.")
    print(f"  Symbols: {symbols}")
    print(f"\n  Add these to config.yaml training_symbols for ML training.")


if __name__ == "__main__":
    main()
