#!/usr/bin/env python3
"""
Universe Data Fetcher
=====================
Fetches all asset classes required for expanded ICT ML training:

  1. Crypto (Binance CCXT)
     - Top-N by 24h USDT volume (auto-detected; default 20)
     Timeframes: 1w, 1d, 4h, 1h, 5m, 15m

  2. Commodities (yfinance — COMEX futures)
     - GC=F   Gold Futures  (COMEX, nearly 24h market — ideal for ICT)
     - SI=F   Silver Futures (COMEX)
     Timeframes: 1w, 1d, 4h, 1h, 5m, 15m
     Note: XAUUSDT / XAGUSDT do not exist on Binance spot.
           COMEX Gold trades Sun–Fri 23:00–22:00 UTC, fully overlapping
           London and NY killzones — perfect for ICT institutional analysis.

  3. Indian Indices (yfinance)
     - ^NSEI    NIFTY 50
     - ^CNX200  NIFTY 200
     - ^BSESN   SENSEX (BSE)
     - ^NSEBANK NIFTY Bank
     Timeframes: 1w, 1d, 1h   (5m/15m limited to ~60 days via --intraday flag)

  4. Top 50 NIFTY 50 Stocks (yfinance .NS)
     Timeframes: 1w, 1d, 1h

Usage:
  python scripts/fetch_universe.py                    # full universe
  python scripts/fetch_universe.py --crypto-only      # only Binance crypto
  python scripts/fetch_universe.py --india-only       # only Indian markets
  python scripts/fetch_universe.py --commodities-only # only Gold/Silver
  python scripts/fetch_universe.py --n 20 --days 730  # top-20 crypto, 2yr history
  python scripts/fetch_universe.py --force            # ignore cache freshness
  python scripts/fetch_universe.py --intraday         # also fetch 5m/15m for Indian

Output:
  cache/<SYMBOL>_<TF>.parquet    (all assets, same normalised OHLCV format)

After fetching, run:
  python scripts/run_ml_train.py
The trainer auto-detects all symbols in cache that have all required TFs.
"""
import sys, os, argparse, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from datetime import datetime, timedelta, timezone

from data.fetcher import build_exchange, fetch_ohlcv
from data.yfinance_fetcher import fetch_yf_ohlcv
from data.validator import validate
from data.cache import cache_path, save, is_stale

# ── Timeframes ─────────────────────────────────────────────────────────────────

# Full ICT pipeline — all fine-grained timeframes.
# 1m  : last 7 days  (Binance) — ultra-precise 1m Silver Bullet entry
# 5m  : 2-year history — LTF execution (T3)
# 15m : 2-year history — ICT Silver Bullet TF, T3 precision
# 30m : 2-year history — intermediate zone confirmation (between 1h and 15m)
# 1h  : 2-year history — T2 zone source
# 4h  : 2-year history — T2 zone source (primary)
# 1d  : 2-year history — T1 daily bias
# 1w  : 2-year history — T1 weekly bias (highest timeframe)
CRYPTO_TIMEFRAMES = ["1w", "1d", "4h", "1h", "30m", "15m"]

# yfinance: 1h/4h up to 720 days, 30m/15m up to 59 days, 1w/1d unlimited
INDIA_FULL_TIMEFRAMES   = ["1w", "1d", "4h", "1h", "30m", "15m"]
INDIA_STABLE_TIMEFRAMES = ["1w", "1d", "4h", "1h"]  # 4h resampled from 1h

# ── Commodities via yfinance (COMEX / NYMEX Futures) ─────────────────────────
COMMODITY_YF = {
    "GOLD":   "GC=F",   # COMEX Gold     — Sun–Fri 23:00–22:00 UTC
    "SILVER": "SI=F",   # COMEX Silver
    "OIL":    "CL=F",   # NYMEX WTI Crude Oil
    "NATGAS": "NG=F",   # NYMEX Natural Gas
    "COPPER": "HG=F",   # COMEX Copper
    "PLATINUM": "PL=F", # NYMEX Platinum
    "PALLADIUM": "PA=F",# NYMEX Palladium
    "COCOA": "CC=F",    # ICE Cocoa
    "COFFEE": "KC=F",   # ICE Coffee
    "WHEAT":  "ZW=F",   # CBOT Wheat
}

# ── Forex via yfinance — Major + Minor pairs (24/5, London/NY killzones apply) ──
FOREX_YF = {
    # ── 7 Major Pairs ─────────────────────────────────────────────────────────
    "EURUSD": "EURUSD=X",   # Euro / USD — highest volume
    "GBPUSD": "GBPUSD=X",   # British Pound / USD
    "USDJPY": "USDJPY=X",   # USD / Japanese Yen
    "USDCHF": "USDCHF=X",   # USD / Swiss Franc
    "AUDUSD": "AUDUSD=X",   # Australian Dollar / USD
    "USDCAD": "USDCAD=X",   # USD / Canadian Dollar
    "NZDUSD": "NZDUSD=X",   # New Zealand Dollar / USD
    # ── Major Crosses ─────────────────────────────────────────────────────────
    "EURGBP": "EURGBP=X",   # Euro / GBP
    "EURJPY": "EURJPY=X",   # Euro / JPY
    "GBPJPY": "GBPJPY=X",   # GBP / JPY — high volatility cross
    "EURAUD": "EURAUD=X",   # Euro / AUD
    "EURCAD": "EURCAD=X",   # Euro / CAD
    "GBPAUD": "GBPAUD=X",   # GBP / AUD
    "GBPCAD": "GBPCAD=X",   # GBP / CAD
    "AUDCAD": "AUDCAD=X",   # AUD / CAD
    "AUDNZD": "AUDNZD=X",   # AUD / NZD
    "NZDJPY": "NZDJPY=X",   # NZD / JPY
    "CADJPY": "CADJPY=X",   # CAD / JPY
    "CHFJPY": "CHFJPY=X",   # CHF / JPY
    "AUDCHF": "AUDCHF=X",   # AUD / CHF — 20th pair
}
FOREX_TIMEFRAMES = ["1w", "1d", "4h", "1h", "30m", "15m"]  # no 5m for long history
# Commodity timeframes: full ICT pipeline incl. fine-grained intraday
COMMODITY_TIMEFRAMES = ["1w", "1d", "4h", "1h", "30m", "15m"]

# ── Indian Indices ─────────────────────────────────────────────────────────────
# yfinance tickers (Yahoo Finance symbol format)
INDIA_INDICES = {
    "NIFTY50":   "^NSEI",
    "NIFTY200":  "^CNX200",
    "SENSEX":    "^BSESN",
    "NIFTYBANK": "^NSEBANK",
}

# ── NIFTY 50 Constituent Stocks ────────────────────────────────────────────────
# Current NIFTY 50 components (NSE format for yfinance: symbol.NS)
NIFTY50_STOCKS_YF = [
    "ADANIENT.NS",   "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS",   "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BHARTIARTL.NS", "BPCL.NS",       "BRITANNIA.NS",  "CIPLA.NS",
    "COALINDIA.NS",  "DIVISLAB.NS",   "DRREDDY.NS",    "EICHERMOT.NS",
    "GRASIM.NS",     "HCLTECH.NS",    "HDFCBANK.NS",   "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS",   "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS",       "ITC.NS",        "JSWSTEEL.NS",
    "KOTAKBANK.NS",  "LT.NS",         "M&M.NS",        "MARUTI.NS",
    "NESTLEIND.NS",  "NTPC.NS",       "ONGC.NS",       "POWERGRID.NS",
    "RELIANCE.NS",   "SBICARD.NS",    "SBIN.NS",       "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",  "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TCS.NS",        "TECHM.NS",      "TITAN.NS",      "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]

# Canonical cache symbol name: strip .NS suffix and ^prefix for clean filenames
def _cache_sym(yf_ticker: str) -> str:
    s = yf_ticker
    if s.startswith("^"):
        s = s[1:]
    if s.endswith(".NS"):
        s = s[:-3]
    return s.replace("&", "")   # M&M → MM (filesystem safe)


# ── Binance fetch helpers ──────────────────────────────────────────────────────

def _get_top_crypto(exchange, n: int) -> list[str]:
    """Rank USDT pairs by 24h quote volume, return top N non-stablecoin symbols."""
    EXCLUDE = {
        "USDCUSDT","BUSDUSDT","TUSDUSDT","USDPUSDT","FDUSDUSDT","DAIUSDT",
        "USD1USDT","USDSUSDT","EURUSDT","GBPUSDT","PYUSDUSDT","SUSDEUSDT",
        "WBTCUSDT","STETHUSDT","WBETHUSDT","LDOUBUSDT","WEETHUSDT",
    }
    logger.info("Fetching Binance 24h tickers for top-%d ranking ...", n)
    tickers = exchange.fetch_tickers()
    candidates = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        binance_sym = sym.replace("/", "")
        if binance_sym in EXCLUDE:
            continue
        vol = t.get("quoteVolume") or 0
        candidates.append((binance_sym, vol))
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in candidates[:n]]
    print(f"\n  Top {n} by 24h volume:")
    for i, (s, v) in enumerate(candidates[:n], 1):
        print(f"    {i:2d}. {s:<14s}  ${v:>18,.0f}")
    return top


def _fetch_binance_symbol(exchange, symbol, timeframes, since_ms, until_ms,
                           cache_dir, force):
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
        time.sleep(0.3)
    return fetched


# ── yfinance fetch helpers ─────────────────────────────────────────────────────

def _fetch_yf_symbol(yf_ticker, cache_sym, timeframes, lookback_days,
                     cache_dir, force):
    """Fetch one yfinance ticker for all requested timeframes."""
    fetched = 0
    for tf in timeframes:
        path = cache_path(cache_dir, cache_sym, tf)
        if not force and not is_stale(path, max_age_hours=6):
            logger.info("  Cache fresh: %s %s", cache_sym, tf)
            continue
        logger.info("  Fetching %s (%s) %s ...", cache_sym, yf_ticker, tf)
        df = fetch_yf_ohlcv(yf_ticker, tf, lookback_days=lookback_days)
        if df.empty:
            logger.warning("  No data: %s %s", cache_sym, tf)
            continue
        save(df, path)
        logger.info("  Saved %d rows → %s", len(df), path)
        fetched += 1
        time.sleep(0.2)   # polite rate limiting
    return fetched


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch expanded symbol universe for ICT ML training"
    )
    parser.add_argument("--n",           type=int, default=20,
                        help="Top-N crypto symbols to fetch (default 20)")
    parser.add_argument("--days",        type=int, default=730,
                        help="Lookback in calendar days (default 730 = 2 years)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-fetch even if cache is fresh")
    parser.add_argument("--cache-dir",   default="cache",
                        help="Cache directory (default: cache/)")
    parser.add_argument("--crypto-only",      action="store_true",
                        help="Only fetch Binance crypto")
    parser.add_argument("--india-only",       action="store_true",
                        help="Only fetch Indian markets")
    parser.add_argument("--commodities-only", action="store_true",
                        help="Only fetch commodities")
    parser.add_argument("--forex-only",       action="store_true",
                        help="Only fetch Forex pairs")
    parser.add_argument("--intraday",    action="store_true",
                        help="Also fetch 5m/15m for Indian markets (~60 days each)")
    parser.add_argument("--no-stocks",   action="store_true",
                        help="Skip NIFTY 50 constituent stocks (fetch indices only)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    total_fetched = 0

    fetch_crypto      = not args.india_only and not args.commodities_only and not args.forex_only
    fetch_commodities = not args.crypto_only and not args.india_only and not args.forex_only
    fetch_india       = not args.crypto_only and not args.commodities_only and not args.forex_only
    fetch_forex       = not args.crypto_only and not args.india_only and not args.commodities_only

    # ── 1. Binance Crypto ───────────────────────────────────────────────────────
    if fetch_crypto:
        print("\n" + "═" * 60)
        print("  CRYPTO (Binance)")
        print("═" * 60)
        exchange = build_exchange("binance")

        until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_ms  = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

        top_crypto = _get_top_crypto(exchange, args.n)
        print(f"\n  Fetching top-{args.n} crypto × {len(CRYPTO_TIMEFRAMES)} TFs ...")
        for i, sym in enumerate(top_crypto, 1):
            print(f"  [{i}/{len(top_crypto)}] {sym}")
            n = _fetch_binance_symbol(
                exchange, sym, CRYPTO_TIMEFRAMES, since_ms, until_ms,
                args.cache_dir, args.force
            )
            total_fetched += n

    # ── 2. Commodities — Gold & Silver (yfinance / COMEX Futures) ───────────────
    if fetch_commodities:
        print("\n" + "═" * 60)
        print("  COMMODITIES (yfinance — COMEX Futures)")
        print("  Gold (GC=F) + Silver (SI=F) — nearly 24h market, full ICT TFs")
        print("═" * 60)
        for name, yf_ticker in COMMODITY_YF.items():
            print(f"\n  Fetching {name} ({yf_ticker}) ...")
            n = _fetch_yf_symbol(
                yf_ticker, name, COMMODITY_TIMEFRAMES, args.days,
                args.cache_dir, args.force
            )
            total_fetched += n

    # ── 4. Indian Markets (yfinance) ────────────────────────────────────────────
    if fetch_india:
        india_tfs = (INDIA_FULL_TIMEFRAMES if args.intraday
                     else INDIA_STABLE_TIMEFRAMES)

        print("\n" + "═" * 60)
        print(f"  INDIAN MARKETS (yfinance)")
        print(f"  Timeframes: {india_tfs}")
        print(f"  Note: 5m/15m limited to ~60 days; 4h resampled from 1h")
        print(f"  Note: Market hours 09:15–15:30 IST — killzone = 'none' in ML features")
        print("═" * 60)

        # Indices
        print("\n  Indices ...")
        for name, yf_ticker in INDIA_INDICES.items():
            sym = _cache_sym(yf_ticker)
            print(f"  Fetching {name} ({yf_ticker}) → cached as {sym}")
            n = _fetch_yf_symbol(
                yf_ticker, sym, india_tfs, args.days, args.cache_dir, args.force
            )
            total_fetched += n

        # NIFTY 50 constituent stocks
        if not args.no_stocks:
            print(f"\n  NIFTY 50 constituent stocks ({len(NIFTY50_STOCKS_YF)} symbols) ...")
            for i, yf_ticker in enumerate(NIFTY50_STOCKS_YF, 1):
                sym = _cache_sym(yf_ticker)
                print(f"  [{i}/{len(NIFTY50_STOCKS_YF)}] {yf_ticker} → {sym}")
                n = _fetch_yf_symbol(
                    yf_ticker, sym, india_tfs, args.days, args.cache_dir, args.force
                )
                total_fetched += n

    # ── 5. Forex Pairs (yfinance) ─────────────────────────────────────────────
    if fetch_forex:
        print("\n" + "═" * 60)
        print(f"  FOREX PAIRS (yfinance)")
        print(f"  Timeframes: {FOREX_TIMEFRAMES}")
        print(f"  London/NY killzones apply — 24/5 market")
        print("═" * 60)
        for name, yf_ticker in FOREX_YF.items():
            print(f"\n  Fetching {name} ({yf_ticker}) ...")
            n = _fetch_yf_symbol(
                yf_ticker, name, FOREX_TIMEFRAMES, args.days,
                args.cache_dir, args.force
            )
            total_fetched += n

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Done. Fetched/refreshed {total_fetched} datasets.")

    # List what's in cache
    cached = sorted([
        f.replace(".parquet", "")
        for f in os.listdir(args.cache_dir)
        if f.endswith(".parquet")
    ])
    symbols_cached = sorted(set("_".join(f.split("_")[:-1]) for f in cached))
    print(f"  Total symbols in cache: {len(symbols_cached)}")
    print(f"  Symbols: {symbols_cached}")
    print(f"\n  To retrain ML: python scripts/run_ml_train.py")
    print(f"  (training_symbols: [] in config.yaml = auto-detect all available)")


if __name__ == "__main__":
    main()
