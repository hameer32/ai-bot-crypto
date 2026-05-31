from __future__ import annotations
import time
import logging
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def build_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    batch_size: int = 1000,
) -> pd.DataFrame:
    all_candles: list = []
    current = since_ms

    while current < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=batch_size)
        except Exception as exc:
            logger.warning("fetch error, retrying: %s", exc)
            time.sleep(5)
            continue

        if not batch:
            break

        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame(columns=OHLCV_COLS)

    df = pd.DataFrame(all_candles, columns=OHLCV_COLS)
    df = df[df["timestamp"] <= until_ms].copy()
    logger.info("Fetched %d candles for %s %s", len(df), symbol, timeframe)
    return df


def fetch_all(
    exchange: ccxt.Exchange,
    symbols: list[str],
    timeframes: list[str],
    lookback_days: int = 730,
) -> dict[tuple[str, str], pd.DataFrame]:
    until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)

    results: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in symbols:
        for tf in timeframes:
            logger.info("Fetching %s %s ...", symbol, tf)
            df = fetch_ohlcv(exchange, symbol, tf, since_ms, until_ms)
            results[(symbol, tf)] = df

    return results
