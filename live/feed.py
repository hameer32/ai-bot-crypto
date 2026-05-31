from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from data.fetcher import build_exchange, fetch_ohlcv
from data.validator import validate

logger = logging.getLogger(__name__)

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


class PriceFeed:
    """Polling-based price feed. Fetches recent closed candles on interval."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.exchange = build_exchange(cfg.exchange)

    def fetch_latest(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch recent closed candles (last candle excluded — still forming)."""
        until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        tf_ms = self._tf_to_ms(timeframe)
        since_ms = until_ms - limit * tf_ms

        df = fetch_ohlcv(self.exchange, symbol, timeframe, since_ms, until_ms)
        if df.empty:
            return df

        df = validate(df, timeframe, self.cfg.data.outlier_z_threshold)
        # Drop last row — it's the currently-forming candle
        df = df.iloc[:-1]
        return df

    def poll_loop(
        self,
        interval_seconds: int,
        callback: Callable[[str, str, pd.DataFrame], None],
    ) -> None:
        """
        Calls callback(symbol, timeframe, df) every interval_seconds.
        Handles exceptions with exponential backoff.
        """
        backoff = interval_seconds
        while True:
            for symbol in self.cfg.symbols:
                for tf in [self.cfg.ltf, self.cfg.htf]:
                    try:
                        df = self.fetch_latest(symbol, tf)
                        if not df.empty:
                            callback(symbol, tf, df)
                        backoff = interval_seconds
                    except Exception as exc:
                        logger.error("Feed error %s %s: %s", symbol, tf, exc)
                        backoff = min(backoff * 2, 300)

            time.sleep(backoff)

    @staticmethod
    def _tf_to_ms(timeframe: str) -> int:
        mapping = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
        }
        return mapping.get(timeframe, 900_000)
