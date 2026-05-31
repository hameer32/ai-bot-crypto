from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

from backtest.trade import Trade, TradeState
from strategy.entry_exit import build_trade_setup
from strategy.confidence import compute_confidence
from quant.quant_scorer import compute_P_quant
from fusion.combiner import compute_final_confidence, should_trade
from risk.drawdown_guard import DrawdownGuard

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.capital = cfg.risk.initial_capital
        self._trade_id = 0

    def reset(self):
        self.trades = []
        self.equity_curve = []
        self.capital = self.cfg.risk.initial_capital
        self._trade_id = 0

    def run(
        self,
        ltf_df: pd.DataFrame,
        symbol: str,
        train_end: Optional[str] = None,
        df_other: Optional[pd.DataFrame] = None,
    ) -> dict:
        cfg = self.cfg
        guard = DrawdownGuard(self.capital, cfg.risk.max_drawdown_pct)
        open_trade: Optional[Trade] = None

        df = ltf_df.copy()
        if train_end:
            df = df[df.index <= pd.Timestamp(train_end, tz="UTC")]

        for ts, row in df.iterrows():
            # Update open trade first
            if open_trade is not None and open_trade.is_open():
                open_trade = self._update_trade(open_trade, row, ts)
                if not open_trade.is_open():
                    self.capital += open_trade.pnl
                    open_trade = None

            # Kill switch
            if not guard.update(self.capital):
                break

            # Try to open a new trade if none is open
            if open_trade is None:
                direction = None
                if row.get("long_signal", False):
                    direction = "long"
                elif row.get("short_signal", False):
                    direction = "short"

                if direction:
                    confidence = self._compute_fusion_confidence(df, row, ts, direction, symbol, df_other)
                    if confidence >= cfg.fusion.final_threshold:
                        setup = build_trade_setup(
                            row=row,
                            direction=direction,
                            capital=self.capital,
                            risk_pct=cfg.risk.risk_pct_per_trade,
                            atr_k=cfg.rules.atr_k,
                            partial_exit_rr=cfg.risk.partial_exit_rr,
                            symbol=symbol,
                            confidence=confidence,
                        )
                        if setup:
                            self._trade_id += 1
                            open_trade = Trade(
                                id=self._trade_id,
                                symbol=symbol,
                                direction=direction,
                                entry_time=ts,
                                entry_price=setup.entry_price,
                                sl_price=setup.sl_price,
                                tp1_price=setup.tp1_price,
                                trail_activation=setup.trail_activation,
                                size=setup.position_size,
                                confidence=confidence,
                                atr_at_entry=setup.atr_at_entry,
                                rr_target=setup.rr_target,
                            )
                            self.trades.append(open_trade)
                            logger.debug(
                                "OPEN %s %s @ %.4f SL=%.4f TP=%.4f conf=%.2f",
                                direction, symbol, setup.entry_price,
                                setup.sl_price, setup.tp1_price, confidence,
                            )

            self.equity_curve.append((ts, self.capital + (open_trade.unrealized_pnl(row["close"]) if open_trade else 0)))

        # Close any remaining open trade at last close
        if open_trade and open_trade.is_open():
            last_row = df.iloc[-1]
            open_trade.exit_price = float(last_row["close"])
            open_trade.exit_time = df.index[-1]
            open_trade.pnl = open_trade.pnl_per_unit(open_trade.exit_price) * open_trade.remaining_size
            open_trade.state = TradeState.CLOSED_EOD
            self.capital += open_trade.pnl

        return self._summary()

    def _compute_fusion_confidence(
        self, df, row, ts, direction, symbol, df_other
    ) -> float:
        cfg = self.cfg

        # P_rules
        P_rules = None
        if cfg.rules.enabled:
            single_row = pd.DataFrame([row])
            p = compute_confidence(single_row, direction)
            P_rules = p

        # P_quant
        P_quant_series = None
        if cfg.quant.enabled:
            window = df.loc[:ts].tail(200)
            if len(window) >= 20:
                df_other_window = df_other.loc[:ts].tail(200) if df_other is not None else None
                pq = compute_P_quant(window, cfg, direction, symbol=symbol, df_other=df_other_window)
                P_quant_series = pq

        # P_ml (placeholder — returns 0.5 when disabled)
        P_ml = None
        if cfg.ml.enabled:
            P_ml = pd.Series([0.5], index=[ts])

        idx = pd.Index([ts])
        P_r = P_rules.reindex(idx).fillna(0.5) if P_rules is not None else None
        P_q = P_quant_series.reindex(idx).fillna(0.5) if P_quant_series is not None else None
        P_m = P_ml.reindex(idx).fillna(0.5) if P_ml is not None else None

        final = compute_final_confidence(P_r, P_q, P_m, cfg)
        return float(final.iloc[0])

    def _update_trade(self, trade: Trade, row: pd.Series, ts: pd.Timestamp) -> Trade:
        high = float(row["high"])
        low = float(row["low"])
        cfg = self.cfg

        if trade.direction == "long":
            # Check SL
            if low <= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.exit_time = ts
                loss = (trade.sl_price - trade.entry_price) * trade.remaining_size
                trade.pnl += loss
                trade.state = TradeState.CLOSED_SL
                logger.debug("SL hit %s @ %.4f pnl=%.2f", trade.symbol, trade.sl_price, trade.pnl)
                return trade

            # Check TP1 partial exit
            if not trade.partial_exit_done and high >= trade.tp1_price:
                half = trade.remaining_size * 0.5
                gain = (trade.tp1_price - trade.entry_price) * half
                trade.pnl += gain
                trade.remaining_size -= half
                trade.partial_exit_done = True
                # Activate trailing SL
                trade.trailing_sl = trade.entry_price  # move to breakeven
                logger.debug("TP1 hit %s @ %.4f (partial)", trade.symbol, trade.tp1_price)

            # Update trailing SL (only moves upward for longs)
            if trade.partial_exit_done and trade.trailing_sl is not None:
                new_trail = low - cfg.risk.trail_atr_mult * trade.atr_at_entry
                if new_trail > trade.trailing_sl:
                    trade.trailing_sl = new_trail
                if low <= trade.trailing_sl:
                    trade.exit_price = trade.trailing_sl
                    trade.exit_time = ts
                    gain = (trade.trailing_sl - trade.entry_price) * trade.remaining_size
                    trade.pnl += gain
                    trade.state = TradeState.CLOSED_TRAIL
                    logger.debug("TRAIL hit %s @ %.4f pnl=%.2f", trade.symbol, trade.trailing_sl, trade.pnl)

        else:  # short
            if high >= trade.sl_price:
                trade.exit_price = trade.sl_price
                trade.exit_time = ts
                loss = (trade.entry_price - trade.sl_price) * trade.remaining_size
                trade.pnl += loss
                trade.state = TradeState.CLOSED_SL
                return trade

            if not trade.partial_exit_done and low <= trade.tp1_price:
                half = trade.remaining_size * 0.5
                gain = (trade.entry_price - trade.tp1_price) * half
                trade.pnl += gain
                trade.remaining_size -= half
                trade.partial_exit_done = True
                trade.trailing_sl = trade.entry_price

            if trade.partial_exit_done and trade.trailing_sl is not None:
                new_trail = high + cfg.risk.trail_atr_mult * trade.atr_at_entry
                if new_trail < trade.trailing_sl:
                    trade.trailing_sl = new_trail
                if high >= trade.trailing_sl:
                    trade.exit_price = trade.trailing_sl
                    trade.exit_time = ts
                    gain = (trade.entry_price - trade.trailing_sl) * trade.remaining_size
                    trade.pnl += gain
                    trade.state = TradeState.CLOSED_TRAIL

        return trade

    def _summary(self) -> dict:
        return {
            "total_trades": len(self.trades),
            "final_capital": self.capital,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }
