from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.trade import Trade
    from strategy.entry_exit import TradeSetup

logger = logging.getLogger(__name__)


class LLMReporter:
    """
    Uses Claude API to generate natural language reports.
    Enable via config: ml.llm.enabled = true
    Not used for signal generation — reporting/explainability only.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.ml.llm.model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                logger.error("anthropic SDK not installed. Run: pip install anthropic")
                raise
        return self._client

    def _call(self, prompt: str) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def backtest_summary(self, metrics: dict, trades: list["Trade"]) -> str:
        prompt = f"""You are a quantitative trading analyst. Summarize these backtest results concisely:

Metrics:
{json.dumps(metrics, indent=2)}

Total trades: {len(trades)}
Win rate: {metrics.get('win_rate', 0):.1%}
Expectancy: {metrics.get('expectancy', 0):.4f}
Max drawdown: {metrics.get('max_drawdown', 0):.1%}
Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}

Provide: (1) overall assessment, (2) key strengths, (3) key risks, (4) one improvement suggestion.
Keep it under 200 words."""
        return self._call(prompt)

    def trade_rationale(self, setup: "TradeSetup", context: dict) -> str:
        prompt = f"""You are a trading analyst. Explain why this trade was taken:

Direction: {setup.direction}
Symbol: {setup.symbol}
Entry: {setup.entry_price:.4f}
Stop Loss: {setup.sl_price:.4f}
Take Profit (2R): {setup.tp1_price:.4f}
Confidence: {setup.confidence:.2f}
ATR at entry: {setup.atr_at_entry:.4f}

Market context:
{json.dumps(context, indent=2)}

Explain in 2-3 sentences why this setup was taken based on SMC/liquidity principles."""
        return self._call(prompt)

    def daily_report(self, log_path: str) -> str:
        try:
            with open(log_path) as f:
                entries = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return "No paper trading log found."

        if not entries:
            return "No trades logged today."

        prompt = f"""Summarize today's paper trading activity:

Log entries ({len(entries)} total):
{json.dumps(entries[-20:], indent=2, default=str)}

Provide: (1) number of signals, (2) open/closed trades, (3) estimated session PnL, (4) notable observations.
Under 150 words."""
        return self._call(prompt)
