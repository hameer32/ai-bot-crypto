from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class DrawdownGuard:
    def __init__(self, peak_capital: float, max_dd_pct: float = 0.10):
        self.peak = peak_capital
        self.max_dd_pct = max_dd_pct
        self.active = True
        self.current_drawdown = 0.0

    def update(self, current_capital: float) -> bool:
        """
        Update peak and drawdown. Returns True if trading should continue,
        False if the kill switch has been triggered.
        """
        if current_capital > self.peak:
            self.peak = current_capital

        self.current_drawdown = (self.peak - current_capital) / self.peak

        if self.active and self.current_drawdown >= self.max_dd_pct:
            self.active = False
            logger.warning(
                "KILL SWITCH: drawdown %.1f%% exceeded limit %.1f%%",
                self.current_drawdown * 100,
                self.max_dd_pct * 100,
            )

        return self.active

    def reset(self, capital: float) -> None:
        self.peak = capital
        self.current_drawdown = 0.0
        self.active = True
