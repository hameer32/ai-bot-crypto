"""
Plan B — Confluence Scorer (Scalp)
====================================
Same structure as Plan A but with tighter weighting:
  - Tier 2 zones weighted more heavily (50%) — on scalp TFs, zone precision matters more
  - Tier 1 bias weighted less (20%) — intraday bias can shift quickly
  - Tier 3 entry trigger: 30%

Lower default threshold (0.55) since scalp entries need to fire more frequently.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategies.plan_a_swing.confluence import compute_confluence, above_threshold

# re-export with scalp-adjusted defaults
TIER1_WEIGHT = 0.20
TIER2_WEIGHT = 0.50
TIER3_WEIGHT = 0.30

__all__ = ["compute_confluence", "above_threshold"]
