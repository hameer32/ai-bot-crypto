"""
Strategy registry — maps config name → strategy class.
Add a new strategy here after creating its folder.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.base import BaseStrategy

_REGISTRY: dict[str, str] = {
    "plan_a_swing":  "strategies.plan_a_swing.strategy.PlanASwingStrategy",
    "plan_b_scalp":  "strategies.plan_b_scalp.strategy.PlanBScalpStrategy",
    "original_smc":  "strategies.original_smc.strategy.OriginalSMCStrategy",
    "plan_c_ict":    "strategies.plan_c_ict.strategy.PlanCICTStrategy",
    "plan_d_hybrid": "strategies.plan_d_hybrid.strategy.PlanDHybridStrategy",
}


def get_strategy(name: str, cfg) -> "BaseStrategy":
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(cfg)


def list_strategies() -> list[str]:
    return list(_REGISTRY)
