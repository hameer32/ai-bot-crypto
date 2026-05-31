from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class DataConfig:
    cache_dir: str = "cache/"
    lookback_days: int = 730
    outlier_z_threshold: float = 5.0


@dataclass
class RulesConfig:
    enabled: bool = True
    weight: float = 0.40
    n_period: int = 30
    atr_period: int = 14
    atr_k: float = 1.0
    vol_window: int = 20
    vol_pct_thresh: float = 80.0
    ema_period: int = 50
    body_ratio_thresh: float = 0.4
    confidence_threshold: float = 0.7


@dataclass
class MeanReversionConfig:
    enabled: bool = True
    zscore_window: int = 20
    zscore_entry_thresh: float = 2.0
    zscore_exit_thresh: float = 0.5
    bb_period: int = 20
    bb_std: float = 2.0


@dataclass
class RegimeConfig:
    enabled: bool = True
    n_states: int = 2
    lookback: int = 200


@dataclass
class MicrostructureConfig:
    enabled: bool = True
    kyle_window: int = 20
    amihud_window: int = 20


@dataclass
class MomentumConfig:
    enabled: bool = True
    tsm_lookback: int = 120
    tsm_short_window: int = 20


@dataclass
class CorrelationConfig:
    enabled: bool = True
    corr_window: int = 60
    corr_filter_thresh: float = 0.7


@dataclass
class QuantConfig:
    enabled: bool = True
    weight: float = 0.35
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)


@dataclass
class SupervisedConfig:
    enabled: bool = False
    model: str = "xgboost"
    retrain_days: int = 90
    min_train_samples: int = 200


@dataclass
class RLConfig:
    enabled: bool = False
    algorithm: str = "ppo"
    timesteps: int = 500000
    retrain_days: int = 30


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "claude-sonnet-4-6"
    report_on: str = "backtest_end"


@dataclass
class MLConfig:
    enabled: bool = False
    weight: float = 0.25
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class FusionConfig:
    method: str = "weighted_average"
    final_threshold: float = 0.65


@dataclass
class RiskConfig:
    initial_capital: float = 10000.0
    risk_pct_per_trade: float = 0.01
    max_drawdown_pct: float = 0.10
    partial_exit_rr: float = 2.0
    trail_atr_mult: float = 1.0


@dataclass
class AppConfig:
    exchange: str = "binance"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    ltf: str = "15m"
    htf: str = "4h"
    data: DataConfig = field(default_factory=DataConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def _dict_to_dataclass(cls, d):
    if not isinstance(d, dict):
        return d
    kwargs = {}
    for f in cls.__dataclass_fields__:
        if f not in d:
            continue
        field_type = cls.__dataclass_fields__[f].type
        val = d[f]
        try:
            actual_type = eval(field_type) if isinstance(field_type, str) else field_type
            if hasattr(actual_type, "__dataclass_fields__"):
                val = _dict_to_dataclass(actual_type, val)
        except Exception:
            pass
        kwargs[f] = val
    return cls(**kwargs)


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig(
        exchange=raw.get("exchange", "binance"),
        symbols=raw.get("symbols", ["BTCUSDT", "ETHUSDT"]),
        ltf=raw.get("ltf", "15m"),
        htf=raw.get("htf", "4h"),
        data=_dict_to_dataclass(DataConfig, raw.get("data", {})),
        rules=_dict_to_dataclass(RulesConfig, raw.get("rules", {})),
        quant=_build_quant_config(raw.get("quant", {})),
        ml=_build_ml_config(raw.get("ml", {})),
        fusion=_dict_to_dataclass(FusionConfig, raw.get("fusion", {})),
        risk=_dict_to_dataclass(RiskConfig, raw.get("risk", {})),
    )
    return cfg


def _build_quant_config(d: dict) -> QuantConfig:
    return QuantConfig(
        enabled=d.get("enabled", True),
        weight=d.get("weight", 0.35),
        mean_reversion=_dict_to_dataclass(MeanReversionConfig, d.get("mean_reversion", {})),
        regime=_dict_to_dataclass(RegimeConfig, d.get("regime", {})),
        microstructure=_dict_to_dataclass(MicrostructureConfig, d.get("microstructure", {})),
        momentum=_dict_to_dataclass(MomentumConfig, d.get("momentum", {})),
        correlation=_dict_to_dataclass(CorrelationConfig, d.get("correlation", {})),
    )


def _build_ml_config(d: dict) -> MLConfig:
    return MLConfig(
        enabled=d.get("enabled", False),
        weight=d.get("weight", 0.25),
        supervised=_dict_to_dataclass(SupervisedConfig, d.get("supervised", {})),
        rl=_dict_to_dataclass(RLConfig, d.get("rl", {})),
        llm=_dict_to_dataclass(LLMConfig, d.get("llm", {})),
    )
