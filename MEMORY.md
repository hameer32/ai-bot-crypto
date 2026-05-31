# Project Memory — AI Crypto Trading Bot

## Architecture Overview

Three independent, feature-flagged signal layers fused into a single trade decision:

```
Rules Layer (SMC signals)  →  P_rules ∈ [0,1]
Quant Layer (stat models)  →  P_quant ∈ [0,1]   ← enabled via config
ML Layer    (XGB/RL/LLM)   →  P_ml    ∈ [0,1]   ← enabled via config
                                    ↓
                           Fusion → final_confidence
                                    ↓
                           Risk → TradeSetup → Backtest / Paper
```

All config in `config.yaml`. Each layer has `enabled: bool` and `weight: float`.

---

## Volume Validation (New)

Signals (OB, FVG, Liquidity) are validated using **Relative Volume (RVOL)** and **Volume Spikes**. 
- **Institutional Conviction**: RVOL > 2.0 on impulse bars increases signal weight.
- **Absorption Detection**: High volume spikes at liquidity sweeps indicate traps.

---

## Strategy Framework (strategies/)

**Design principle**: every strategy lives in its own folder, shares the same
`BaseStrategy` interface, and plugs into the universal backtest runner without
touching the engine code. Add a new strategy = new folder + register in `registry.py`.

```
strategies/
├── base.py              # BaseStrategy ABC + TradeSetup dataclass
├── registry.py          # name → class mapping
├── signals/             # shared primitives (used by all strategies)
│   ├── order_blocks.py  # OB detection + nearest_ob()
│   ├── fvg.py           # FVG detection + fvg_tp_target()
│   ├── liquidity_zones.py # BSL/SSL detection + liquidity_tp()
│   └── structure.py     # htf_structure(), htf_trend(), align_htf_series()
├── plan_a_swing/        # Swing: Weekly+Daily → 4H+1H → 15m+5m
│   ├── tier1_bias.py    # Weekly+Daily bias, S/R levels
│   ├── tier2_zones.py   # 4H+1H OBs, FVGs, liquidity pools
│   ├── tier3_entry.py   # 15m entry: sweep+trap+absorption, SL/TP
│   ├── confluence.py    # T1(30%) + T2(35%) + T3(35%) → conf score
│   └── strategy.py      # PlanASwingStrategy (orchestrator)
├── plan_b_scalp/        # Scalp: 4H+1H → 30m+15m → 5m+1m
│   ├── tier1_bias.py    # 4H+1H bias
│   ├── tier2_zones.py   # 30m+15m OBs, FVGs, liquidity pools
│   ├── tier3_entry.py   # 5m entry (TP1=1.5R for scalp)
│   ├── confluence.py    # T1(20%) + T2(50%) + T3(30%)
│   └── strategy.py      # PlanBScalpStrategy
└── original_smc/        # Original system (baseline reference)
    └── strategy.py      # Wraps download_and_run pipeline
```

### Adding a new strategy

1. `mkdir strategies/my_strategy && touch strategies/my_strategy/__init__.py`
2. Create `strategies/my_strategy/strategy.py` implementing `BaseStrategy`
3. Add `"my_strategy": "strategies.my_strategy.strategy.MyStrategy"` to `strategies/registry.py`
4. Add config block to `config.yaml`
5. Set `active_strategy: my_strategy` in `config.yaml`

---

## MTF Cascade Logic

### Plan A — Swing
| Tier | Timeframes | Purpose |
|------|-----------|---------|
| 1 | Weekly + Daily | Macro trend bias, key S/R |
| 2 | 4H + 1H | Order blocks, FVGs, liquidity pools |
| 3 | 15m + 5m | Precision entry: sweep → trap/absorption |

### Plan B — Scalp
| Tier | Timeframes | Purpose |
|------|-----------|---------|
| 1 | 4H + 1H | Intraday bias |
| 2 | 30m + 15m | Intraday zones |
| 3 | 5m + 1m | Scalp entry |

**Entry logic (both plans)**:
- Tier 1 must give bullish/bearish (not neutral)
- Tier 2: price must be within `proximity_atr × ATR` of an active OB, FVG, or liquidity zone
- Tier 3: liquidity sweep within last `sweep_lookback` bars + trap OR order-flow trigger

**SL**: `sweep_level ± atr_k × ATR`
**TP1**: 2R (Plan A) / 1.5R (Plan B) → 50% exit
**TP2**: nearest HTF FVG fill or liquidity pool target (if > 3R, else 3R)

---

## Key Signal Definitions

**Order Block (OB)**: Last opposing candle before a strong impulse move
(impulse ≥ `ob_impulse_mult × ATR`, closes beyond prior bar's high/low).
Mitigated when price closes through it. **Validated by RVOL spike on impulse.**

**Fair Value Gap (FVG)**: 3-candle imbalance: `candle[i-2].high < candle[i].low` (bull)
or `candle[i-2].low > candle[i].high` (bear). Size ≥ `fvg_min_atr × ATR`.
Acts as entry zone (retracement) and TP target (gap fill from above).
**High RVOL on gap candle increases conviction.**

**Liquidity Zone**: Cluster of ≥2 swing highs (BSL = buy-side) or lows (SSL = sell-side)
within `cluster_atr × ATR`. Price hunting these → reversal setup.

---

## Running Strategies

```bash
# Run active strategy from config
python scripts/run_strategy.py

# Run specific strategy
python scripts/run_strategy.py --strategy plan_a_swing
python scripts/run_strategy.py --strategy plan_b_scalp

# Compare all strategies side-by-side
python scripts/run_strategy.py --compare

# List available strategies
python scripts/run_strategy.py --list
```

---

## Layer Comparison Results (2024 backtest)

Best configs per timeframe:
- **5m**: ML only — Expectancy +86, WinRate 53%
- **15m**: Rules + Quant — Expectancy +39, WinRate 46%, Return +21.8%
- **30m**: Rules + Quant — Expectancy +85, WinRate 52%, Return +19.6%
- **1h**: Quant only — Expectancy +60, WinRate 56%

Layer impact:
- Rules: +3.5 avg delta → helps
- ML: +14.2 avg delta → helps significantly
- Quant standalone: -5.7 (but better in combo with Rules)
- Triple-layer avg expectancy (+21.6) > dual (+8.7) > single (+5.4)

---

## Files Reference

| File | Purpose |
|------|---------|
| `config.yaml` | All config — single source of truth |
| `config/config.py` | Dataclass loader |
| `strategies/` | All strategies (isolated) |
| `scripts/run_strategy.py` | Universal backtest runner |
| `scripts/layer_comparison.py` | 7-combo × 8-TF comparison |
| `optimization/grid_search.py` | Parameter grid search |
| `scripts/download_and_run.py` | Data download + original backtest |
| `results/layer_comparison.csv` | Layer comparison output |
| `results/grid_search_results.csv` | Grid search output |
| `cache/` | Parquet data cache |
