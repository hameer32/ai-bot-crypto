#!/usr/bin/env python3
"""
ICT ML Trainer — Institutional Edition
=======================================
1. Loads cached OHLCV for all available symbols
2. Collects labeled signal dataset across ALL symbols (for training diversity)
3. Trains LightGBM with Optuna + TimeSeriesCV + Platt calibration
4. Reports: dataset stats, per-symbol WR, feature importances, threshold sweep
5. Compares rules-only vs ML-gated backtest on TRADING symbols (BTC default)
6. Saves model to models/ict_predictor.pkl

Usage:
  python scripts/run_ml_train.py
  python scripts/run_ml_train.py --threshold 0.52
  python scripts/run_ml_train.py --training-symbols BTCUSDT,ETHUSDT,SOLUSDT
  python scripts/run_ml_train.py --no-save --trials 20
"""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from config.config import load_config
from strategies.registry import get_strategy
from scripts.run_strategy import load_cache, run_backtest, compute_metrics, print_metrics
from ml.feature_builder import collect_signal_dataset, FEATURE_COLS
from ml.supervised import ICTTradePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold",        type=float, default=None)
    parser.add_argument("--training-symbols", default=None,
                        help="Comma-separated symbols for training (default: all available)")
    parser.add_argument("--no-save",   action="store_true")
    parser.add_argument("--trials",    type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--model-path",  default=None,
                        help="Override model save path (default: models/ict_predictor.pkl)")
    parser.add_argument("--ltf",         default=None,
                        help="Override LTF execution timeframe (e.g. 1h for Indian/commodity)")
    parser.add_argument("--segment",     default=None,
                        help="Segment label printed in output")
    parser.add_argument("--plan-key",    default=None,
                        help="Config block to use, e.g. plan_c_ict_crypto (overrides --ltf/--model-path)")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    for key in ("plan_a_swing", "plan_b_scalp", "plan_c_ict"):
        if key in raw:
            setattr(cfg, key, raw[key])

    # ── Apply CLI overrides to plan_c_ict config ─────────────────────────────
    # --plan-key loads a segment-specific config block from config.yaml
    if args.plan_key and args.plan_key in raw:
        seg_cfg = dict(raw[args.plan_key])
        # Merge segment config onto plan_c_ict (segment overrides win)
        base = dict(raw.get("plan_c_ict", {}))
        base.update(seg_cfg)
        setattr(cfg, "plan_c_ict", base)
        raw["plan_c_ict"] = base
        # Extract model path and segment from plan-key config
        if not args.model_path and "model_path" in seg_cfg:
            args.model_path = seg_cfg["model_path"]
        if not args.segment:
            args.segment = args.plan_key.replace("plan_c_ict_", "")
    elif args.ltf:
        cfg.plan_c_ict["ltf"] = args.ltf
        raw.setdefault("plan_c_ict", {})["ltf"] = args.ltf
    if args.segment:
        print(f"\n  ═══ SEGMENT: {args.segment.upper()} ═══")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    all_data = load_cache()
    print(f"\n  Loaded {len(all_data)} cached datasets")
    available_symbols = sorted({s for s, _ in all_data.keys()})
    print(f"  Available symbols: {available_symbols}")

    # ── 2. Determine training symbols ─────────────────────────────────────────
    ltf_name = cfg.plan_c_ict.get("ltf", "15m") if isinstance(cfg.plan_c_ict, dict) else raw.get("plan_c_ict", {}).get("ltf", "15m")

    cfg_train_syms = raw.get("plan_c_ict", {}).get("training_symbols", [])

    if args.training_symbols:
        train_syms = [s.strip() for s in args.training_symbols.split(",")]
    elif cfg_train_syms:
        train_syms = list(cfg_train_syms)
    else:
        # Auto-detect: all cached symbols with full ICT stack (15m confirmation + 5m execution)
        required_tfs = {"1w", "1d", "4h", "1h", "15m", "5m"}
        train_syms = [
            s for s in available_symbols
            if required_tfs.issubset({tf for sym, tf in all_data.keys() if sym == s})
        ]
    print(f"\n  Training symbols ({len(train_syms)}): {train_syms}")

    # ── 3. Collect signal dataset ─────────────────────────────────────────────
    print("\n  Building ICT signal dataset across all training symbols...")
    strategy = get_strategy("plan_c_ict", cfg)
    dataset  = collect_signal_dataset(
        strategy, all_data, cfg,
        ltf_name=ltf_name,
        training_symbols=train_syms,
    )

    if len(dataset) == 0:
        print("  ERROR: No signals found. Ensure cache data exists for training symbols.")
        return

    # ── 4. Dataset summary ────────────────────────────────────────────────────
    _print_dataset_summary(dataset, ltf_name)

    if len(dataset) < 50:
        print("\n  WARNING: <50 samples — ML results unreliable. Fetch more symbols/data.")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"\n  ── Training LightGBM + Optuna ──────────────────────────────")
    predictor = ICTTradePredictor()
    metrics   = predictor.train(dataset, n_optuna_trials=args.trials)

    # ── 6. Feature importances ────────────────────────────────────────────────
    print(f"\n  ── Feature Importances ─────────────────────────────────────")
    fi = predictor.feature_importances()
    for feat, imp in fi.items():
        if imp > 0:
            bar = "█" * max(1, int(imp * 50))
            print(f"    {feat:22s} {imp:.4f}  {bar}")

    # ── 7. Threshold sweep ────────────────────────────────────────────────────
    print(f"\n  ── Threshold Sweep ─────────────────────────────────────────")
    print(f"  {'Threshold':>10}  {'N':>6}  {'WR%':>6}  {'Kept%':>6}")
    X_all  = dataset[FEATURE_COLS].fillna(0).astype(float)
    probas = predictor.model.predict_proba(X_all)[:, 1]
    dataset = dataset.copy()
    dataset["ml_proba"] = probas

    sweep_thrs = np.arange(0.35, 0.75, 0.05)
    if args.threshold:
        sweep_thrs = sorted(set(list(sweep_thrs) + [args.threshold]))

    best_thr = args.threshold or 0.50
    for thr in sweep_thrs:
        kept = dataset[dataset["ml_proba"] >= thr]
        if len(kept) == 0:
            print(f"  {thr:>10.2f}  {'0':>6}  {'—':>6}  {'0%':>6}")
        else:
            print(f"  {thr:>10.2f}  {len(kept):>6}  {kept['label'].mean():>5.1%}  "
                  f"{len(kept)/len(dataset):>5.0%}")

    # ── 8. Per-symbol ML probability analysis ─────────────────────────────────
    print(f"\n  ── Per-Symbol Analysis (ML proba ≥ {best_thr}) ─────────────")
    print(f"  {'Symbol':<14} {'Total':>6} {'WR_raw':>7} {'WR_ml':>7} {'Kept%':>6}")
    for sym in sorted(dataset["sym"].unique()):
        sub = dataset[dataset["sym"] == sym]
        ml_sub = sub[sub["ml_proba"] >= best_thr]
        wr_raw = sub["label"].mean()
        wr_ml  = ml_sub["label"].mean() if len(ml_sub) else float("nan")
        kept   = len(ml_sub) / len(sub) if len(sub) else 0
        print(f"  {sym:<14} {len(sub):>6} {wr_raw:>6.1%}  {wr_ml:>6.1%}  {kept:>5.0%}")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        model_path = args.model_path or "models/ict_predictor.pkl"
        predictor.save(model_path)
        os.makedirs("results", exist_ok=True)
        seg_suffix = f"_{args.segment}" if args.segment else ""
        csv_path = f"results/ml_signal_dataset{seg_suffix}.csv"
        dataset.to_csv(csv_path, index=False)
        print(f"\n  Dataset saved → {csv_path}")

    # ── 10. Backtest comparison on trading symbols ────────────────────────────
    trade_syms_cfg = raw.get("plan_c_ict", {}).get("symbols", cfg.symbols)
    trade_syms     = list(trade_syms_cfg) if trade_syms_cfg else cfg.symbols
    print(f"\n  ── Backtest Comparison (trading symbols: {trade_syms}) ──────")
    _run_comparison(cfg, raw, all_data, ltf_name, predictor, best_thr, trade_syms)


# ── Dataset summary ───────────────────────────────────────────────────────────

def _print_dataset_summary(dataset: pd.DataFrame, ltf_name: str):
    print(f"\n  ── Dataset Summary ─────────────────────────────────────────")
    print(f"  Total signals    : {len(dataset)}")
    print(f"  Win rate         : {dataset['label'].mean():.1%}")
    print(f"  Date range       : {dataset['ts'].min().date()} → {dataset['ts'].max().date()}")
    print(f"  Symbols          : {sorted(dataset['sym'].unique())}")

    print(f"\n  By direction:")
    for d_val, d_name in [(1, "long"), (0, "short")]:
        sub = dataset[dataset["direction"] == d_val]
        if len(sub):
            print(f"    {d_name:5s}: {len(sub):4d} signals  WR={sub['label'].mean():.1%}")

    print(f"\n  By session:")
    sess_map = {0: "london", 1: "ny_am", 2: "ny_pm", -1: "none", -2: "asia"}
    for code, name in sess_map.items():
        sub = dataset[dataset["session"] == code]
        if len(sub):
            print(f"    {name:8s}: {len(sub):4d} signals  WR={sub['label'].mean():.1%}")

    print(f"\n  By OB type:")
    for flag, label in [("t2_in_4h_ob", "4H OB"), ("t2_in_1h_ob", "1H OB")]:
        sub = dataset[dataset[flag] == 1]
        if len(sub):
            print(f"    {label}: {len(sub):4d} signals  WR={sub['label'].mean():.1%}")


# ── Backtest comparison ───────────────────────────────────────────────────────

def _run_comparison(cfg, raw, all_data, ltf_name, predictor, ml_threshold, symbols):
    from strategies.plan_c_ict.strategy import PlanCICTStrategy
    from ml.feature_builder import extract_features

    class MLGatedStrategy(PlanCICTStrategy):
        """Rules strategy with ML filter injected — overrides _apply_ml_filter."""
        def __init__(self, base_cfg, ml_model, ml_thr):
            super().__init__(base_cfg)
            self._ml_model     = ml_model
            self._ml_threshold = ml_thr  # used by parent's _apply_ml_filter

    # Rules-only: disable any ML model that may have been loaded from disk
    strat_base = get_strategy("plan_c_ict", cfg)
    strat_base._ml_model     = None
    strat_base._ml_threshold = 0.0
    trades_b, equity_b, _ = run_backtest(
        strat_base, all_data, symbols, ltf_name,
        capital=10_000.0, risk_pct=0.01,
        max_dd=cfg.risk.max_drawdown_pct,
    )
    m_base = compute_metrics(trades_b, equity_b, 10_000.0)

    # ML-gated
    strat_ml = MLGatedStrategy(cfg, predictor, ml_threshold)
    trades_m, equity_m, _ = run_backtest(
        strat_ml, all_data, symbols, ltf_name,
        capital=10_000.0, risk_pct=0.01,
        max_dd=cfg.risk.max_drawdown_pct,
    )
    m_ml = compute_metrics(trades_m, equity_m, 10_000.0)

    print(f"\n  {'Strategy':<30} {'Trades':>7} {'WinR':>6} {'AvgRR':>7} "
          f"{'Expect':>8} {'MaxDD':>7} {'Return':>8}")
    print("  " + "─" * 74)
    for name, m in [("plan_c_ict (rules only)", m_base),
                    (f"plan_c_ict + ML≥{ml_threshold:.2f}", m_ml)]:
        print(f"  {name:<30} {m['trades']:>7} {m['win_rate']:>5.0%} "
              f"{m['avg_rr']:>+7.2f} {m['expectancy']:>+8.1f} "
              f"{m['max_dd']:>6.1%} {m['ret_pct']:>+7.1f}%")


if __name__ == "__main__":
    main()
