#!/usr/bin/env python3
"""
Plan D — ICT + Ichimoku + Indicators Comparison
================================================
Trains Plan D models and compares against Plan C.

Output table for each asset class:
  Plan C rules-only   WR  Return  MaxDD
  Plan C + ML         WR  Return  MaxDD
  Plan D rules-only   WR  Return  MaxDD   ← new
  Plan D + ML         WR  Return  MaxDD   ← new

Plan D models saved to: models/plan_d_{segment}.pkl
Comparison saved to:    results/plan_d_comparison.csv

Usage:
  python scripts/run_plan_d.py
  python scripts/run_plan_d.py --segment crypto
  python scripts/run_plan_d.py --no-train   # backtest only with existing models
"""
import sys, os, argparse, warnings, yaml
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

SEGMENTS = {
    "crypto": {
        "symbols":      ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
                         "DOGEUSDT","SUIUSDT","NEARUSDT","ONDOUSDT","ZECUSDT"],
        "ltf":          "5m",
        "plan_c_key":   "plan_c_ict",
        "plan_c_model": "models/ict_predictor.pkl",
        "plan_c_thr":   0.42,
        "plan_d_model": "models/plan_d_crypto.pkl",
        "plan_d_thr":   0.40,
        "train_syms":   None,   # auto-detect
    },
    "commodity": {
        "symbols":      ["GOLD","SILVER","OIL","NATGAS","COPPER","PLATINUM"],
        "ltf":          "1h",
        "plan_c_key":   "plan_c_ict_commodity",
        "plan_c_model": "models/ict_predictor_commodity.pkl",
        "plan_c_thr":   0.30,
        "plan_d_model": "models/plan_d_commodity.pkl",
        "plan_d_thr":   0.35,
        "train_syms":   ["GOLD","SILVER","OIL","NATGAS","COPPER","PLATINUM",
                         "PALLADIUM","COCOA","COFFEE","WHEAT"],
    },
    "forex": {
        "symbols":      ["EURUSD","GBPUSD","USDJPY","EURJPY","GBPJPY","AUDUSD","USDCAD","EURGBP"],
        "ltf":          "5m",
        "plan_c_key":   "plan_c_ict_forex",
        "plan_c_model": "models/ict_predictor_forex.pkl",
        "plan_c_thr":   0.25,
        "plan_d_model": "models/plan_d_forex.pkl",
        "plan_d_thr":   0.35,
        "train_syms":   ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
                         "EURGBP","EURJPY","GBPJPY","EURAUD","EURCAD","GBPAUD","GBPCAD",
                         "AUDCAD","AUDNZD","NZDJPY","CADJPY","CHFJPY","AUDCHF"],
    },
    "india_equity": {
        "symbols":      ["RELIANCE","HDFCBANK","TCS","INFY","ICICIBANK",
                         "SBIN","KOTAKBANK","LT","SUNPHARMA","HINDUNILVR",
                         "AXISBANK","WIPRO","ITC","NTPC","ONGC"],
        "ltf":          "5m",
        "plan_c_key":   "plan_c_ict_india_equity",
        "plan_c_model": "models/ict_predictor_india_equity.pkl",
        "plan_c_thr":   0.50,
        "plan_d_model": "models/plan_d_india_equity.pkl",
        "plan_d_thr":   0.40,
        "train_syms":   None,   # all NSE stocks
    },
}


def train_plan_d(seg_name, seg, all_data, cfg, raw, trials=50):
    """Train Plan D ML model for one segment."""
    from strategies.plan_d_hybrid.strategy import PlanDHybridStrategy
    from ml.feature_builder import collect_signal_dataset, FEATURE_COLS
    from ml.supervised import ICTTradePredictor

    # Build plan_d config
    plan_d_cfg = {
        "ltf":        seg["ltf"],
        "symbols":    seg["train_syms"] or seg["symbols"],
        "market":     raw.get(seg["plan_c_key"], {}).get("market", "global"),
        "model_path": seg["plan_d_model"],
        "ml_threshold": seg["plan_d_thr"],
        # Relaxed ICT parameters for more signals
        "impulse_mult": 1.0,
        "confidence_threshold": 0.30,
        "min_fvg_atr": 0.15,
        "choch_lookback": 25,
        "sweep_lookback": 10,
        "zone_gate": "all_zones",
    }
    setattr(cfg, "plan_d_hybrid", plan_d_cfg)

    # CRITICAL: collect_signal_dataset reads from cfg.plan_c_ict for threshold/zone_gate
    # Temporarily override plan_c_ict with plan_d settings so the collector uses them
    orig_plan_c = getattr(cfg, "plan_c_ict", {})
    plan_c_override = dict(raw.get("plan_c_ict", {}))
    plan_c_override.update({
        "confidence_threshold": plan_d_cfg["confidence_threshold"],
        "impulse_mult":         plan_d_cfg["impulse_mult"],
        "min_fvg_atr":          plan_d_cfg["min_fvg_atr"],
        "choch_lookback":       plan_d_cfg["choch_lookback"],
        "sweep_lookback":       plan_d_cfg["sweep_lookback"],
        "zone_gate":            plan_d_cfg.get("zone_gate", "all_zones"),
        "ltf":                  plan_d_cfg["ltf"],
        "market":               plan_d_cfg.get("market", "global"),
    })
    setattr(cfg, "plan_c_ict", plan_c_override)

    ltf_name = seg["ltf"]
    train_syms = seg["train_syms"] or seg["symbols"]

    strategy = PlanDHybridStrategy(cfg)
    dataset  = collect_signal_dataset(
        strategy, all_data, cfg,
        ltf_name=ltf_name,
        training_symbols=train_syms,
    )

    # Restore original plan_c_ict config
    setattr(cfg, "plan_c_ict", orig_plan_c)

    if len(dataset) < 20:
        print(f"  [PlanD-{seg_name}] Too few signals ({len(dataset)}) — skip training")
        return None

    n = len(dataset)
    wr = dataset["label"].mean()
    print(f"  [PlanD-{seg_name}] Dataset: {n} signals, {wr:.1%} WR")

    # Add Plan D specific features to the dataset so trainer picks them up
    plan_d_extra = ["ichi_bull_score","ichi_bear_score","rsi_div","macd_hist_norm","bb_pct"]
    # Inject any extra columns that exist into FEATURE_COLS via a temporary override
    import ml.feature_builder as fb
    orig_cols = list(fb.FEATURE_COLS)
    extra = [c for c in plan_d_extra if c in dataset.columns and c not in fb.FEATURE_COLS]
    if extra:
        fb.FEATURE_COLS = orig_cols + extra

    # Train
    predictor = ICTTradePredictor()
    predictor.train(dataset, n_optuna_trials=trials)

    # Restore original FEATURE_COLS
    fb.FEATURE_COLS = orig_cols

    os.makedirs("models", exist_ok=True)
    predictor.save(seg["plan_d_model"])
    print(f"  [PlanD-{seg_name}] Model saved → {seg['plan_d_model']}")

    # Threshold sweep
    X = dataset[[c for c in orig_cols + extra if c in dataset.columns]].fillna(0).astype(float)
    probas = predictor.model.predict_proba(X)[:, 1]
    dataset["ml_proba"] = probas
    print(f"  Prob range: [{probas.min():.3f}, {probas.max():.3f}]")
    for thr in [0.30, 0.35, 0.38, 0.40, 0.45]:
        k = dataset[dataset["ml_proba"] >= thr]
        if len(k) > 0:
            print(f"    {thr:.2f}: {len(k):4d} signals  WR={k['label'].mean():.1%}")

    return predictor


def backtest_plan(strategy, all_data, symbols, ltf, capital=10000, max_dd=0.10):
    """Run backtest and return metrics."""
    from scripts.run_strategy import run_backtest, compute_metrics
    try:
        trades, equity, _ = run_backtest(strategy, all_data, symbols, ltf,
                                          capital=capital, risk_pct=0.01, max_dd=max_dd)
        return compute_metrics(trades, equity, capital)
    except Exception as e:
        print(f"  Backtest error: {e}")
        return {"trades":0,"win_rate":0,"avg_rr":0,"max_dd":0,"ret_pct":0}


def run_comparison(seg_name, seg, all_data, cfg, raw):
    """Run Plan C vs Plan D comparison for one segment."""
    from strategies.registry import get_strategy
    from strategies.plan_d_hybrid.strategy import PlanDHybridStrategy
    from ml.supervised import ICTTradePredictor

    print(f"\n{'═'*70}")
    print(f"  {seg_name.upper()} SEGMENT — Plan C vs Plan D")
    print(f"{'═'*70}")
    print(f"  Symbols: {seg['symbols']}")

    results = {}

    # ── Plan C: Rules-only ────────────────────────────────────────────────────
    plan_c_cfg = dict(raw.get(seg["plan_c_key"], raw.get("plan_c_ict", {})))
    base_c = dict(raw.get("plan_c_ict", {}))
    base_c.update(plan_c_cfg)
    base_c["symbols"] = seg["symbols"]
    base_c["training_symbols"] = []
    setattr(cfg, "plan_c_ict", base_c)

    strat_c_r = get_strategy("plan_c_ict", cfg)
    strat_c_r._ml_model = None; strat_c_r._ml_threshold = 0.0
    m = backtest_plan(strat_c_r, all_data, seg["symbols"], seg["ltf"],
                      max_dd=cfg.risk.max_drawdown_pct)
    results["plan_c_rules"] = m
    print(f"\n  Plan C (rules)    : {m['trades']:4d} trades  {m['win_rate']:>5.0%} WR  {m['avg_rr']:>+6.2f} AvgRR  {m['max_dd']:>5.1%} DD  {m['ret_pct']:>+7.1f}%")

    # ── Plan C: ML-gated ─────────────────────────────────────────────────────
    pred_c = ICTTradePredictor()
    if pred_c.load(seg["plan_c_model"]):
        strat_c_ml = get_strategy("plan_c_ict", cfg)
        strat_c_ml._ml_model = pred_c
        strat_c_ml._ml_threshold = seg["plan_c_thr"]
        m = backtest_plan(strat_c_ml, all_data, seg["symbols"], seg["ltf"],
                          max_dd=cfg.risk.max_drawdown_pct)
        results["plan_c_ml"] = m
        print(f"  Plan C + ML≥{seg['plan_c_thr']:.2f} : {m['trades']:4d} trades  {m['win_rate']:>5.0%} WR  {m['avg_rr']:>+6.2f} AvgRR  {m['max_dd']:>5.1%} DD  {m['ret_pct']:>+7.1f}%")

    # ── Plan D: Rules-only ────────────────────────────────────────────────────
    plan_d_cfg = {
        "ltf": seg["ltf"], "symbols": seg["symbols"], "training_symbols": [],
        "market": raw.get(seg["plan_c_key"], {}).get("market", "global"),
        "model_path": seg["plan_d_model"],
        "impulse_mult": 1.0, "confidence_threshold": 0.30,
        "min_fvg_atr": 0.15, "choch_lookback": 25, "sweep_lookback": 10,
    }
    setattr(cfg, "plan_d_hybrid", plan_d_cfg)

    strat_d_r = PlanDHybridStrategy(cfg)
    strat_d_r._ml_model = None; strat_d_r._ml_threshold = 0.0
    m = backtest_plan(strat_d_r, all_data, seg["symbols"], seg["ltf"],
                      max_dd=cfg.risk.max_drawdown_pct)
    results["plan_d_rules"] = m
    print(f"  Plan D (rules)    : {m['trades']:4d} trades  {m['win_rate']:>5.0%} WR  {m['avg_rr']:>+6.2f} AvgRR  {m['max_dd']:>5.1%} DD  {m['ret_pct']:>+7.1f}%")

    # ── Plan D: ML-gated ─────────────────────────────────────────────────────
    pred_d = ICTTradePredictor()
    if pred_d.load(seg["plan_d_model"]):
        strat_d_ml = PlanDHybridStrategy(cfg)
        strat_d_ml._ml_model = pred_d
        strat_d_ml._ml_threshold = seg["plan_d_thr"]
        m = backtest_plan(strat_d_ml, all_data, seg["symbols"], seg["ltf"],
                          max_dd=cfg.risk.max_drawdown_pct)
        results["plan_d_ml"] = m
        winner = "⭐ Plan D" if m["ret_pct"] > results.get("plan_c_ml",{}).get("ret_pct",-999) else "   Plan C"
        print(f"  Plan D + ML≥{seg['plan_d_thr']:.2f} : {m['trades']:4d} trades  {m['win_rate']:>5.0%} WR  {m['avg_rr']:>+6.2f} AvgRR  {m['max_dd']:>5.1%} DD  {m['ret_pct']:>+7.1f}%  {winner}")
    else:
        print(f"  Plan D + ML       : model not found ({seg['plan_d_model']})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment",   default=None, choices=list(SEGMENTS.keys()))
    parser.add_argument("--trials",    type=int, default=50)
    parser.add_argument("--no-train",  action="store_true", help="Skip training, compare existing models")
    args = parser.parse_args()

    from config.config import load_config
    from scripts.run_strategy import load_cache

    cfg = load_config("config.yaml")
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    for key in ("plan_a_swing","plan_b_scalp","plan_c_ict"):
        if key in raw:
            setattr(cfg, key, raw[key])

    all_data = load_cache()
    print(f"Loaded {len(all_data)} datasets\n")

    segs = {args.segment: SEGMENTS[args.segment]} if args.segment else SEGMENTS

    all_results = {}
    for seg_name, seg in segs.items():
        # Train Plan D
        if not args.no_train:
            print(f"\n▶ Training Plan D for {seg_name.upper()}...")
            train_plan_d(seg_name, seg, all_data, cfg, raw, args.trials)

        # Compare
        results = run_comparison(seg_name, seg, all_data, cfg, raw)
        all_results[seg_name] = results

    # Final summary
    print(f"\n{'═'*70}")
    print(f"  FINAL COMPARISON SUMMARY")
    print(f"{'═'*70}")
    print(f"\n  {'Segment':<16} {'Plan C Rules':>13} {'Plan C+ML':>10} {'Plan D Rules':>13} {'Plan D+ML':>10} {'Winner'}")
    print("  " + "─"*75)
    for seg_name, res in all_results.items():
        c_r  = res.get("plan_c_rules",{}).get("ret_pct", 0)
        c_ml = res.get("plan_c_ml",   {}).get("ret_pct", 0)
        d_r  = res.get("plan_d_rules",{}).get("ret_pct", 0)
        d_ml = res.get("plan_d_ml",   {}).get("ret_pct", 0)
        best = max(c_r, c_ml, d_r, d_ml)
        winner = "C-rules" if best==c_r else ("C+ML" if best==c_ml else ("D-rules" if best==d_r else "D+ML"))
        print(f"  {seg_name:<16} {c_r:>+12.1f}% {c_ml:>+9.1f}% {d_r:>+12.1f}% {d_ml:>+9.1f}%  {winner}")

    # Save to CSV
    rows = []
    for seg_name, res in all_results.items():
        for model_name, m in res.items():
            rows.append({"segment": seg_name, "model": model_name, **m})
    pd.DataFrame(rows).to_csv("results/plan_d_comparison.csv", index=False)
    print(f"\n  Saved → results/plan_d_comparison.csv")


if __name__ == "__main__":
    main()
