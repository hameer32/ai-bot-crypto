#!/usr/bin/env python3
"""
Segment Model Grid Search
==========================
Trains multiple model variants per segment with different collection thresholds,
then tests each at multiple inference thresholds.
Creates a separate .pkl file for every (segment, collection_threshold) combination
so you can compare which configuration wins per asset class.

Outputs:
  models/ict_predictor_{segment}_col{X}.pkl   — one model per collection threshold
  results/segment_grid_results.csv             — full comparison table

Usage:
  python scripts/run_segment_grid.py
  python scripts/run_segment_grid.py --segment commodity
"""
import sys, os, argparse, yaml, warnings, subprocess
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ── Segments to optimise ────────────────────────────────────────────────────────
SEGMENTS = {
    "commodity": {
        "plan_key":      "plan_c_ict_commodity",
        "symbols":       ["GOLD","SILVER","OIL","NATGAS","COPPER","PLATINUM"],
        "ltf":           "1h",
        # Collection thresholds to try (lower = more signals, lower raw WR)
        "col_thresholds": [0.20, 0.25, 0.28, 0.32],
        # Inference thresholds to test for each trained model
        "inf_thresholds": [0.30, 0.35, 0.38, 0.40, 0.45],
        "base_model":    "models/ict_predictor_commodity.pkl",  # current baseline
    },
    "forex": {
        "plan_key":      "plan_c_ict_forex",
        "symbols":       ["EURUSD","GBPUSD","USDJPY","EURJPY","GBPJPY","AUDUSD","USDCAD","EURGBP","NZDUSD","AUDCAD"],
        "ltf":           "1h",
        "col_thresholds": [0.20, 0.25, 0.28, 0.32],
        "inf_thresholds": [0.28, 0.30, 0.33, 0.35, 0.38],
        "base_model":    "models/ict_predictor_forex.pkl",
    },
    "india_equity": {
        "plan_key":      "plan_c_ict_india_equity",
        "symbols":       ["RELIANCE","HDFCBANK","TCS","INFY","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","LT","HINDUNILVR"],
        "ltf":           "1h",
        "col_thresholds": [0.20, 0.22, 0.25, 0.28],
        "inf_thresholds": [0.38, 0.40, 0.45, 0.50, 0.55],
        "base_model":    "models/ict_predictor_india_equity.pkl",
    },
}


def train_model(plan_key: str, col_threshold: float, seg_name: str, trials: int = 50) -> str:
    """Train one model variant. Returns path to saved pkl."""
    col_str = str(int(col_threshold * 100))
    model_path = f"models/ict_predictor_{seg_name}_col{col_str}.pkl"

    # Override confidence_threshold in config temporarily via env
    cmd = [
        sys.executable, "scripts/run_ml_train.py",
        "--plan-key",   plan_key,
        "--model-path", model_path,
        "--segment",    f"{seg_name}_col{col_str}",
        "--trials",     str(trials),
        "--threshold",  str(col_threshold),   # sets inference threshold for sweep display
        "--no-save",                            # don't save dataset CSV to avoid overwrite
    ]

    # Patch confidence_threshold in the plan config for this run
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    seg_cfg = dict(raw.get(plan_key, {}))
    seg_cfg["confidence_threshold"] = col_threshold

    import tempfile, shutil
    # Write a temporary patched config
    tmp_cfg = dict(raw)
    tmp_cfg[plan_key] = seg_cfg
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(tmp_cfg, tf)
        tmp_path = tf.name

    # Re-run using the patched config path — simplest approach: pass via env override
    # Since we can't easily pass a config path, we'll temporarily patch config.yaml
    shutil.copy("config.yaml", "config.yaml.grid_backup")
    with open("config.yaml", "w") as f:
        yaml.dump(tmp_cfg, f)

    try:
        result = subprocess.run(cmd, capture_output=False, timeout=1800)
        ok = result.returncode == 0
    finally:
        shutil.copy("config.yaml.grid_backup", "config.yaml")
        os.unlink("config.yaml.grid_backup")
        os.unlink(tmp_path)

    return model_path if ok and os.path.exists(model_path) else None


def backtest_variant(model_path: str, plan_key: str, symbols: list,
                     ltf: str, inf_threshold: float, all_data: dict, cfg) -> dict:
    """Run backtest for one (model, inference_threshold) pair. Returns metrics dict."""
    from scripts.run_strategy import run_backtest, compute_metrics
    from strategies.registry import get_strategy
    from ml.supervised import ICTTradePredictor

    predictor = ICTTradePredictor()
    if not predictor.load(model_path):
        return {}

    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    seg_cfg = dict(raw.get(plan_key, raw.get("plan_c_ict", {})))
    base = dict(raw.get("plan_c_ict", {}))
    base.update(seg_cfg)
    base["symbols"] = symbols
    base["training_symbols"] = []

    from config.config import load_config
    cfg2 = load_config("config.yaml")
    setattr(cfg2, "plan_c_ict", base)

    strat = get_strategy("plan_c_ict", cfg2)
    strat._ml_model = predictor
    strat._ml_threshold = inf_threshold

    try:
        trades, equity, _ = run_backtest(strat, all_data, symbols, ltf,
                                          capital=10_000, risk_pct=0.01,
                                          max_dd=cfg2.risk.max_drawdown_pct)
        m = compute_metrics(trades, equity, 10_000)
        m["inf_threshold"] = inf_threshold
        m["model_path"] = model_path
        return m
    except Exception as e:
        return {"error": str(e), "inf_threshold": inf_threshold, "model_path": model_path}


def rules_baseline(plan_key: str, symbols: list, ltf: str, all_data: dict) -> dict:
    """Rules-only baseline for a segment."""
    from scripts.run_strategy import run_backtest, compute_metrics
    from strategies.registry import get_strategy
    from config.config import load_config

    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    seg_cfg = dict(raw.get(plan_key, raw.get("plan_c_ict", {})))
    base = dict(raw.get("plan_c_ict", {}))
    base.update(seg_cfg)
    base["symbols"] = symbols
    base["training_symbols"] = []

    cfg2 = load_config("config.yaml")
    setattr(cfg2, "plan_c_ict", base)

    strat = get_strategy("plan_c_ict", cfg2)
    strat._ml_model = None; strat._ml_threshold = 0.0

    try:
        trades, equity, _ = run_backtest(strat, all_data, symbols, ltf,
                                          capital=10_000, risk_pct=0.01,
                                          max_dd=cfg2.risk.max_drawdown_pct)
        return compute_metrics(trades, equity, 10_000)
    except Exception:
        return {}


def print_segment_table(seg_name: str, results: list, baseline: dict):
    print(f"\n{'═'*80}")
    print(f"  {seg_name.upper()} SEGMENT — GRID RESULTS")
    print(f"{'═'*80}")
    if baseline:
        print(f"  Rules-only baseline: {baseline.get('trades',0)} trades  "
              f"{baseline.get('win_rate',0):.0%} WR  "
              f"{baseline.get('avg_rr',0):+.2f} AvgRR  "
              f"{baseline.get('max_dd',0):.1%} MaxDD  "
              f"{baseline.get('ret_pct',0):+.1f}%")
    print()
    print(f"  {'Model':>25}  {'InfThr':>7}  {'Trades':>7}  {'WinR':>6}  "
          f"{'AvgRR':>7}  {'MaxDD':>7}  {'Return':>9}  {'Best?':>6}")
    print("  " + "─" * 76)
    best_return = max((r.get("ret_pct", -999) for r in results if "trades" in r), default=-999)
    for r in results:
        if "trades" not in r:
            continue
        is_best = "⭐" if abs(r.get("ret_pct", -999) - best_return) < 0.01 and best_return > 0 else ""
        col_tag = os.path.basename(r["model_path"]).replace("ict_predictor_","").replace(".pkl","")
        print(f"  {col_tag:>25}  {r['inf_threshold']:>7.2f}  {r['trades']:>7}  "
              f"{r['win_rate']:>5.0%}  {r['avg_rr']:>+7.2f}  "
              f"{r['max_dd']:>6.1%}  {r['ret_pct']:>+8.1f}%  {is_best:>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", default=None, choices=list(SEGMENTS.keys()))
    parser.add_argument("--trials",  type=int, default=50)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only run backtests on existing models")
    args = parser.parse_args()

    segs = {args.segment: SEGMENTS[args.segment]} if args.segment else SEGMENTS

    from scripts.run_strategy import load_cache
    from config.config import load_config
    all_data = load_cache()
    print(f"Loaded {len(all_data)} cached datasets\n")

    all_results = []

    for seg_name, seg in segs.items():
        print(f"\n{'▶'*3} SEGMENT: {seg_name.upper()} {'▶'*3}")

        # Rules-only baseline
        print(f"  Running rules-only baseline...")
        baseline = rules_baseline(seg["plan_key"], seg["symbols"], seg["ltf"], all_data)

        # Train one model per collection threshold
        model_paths = {}
        for col_thr in seg["col_thresholds"]:
            col_str = str(int(col_thr * 100))
            model_path = f"models/ict_predictor_{seg_name}_col{col_str}.pkl"

            if args.skip_train and os.path.exists(model_path):
                print(f"  Using existing model: {model_path}")
                model_paths[col_thr] = model_path
            elif os.path.exists(model_path) and not args.skip_train:
                print(f"  Retraining {model_path} (col_threshold={col_thr:.2f})...")
                mp = train_model(seg["plan_key"], col_thr, seg_name, args.trials)
                model_paths[col_thr] = mp if mp else model_path
            else:
                print(f"  Training {model_path} (col_threshold={col_thr:.2f})...")
                mp = train_model(seg["plan_key"], col_thr, seg_name, args.trials)
                if mp:
                    model_paths[col_thr] = mp

        # Test each model at each inference threshold
        seg_results = []
        for col_thr, model_path in sorted(model_paths.items()):
            if not model_path or not os.path.exists(model_path):
                continue
            for inf_thr in seg["inf_thresholds"]:
                print(f"  Backtesting {os.path.basename(model_path)} @ ML≥{inf_thr:.2f}...")
                m = backtest_variant(model_path, seg["plan_key"], seg["symbols"],
                                     seg["ltf"], inf_thr, all_data, None)
                m["segment"] = seg_name
                m["col_threshold"] = col_thr
                seg_results.append(m)
                all_results.append(m)

        print_segment_table(seg_name, seg_results, baseline)

    # Save full results
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame([r for r in all_results if "trades" in r])
    if not df_results.empty:
        df_results.to_csv("results/segment_grid_results.csv", index=False)
        print(f"\n  Full results saved → results/segment_grid_results.csv")

    print("\n  MODELS SAVED:")
    for f in sorted(os.listdir("models")):
        if f.endswith(".pkl"):
            size = os.path.getsize(f"models/{f}") // 1024
            print(f"    models/{f:<50} {size}KB")


if __name__ == "__main__":
    main()
