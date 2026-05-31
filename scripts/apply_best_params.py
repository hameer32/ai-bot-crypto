#!/usr/bin/env python3
"""
Apply best grid search parameters to config.yaml.

Reads results/fast_grid_search_results.csv (or grid_search_results.csv),
picks the best params per timeframe by test expectancy,
then updates plan_a_swing and plan_b_scalp blocks in config.yaml.

Mapping:
  5m  → plan_b_scalp entry TF  (also plan_a T3)
  15m → plan_a_swing entry TF
  30m → plan_b_scalp zone TF
  1h  → plan_a zone / plan_b bias TF
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml


CSV_PRIORITY = [
    "results/fast_grid_search_results.csv",
    "results/grid_search_results.csv",
]

# Which TF's best params map to which config block + keys
TF_STRATEGY_MAP = {
    "5m":  ("plan_b_scalp", "entry"),  # scalp entry TF
    "15m": ("plan_a_swing", "entry"),  # swing entry TF
    "30m": ("plan_b_scalp", "zone"),   # scalp zone TF — use as secondary reference
    "1h":  ("plan_a_swing", "zone"),   # swing zone TF — use as secondary reference
}

PARAM_KEYS = ["n_period", "vol_pct_thresh", "body_ratio_thresh", "sweep_lookback", "atr_k"]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names — fast_grid uses param_ prefix
    renames = {}
    for k in PARAM_KEYS:
        if f"param_{k}" in df.columns and k not in df.columns:
            renames[f"param_{k}"] = k
    if renames:
        df = df.rename(columns=renames)
    return df


def best_params_per_tf(df: pd.DataFrame) -> dict:
    """Return {tf: {param: value}} for best test expectancy combo per TF."""
    results = {}
    for tf, group in df.groupby("tf"):
        # Filter: need at least 5 test trades
        trades_col = "test_trades" if "test_trades" in group.columns else "tst_trades" if "tst_trades" in group.columns else None
        valid = group[group[trades_col] >= 5] if trades_col else group
        if valid.empty:
            valid = group
        best = valid.loc[valid["test_exp"].idxmax()]
        params = {k: best[k] for k in PARAM_KEYS if k in best.index}
        results[tf] = params
        print(f"  {tf}: test_exp={best['test_exp']:+.1f}  {params}")
    return results


def update_config(config_path: str, best: dict) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    changed = []

    # plan_a_swing: use 15m (entry) params as primary
    if "15m" in best:
        p = best["15m"]
        pa = cfg.setdefault("plan_a_swing", {})
        for k, v in p.items():
            old = pa.get(k)
            pa[k] = round(float(v), 4) if isinstance(v, float) else int(v)
            if old != pa[k]:
                changed.append(f"plan_a_swing.{k}: {old} → {pa[k]}")

    # plan_b_scalp: use 5m (entry) params as primary
    if "5m" in best:
        p = best["5m"]
        pb = cfg.setdefault("plan_b_scalp", {})
        for k, v in p.items():
            old = pb.get(k)
            pb[k] = round(float(v), 4) if isinstance(v, float) else int(v)
            if old != pb[k]:
                changed.append(f"plan_b_scalp.{k}: {old} → {pb[k]}")

    # Also update rules block (used by original_smc) with 15m params if available
    if "15m" in best:
        p = best["15m"]
        r = cfg.setdefault("rules", {})
        for k, v in p.items():
            old = r.get(k)
            new_v = round(float(v), 4) if isinstance(v, float) else int(v)
            r[k] = new_v
            if old != new_v:
                changed.append(f"rules.{k}: {old} → {new_v}")

    if changed:
        print(f"\n  Applying {len(changed)} changes to {config_path}:")
        for c in changed:
            print(f"    {c}")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"\n  config.yaml updated.")
    else:
        print("\n  No changes needed — config already matches best params.")


def main():
    parser = argparse.ArgumentParser(description="Apply best grid search params to config.yaml")
    parser.add_argument("--csv", default=None, help="Path to results CSV (auto-detected if omitted)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        for p in CSV_PRIORITY:
            if os.path.exists(p):
                csv_path = p
                break

    if csv_path is None or not os.path.exists(csv_path):
        print("ERROR: No results CSV found. Run grid search first.")
        sys.exit(1)

    print(f"  Reading: {csv_path}")
    df = load_results(csv_path)
    print(f"  {len(df)} rows, TFs: {sorted(df['tf'].unique())}")

    print("\n  Best params per timeframe (by test expectancy):")
    best = best_params_per_tf(df)

    if args.dry_run:
        print("\n  [dry-run] Not writing config.")
        return

    update_config(args.config, best)


if __name__ == "__main__":
    main()
