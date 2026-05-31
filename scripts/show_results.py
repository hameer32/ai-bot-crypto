#!/usr/bin/env python3
"""Print a formatted summary of grid search or layer comparison results."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

CSV_OPTIONS = {
    "grid":  ["results/fast_grid_search_results.csv", "results/grid_search_results.csv"],
    "layer": ["results/layer_comparison.csv"],
}


def show_grid(path: str) -> None:
    df = pd.read_csv(path)
    # Normalise param column names
    for k in ["n_period", "vol_pct_thresh", "body_ratio_thresh", "sweep_lookback", "atr_k"]:
        if f"param_{k}" in df.columns:
            df[k] = df[f"param_{k}"]

    print(f"\n  Grid Search Results — {path}")
    print(f"  {len(df)} rows | TFs: {sorted(df['tf'].unique())}\n")
    print(f"  {'Symbol':<10} {'TF':<5} {'TstExp':>8} {'TstWR':>6} {'Trades':>7}  "
          f"{'n':>3} {'vol':>4} {'brt':>5} {'slb':>4} {'atrk':>5}")
    print("  " + "─" * 75)

    for _, r in df.sort_values("test_exp", ascending=False).iterrows():
        print(f"  {r['symbol']:<10} {r['tf']:<5} {r['test_exp']:>+8.1f} "
              f"{r.get('test_wr', 0):>5.0%} {int(r.get('test_trades', 0)):>7}  "
              f"{int(r.get('n_period', 0)):>3} {int(r.get('vol_pct_thresh', 0)):>4} "
              f"{r.get('body_ratio_thresh', 0):>5.2f} {int(r.get('sweep_lookback', 0)):>4} "
              f"{r.get('atr_k', 0):>5.2f}")


def show_layer(path: str) -> None:
    df = pd.read_csv(path)
    print(f"\n  Layer Comparison Results — {path}")
    print(f"  {len(df)} rows\n")
    for col in ["symbol", "tf", "combo", "expectancy", "win_rate", "trades", "ret_pct"]:
        if col not in df.columns:
            df[col] = ""

    print(f"  {'Symbol':<10} {'TF':<5} {'Combo':<25} {'Exp':>8} {'WR':>6} {'Trades':>7} {'Ret%':>7}")
    print("  " + "─" * 75)
    for _, r in df.sort_values("expectancy", ascending=False).iterrows():
        print(f"  {r['symbol']:<10} {r['tf']:<5} {str(r['combo']):<25} "
              f"{r['expectancy']:>+8.1f} {r.get('win_rate', 0):>5.0%} "
              f"{int(r.get('trades', 0)):>7} {r.get('ret_pct', 0):>+6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type", nargs="?", default="grid", choices=["grid", "layer"])
    parser.add_argument("--file", default=None)
    args = parser.parse_args()

    if args.file:
        paths = [args.file]
    else:
        paths = CSV_OPTIONS[args.type]

    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        print(f"ERROR: No results file found. Tried: {paths}")
        sys.exit(1)

    if args.type == "grid":
        show_grid(path)
    else:
        show_layer(path)


if __name__ == "__main__":
    main()
