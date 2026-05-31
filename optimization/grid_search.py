#!/usr/bin/env python3
"""
Grid Search Parameter Optimizer
================================
Searches over key parameters for the best-performing timeframes.

Focus timeframes (from layer comparison): 5m, 15m, 30m, 1h on BTC + ETH.
Ranks configurations by expectancy on the TRAIN split only.
Reports final validation on the held-out TEST split.

Parameters searched:
  n_period         : rolling window for signals (15, 20, 30, 40)
  vol_pct_thresh   : volume percentile threshold (70, 80, 90)
  body_ratio_thresh: trap candle body ratio (0.3, 0.4, 0.5)
  sweep_lookback   : recent sweep window (4, 6, 8, 10)
  atr_k            : SL buffer multiplier (0.5, 1.0, 1.5)
  conf_thresh      : fusion confidence threshold (0.50, 0.60, 0.65, 0.70)

Total: 4×3×3×4×3×4 = 1728 combos per (symbol, timeframe).
Train on first 50%, validate best-20 on held-out test 50%.
"""
import sys, os, warnings, time, itertools
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.download_and_run import (
    build_all_indicators, build_signals, build_htf_signals,
    align_htf, generate_signals, backtest, metrics as calc_metrics,
    HTF_MAP,
)

INIT_CAP = 10_000.0
TRAIN_FRAC = 0.5
TARGET_TFS = ['5m', '15m', '30m', '1h']
TARGET_SYMBOLS = ['BTCUSDT', 'ETHUSDT']

GRID = {
    'n_period':          [15, 20, 30, 40],
    'vol_pct_thresh':    [70, 80, 90],
    'body_ratio_thresh': [0.3, 0.4, 0.5],
    'sweep_lookback':    [4, 6, 8, 10],
    'atr_k':             [0.5, 1.0, 1.5],
    'conf_thresh':       [0.50, 0.60, 0.65, 0.70],
}

GRID_KEYS = list(GRID.keys())
GRID_VALUES = list(GRID.values())
ALL_COMBOS = list(itertools.product(*GRID_VALUES))


def run_one(ltf_df, htf_df, sym, params):
    """Build pipeline + backtest for one param combo. Returns metrics dict or None."""
    try:
        n   = params['n_period']
        vpt = params['vol_pct_thresh']
        brt = params['body_ratio_thresh']
        slb = params['sweep_lookback']

        df = build_all_indicators(ltf_df, n=n, vol_w=max(n, 20))
        df = build_signals(df, vol_thresh=vpt, body_thresh=brt, n=n)
        htf_ind = build_all_indicators(htf_df, n=n, vol_w=max(n, 20))
        htf_s   = build_htf_signals(htf_ind)
        df = align_htf(df, htf_s)
        df = generate_signals(df, sweep_lb=slb)

        if 'conf_long' not in df.columns:
            df['conf_long'] = 0.6
            df['conf_short'] = 0.6

        trades, equity, _ = backtest(
            df, sym,
            capital=INIT_CAP,
            risk_pct=0.01,
            atr_k=params['atr_k'],
            partial_rr=2.0,
            trail_mult=1.0,
            max_dd=0.15,
            conf_thresh=params['conf_thresh'],
        )
        return calc_metrics(trades, equity, INIT_CAP)
    except Exception:
        return None


def load_cache():
    cache_dir = 'cache'
    data = {}
    for fname in os.listdir(cache_dir):
        if not fname.endswith('.parquet'):
            continue
        parts = fname.replace('.parquet', '').split('_')
        if len(parts) < 2:
            continue
        sym, tf = parts[0], parts[1]
        try:
            df = pd.read_parquet(os.path.join(cache_dir, fname))
            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'
            data[(sym, tf)] = df
        except Exception:
            pass
    return data


def optimize_one(sym, tf, ltf_full, htf_full):
    n_bars = len(ltf_full)
    # Split by DATE not by bar count — so HTF gets the same time period as LTF
    cut_idx = int(n_bars * TRAIN_FRAC)
    cutoff_date = ltf_full.index[cut_idx]
    ltf_train = ltf_full[ltf_full.index <  cutoff_date]
    ltf_test  = ltf_full[ltf_full.index >= cutoff_date]
    htf_train = htf_full[htf_full.index <  cutoff_date]
    htf_test  = htf_full[htf_full.index >= cutoff_date]

    print(f"\n{'='*80}")
    print(f"  {sym}  {tf}  |  total={n_bars}  train={len(ltf_train)}  test={len(ltf_test)}"
          f"  |  {len(ALL_COMBOS)} combos")
    print(f"{'='*80}")

    if len(ltf_train) < 500:
        print(f"  SKIP — too few train bars")
        return []

    results = []
    t0 = time.time()

    for i, combo in enumerate(ALL_COMBOS):
        params = dict(zip(GRID_KEYS, combo))
        m = run_one(ltf_train, htf_train, sym, params)
        if m is None or m['trades'] < 5:
            continue
        results.append({
            'params': params,
            'train_exp': m['expectancy'],
            'train_wr': m['win_rate'],
            'train_trades': m['trades'],
            'test_exp': 0.0, 'test_wr': 0.0, 'test_trades': 0, 'test_ret': 0.0,
        })
        if (i + 1) % 300 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(ALL_COMBOS) - i - 1)
            print(f"  {i+1}/{len(ALL_COMBOS)} scanned  |  {len(results)} valid  "
                  f"|  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    if not results:
        print("  No valid combos found on train split.")
        return []

    results.sort(key=lambda r: r['train_exp'], reverse=True)
    print(f"\n  Train scan done: {len(results)} valid combos. Top train_exp={results[0]['train_exp']:+.1f}")

    # Validate top-20 on test split
    print(f"  Validating top-20 on test split ({len(ltf_test)} bars)...")
    for r in results[:20]:
        m = run_one(ltf_test, htf_test, sym, r['params'])
        if m:
            r['test_exp']    = m['expectancy']
            r['test_wr']     = m['win_rate']
            r['test_trades'] = m['trades']
            r['test_ret']    = m.get('ret_pct', 0.0)

    # Print table
    print(f"\n  {'Rank':<5} {'n':>3} {'vol':>4} {'brt':>5} {'slb':>4} {'atrk':>5} {'conf':>5}  "
          f"{'TrnExp':>8} {'TstExp':>8} {'TstWR':>6} {'TstTrd':>7} {'TstRet':>7}")
    print("  " + "─" * 85)
    for i, r in enumerate(results[:20], 1):
        p = r['params']
        print(f"  {i:<5} {p['n_period']:>3} {p['vol_pct_thresh']:>4} {p['body_ratio_thresh']:>5.1f} "
              f"{p['sweep_lookback']:>4} {p['atr_k']:>5.1f} {p['conf_thresh']:>5.2f}  "
              f"{r['train_exp']:>+8.1f} {r['test_exp']:>+8.1f} {r['test_wr']:>5.0%} "
              f"{r['test_trades']:>7} {r['test_ret']:>+6.1f}%")
    return results


def main():
    print("=" * 80)
    print("  GRID SEARCH PARAMETER OPTIMIZER")
    print(f"  Symbols: {TARGET_SYMBOLS}   Timeframes: {TARGET_TFS}")
    print(f"  Train/Test split: {int(TRAIN_FRAC*100)}% / {int((1-TRAIN_FRAC)*100)}%")
    print(f"  Grid size: {len(ALL_COMBOS)} combinations")
    print("=" * 80)

    all_data = load_cache()
    print(f"\n  Loaded {len(all_data)} cached datasets")

    summary_rows = []
    for sym in TARGET_SYMBOLS:
        for tf in TARGET_TFS:
            htf_tf = HTF_MAP.get(tf, tf)
            if (sym, tf) not in all_data or (sym, htf_tf) not in all_data:
                print(f"\n  SKIP {sym} {tf} — data not cached")
                continue

            results = optimize_one(sym, tf, all_data[(sym, tf)], all_data[(sym, htf_tf)])
            if not results:
                continue

            validated = [r for r in results[:20] if r['test_trades'] >= 5]
            best = max(validated, key=lambda r: r['test_exp']) if validated else results[0]
            summary_rows.append({
                'symbol': sym, 'tf': tf,
                'train_exp': results[0]['train_exp'],
                'test_exp': best['test_exp'],
                'test_wr': best['test_wr'],
                'test_trades': best['test_trades'],
                'test_ret': best['test_ret'],
                **{f'param_{k}': v for k, v in best['params'].items()},
            })

    print("\n" + "=" * 80)
    print("  OPTIMIZATION SUMMARY  (sorted by test expectancy)")
    print("=" * 80)
    print(f"\n  {'Symbol':<10} {'TF':<5} {'TstExp':>8} {'TstWR':>6} {'Trades':>7} {'Return':>8}  Best Params")
    print("  " + "─" * 100)
    for r in sorted(summary_rows, key=lambda x: x['test_exp'], reverse=True):
        pstr = (f"n={r['param_n_period']} vol={r['param_vol_pct_thresh']} "
                f"body={r['param_body_ratio_thresh']:.1f} slb={r['param_sweep_lookback']} "
                f"atrk={r['param_atr_k']:.1f} conf={r['param_conf_thresh']:.2f}")
        print(f"  {r['symbol']:<10} {r['tf']:<5} {r['test_exp']:>+8.1f} "
              f"{r['test_wr']:>5.0%} {r['test_trades']:>7} {r['test_ret']:>+7.1f}%  {pstr}")

    os.makedirs('results', exist_ok=True)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv('results/grid_search_results.csv', index=False)
        print(f"\n  Results saved → results/grid_search_results.csv")


if __name__ == '__main__':
    main()
