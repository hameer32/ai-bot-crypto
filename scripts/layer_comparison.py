#!/usr/bin/env python3
"""
Layer Comparison Engine
=======================
Tests every combination of the 3 signal layers (Rules, Quant, ML-stub)
across all timeframes for BTC and ETH.

Configurations tested per timeframe:
  1.  Rules only
  2.  Quant only
  3.  ML only (stub — XGBoost trained on prior trades)
  4.  Rules + Quant
  5.  Rules + ML
  6.  Quant  + ML
  7.  Rules + Quant + ML   (full system)

Each config runs a full backtest and reports all metrics.
Final comparison table ranks every combo by expectancy.
"""
import sys, os, time, warnings, logging, itertools, copy
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Re-use helpers from download_and_run ─────────────────────────────────────
from scripts.download_and_run import (
    build_all_indicators, build_signals, build_htf_signals,
    align_htf, generate_signals, backtest, metrics as calc_metrics,
    HTF_MAP, TIMEFRAMES,
)

INIT_CAP = 10_000.0

# ── Quant scoring (inline, no hmmlearn required for speed) ──────────────────

def quant_scores(df, cfg_q):
    """Compute per-row quant sub-scores, return mean as P_quant series."""
    scores = []

    if cfg_q.get('mean_reversion'):
        c = cfg_q['mean_reversion']
        mid = df['close'].rolling(c['bb_period']).mean()
        sig = df['close'].rolling(c['bb_period']).std()
        upper = mid + c['bb_std']*sig
        lower = mid - c['bb_std']*sig
        rng = (upper-lower).replace(0, np.nan)
        pos = ((df['close']-lower)/rng).clip(0,1)
        bb = 1.0 - pos          # 1 = at lower band (bullish setup)
        scores.append(bb.fillna(0.5))

    if cfg_q.get('microstructure'):
        c = cfg_q['microstructure']
        w = c['kyle_window']
        kl = (df['close'].diff().abs() / df['volume'].replace(0,np.nan)).rolling(w).mean()
        am = (np.log(df['close']/df['close'].shift(1)).abs() / (df['close']*df['volume']).replace(0,np.nan)).rolling(w).mean()
        def norm(s):
            mn=s.rolling(100,min_periods=20).min(); mx=s.rolling(100,min_periods=20).max()
            return ((s-mn)/(mx-mn+1e-12)).clip(0,1).fillna(0.5)
        scores.append(((norm(kl)+norm(am))/2))

    if cfg_q.get('momentum'):
        c = cfg_q['momentum']
        lb = c['tsm_lookback']
        ret = df['close'].pct_change(lb)
        rm = ret.rolling(lb).mean(); rs = ret.rolling(lb).std().replace(0,np.nan)
        z = (ret-rm)/rs
        tsm = (1/(1+np.exp(-z.clip(-5,5)))).fillna(0.5)
        scores.append(tsm)

    if cfg_q.get('regime'):
        # Lightweight HMM-free regime: use ATR percentile as trending proxy
        atr_pct = df['atr_pct'].fillna(50)
        trending = (atr_pct / 100).clip(0,1)   # high ATR → trending
        scores.append(trending)

    if not scores:
        return pd.Series(0.5, index=df.index)

    result = pd.concat(scores, axis=1).mean(axis=1).clip(0,1).fillna(0.5)
    result.name = 'P_quant'
    return result


# ── Supervised ML stub (trains on backtest results) ─────────────────────────

FEATURE_COLS = ['atr','atr_pct','vol_pct','spread_pct','body_ratio','ema_slope',
                'sweep_bull','sweep_bear','trap_bull','trap_bear','of_bull','of_bear']

def train_ml_model(ltf_df, trades, direction):
    """Train classifier on trades from a prior backtest run."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return None

    rows, labels = [], []
    for t in trades:
        if t.get('dir') != direction: continue
        ts = t.get('ts')
        if ts not in ltf_df.index: continue
        row = ltf_df.loc[ts]
        feat = {c: float(row.get(c, 0)) for c in FEATURE_COLS}
        feat['htf_bull'] = 1 if str(row.get('htf_structure','')) == 'bullish' else 0
        feat['htf_bear'] = 1 if str(row.get('htf_structure','')) == 'bearish' else 0
        rows.append(feat)
        labels.append(1 if t['pnl'] > 0 else 0)

    if len(rows) < 20: return None
    X = pd.DataFrame(rows).fillna(0)
    y = pd.Series(labels)
    if y.nunique() < 2: return None
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf


def ml_predict(clf, ltf_df):
    """Return P_ml series for each row."""
    if clf is None:
        return pd.Series(0.5, index=ltf_df.index)
    X = pd.DataFrame({c: ltf_df.get(c, pd.Series(0, index=ltf_df.index)) for c in FEATURE_COLS})
    X['htf_bull'] = (ltf_df.get('htf_structure', pd.Series('', index=ltf_df.index)) == 'bullish').astype(float)
    X['htf_bear'] = (ltf_df.get('htf_structure', pd.Series('', index=ltf_df.index)) == 'bearish').astype(float)
    proba = clf.predict_proba(X.fillna(0))
    return pd.Series(proba[:, 1], index=ltf_df.index, name='P_ml')


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse(P_rules, P_quant, P_ml, weights, method='weighted_average'):
    """Combine active layer scores into final confidence series."""
    active = [(w, s) for w, s in zip(weights, [P_rules, P_quant, P_ml]) if s is not None]
    if not active:
        return pd.Series(0.0, index=(P_rules or P_quant or P_ml).index)

    idx = active[0][1].index

    if method == 'weighted_average':
        total_w = sum(w for w, _ in active)
        result = sum(w * s.reindex(idx).fillna(0.5) for w, s in active) / total_w
    elif method == 'min_score':
        result = pd.concat([s.reindex(idx).fillna(0.5) for _, s in active], axis=1).min(axis=1)
    elif method == 'product':
        n = len(active)
        result = pd.concat([s.reindex(idx).fillna(0.5) for _, s in active], axis=1).prod(axis=1) ** (1/n)
    else:
        result = active[0][1]

    return result.clip(0, 1)


def confidence_rules(df, direction):
    w = {'htf':2,'sweep':2,'trap':3,'of':2,'vol':1}
    s = pd.Series(0., index=df.index)
    htf = df.get('htf_structure', pd.Series('neutral', index=df.index))
    if direction == 'long':
        s += (htf=='bullish').astype(float)*w['htf']
        s += df['sweep_bull'].astype(float)*w['sweep']
        s += df['trap_bull'].astype(float)*w['trap']
        s += df['of_bull'].astype(float)*w['of']
    else:
        s += (htf=='bearish').astype(float)*w['htf']
        s += df['sweep_bear'].astype(float)*w['sweep']
        s += df['trap_bear'].astype(float)*w['trap']
        s += df['of_bear'].astype(float)*w['of']
    s += df['vol_regime'].isin(['normal','high']).astype(float)*w['vol']
    return (s/10).clip(0,1)


# ── Layer config definitions ──────────────────────────────────────────────────

QUANT_CFG = {
    'mean_reversion': {'bb_period':20, 'bb_std':2.0},
    'microstructure': {'kyle_window':20},
    'momentum':       {'tsm_lookback':60},
    'regime':         True,
}

LAYER_COMBOS = [
    {'name':'Rules only',         'rules':True,  'quant':False, 'ml':False},
    {'name':'Quant only',         'rules':False, 'quant':True,  'ml':False},
    {'name':'ML only',            'rules':False, 'quant':False, 'ml':True},
    {'name':'Rules + Quant',      'rules':True,  'quant':True,  'ml':False},
    {'name':'Rules + ML',         'rules':True,  'quant':False, 'ml':True},
    {'name':'Quant + ML',         'rules':False, 'quant':True,  'ml':True},
    {'name':'Rules + Quant + ML', 'rules':True,  'quant':True,  'ml':True},
]

WEIGHTS = {'rules':0.40, 'quant':0.35, 'ml':0.25}
CONF_THRESHOLD = 0.50


# ── Per-timeframe, per-combo runner ──────────────────────────────────────────

def run_combo(ltf, sym, tf, combo, quant_cfg, ml_clf_long, ml_clf_short):
    """
    Apply fusion confidence to an already-signalled ltf dataframe,
    then run backtest. Returns metrics dict.
    """
    df = ltf.copy()

    # Compute each layer's confidence series (direction-agnostic: use long score for long, short for short)
    P_rules_long  = confidence_rules(df, 'long')  if combo['rules'] else None
    P_rules_short = confidence_rules(df, 'short') if combo['rules'] else None

    P_quant = quant_scores(df, quant_cfg) if combo['quant'] else None
    # Quant is direction-neutral (same score for long/short — reflects market conditions)

    P_ml_long  = ml_predict(ml_clf_long,  df) if combo['ml'] else None
    P_ml_short = ml_predict(ml_clf_short, df) if combo['ml'] else None

    w = [WEIGHTS['rules'], WEIGHTS['quant'], WEIGHTS['ml']]

    conf_long  = fuse(P_rules_long,  P_quant, P_ml_long,  w)
    conf_short = fuse(P_rules_short, P_quant, P_ml_short, w)

    df['conf_long']  = conf_long
    df['conf_short'] = conf_short

    t_list, eq, final = backtest(
        df, sym,
        capital=INIT_CAP, risk_pct=0.01, atr_k=1.0,
        partial_rr=2.0, trail_mult=1.0, max_dd=0.10,
        conf_thresh=CONF_THRESHOLD
    )
    m = calc_metrics(t_list, eq, INIT_CAP)
    return m, t_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    CACHE_DIR = 'cache'
    SYMBOLS   = ['BTCUSDT', 'ETHUSDT']
    N_PERIOD  = 20

    print('='*80)
    print('  LAYER COMPARISON ENGINE')
    print('  Testing 7 layer combinations × 8 timeframes × 2 symbols')
    print('='*80)

    # Load cached data
    all_data = {}
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            path = f'{CACHE_DIR}/{sym}_{tf}.parquet'
            if not os.path.exists(path):
                print(f'  [MISS] {sym} {tf} — run download_and_run.py first')
                continue
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            all_data[(sym, tf)] = df

    if not all_data:
        print('No cached data found. Run: python3 scripts/download_and_run.py')
        sys.exit(1)

    print(f'  Loaded {len(all_data)} symbol/timeframe datasets\n')

    all_results = []

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            ltf_raw = all_data.get((sym, tf))
            if ltf_raw is None or ltf_raw.empty or len(ltf_raw) < 200:
                continue

            htf_tf = HTF_MAP.get(tf, tf)
            htf_raw = all_data.get((sym, htf_tf))

            # Build indicators + signals once per (sym, tf)
            ltf = build_all_indicators(ltf_raw.copy(), n=N_PERIOD)
            ltf = build_signals(ltf, n=N_PERIOD)

            if htf_raw is not None and not htf_raw.empty and htf_tf != tf:
                htf = build_all_indicators(htf_raw.copy(), n=N_PERIOD)
                htf = build_htf_signals(htf)
                ltf = align_htf(ltf, htf)
            else:
                ltf['htf_structure']  = ltf['structure']
                ltf['htf_vol_regime'] = ltf['vol_regime']

            ltf = generate_signals(ltf)

            # Pre-train ML on Rules-only backtest (use first half as training)
            split_idx = len(ltf) // 2
            train_df  = ltf.iloc[:split_idx]
            train_df['conf_long']  = confidence_rules(train_df, 'long')
            train_df['conf_short'] = confidence_rules(train_df, 'short')
            train_trades, _, _ = backtest(train_df, sym, capital=INIT_CAP, conf_thresh=CONF_THRESHOLD)

            ml_clf_long  = train_ml_model(train_df, train_trades, 'long')
            ml_clf_short = train_ml_model(train_df, train_trades, 'short')
            ml_available = ml_clf_long is not None

            # Test df: second half only (avoids lookahead on ML)
            test_df = ltf.iloc[split_idx:].copy()

            ls = test_df['long_signal'].sum()
            ss = test_df['short_signal'].sum()

            print(f'\n{"─"*80}')
            print(f'  {sym}  {tf}  │  {len(test_df):,} test bars  │  {ls+ss} raw signals  │  HTF: {htf_tf}')
            print(f'{"─"*80}')
            print(f'  {"Config":<22}  {"Trades":>6}  {"WinRate":>7}  {"AvgRR":>6}  {"Expect":>8}  {"MaxDD":>6}  {"ProfFactor":>10}  {"Return":>7}')
            print(f'  {"─"*22}  {"─"*6}  {"─"*7}  {"─"*6}  {"─"*8}  {"─"*6}  {"─"*10}  {"─"*7}')

            for combo in LAYER_COMBOS:
                if combo['ml'] and not ml_available:
                    label = combo['name'] + ' [no ML data]'
                    print(f'  {label:<22}  {"—":>6}  {"—":>7}  {"—":>6}  {"—":>8}  {"—":>6}  {"—":>10}  {"—":>7}')
                    continue

                m, _ = run_combo(test_df, sym, tf, combo, QUANT_CFG,
                                 ml_clf_long, ml_clf_short)

                wr_str  = f'{m["win_rate"]:.0%}'
                rr_str  = f'{m["avg_rr"]:+.2f}'
                ex_str  = f'{m["expectancy"]:+.1f}'
                dd_str  = f'{m["max_dd"]:.1%}'
                pf_str  = f'{m["pf"]:.2f}' if m["pf"] < 99 else '>99'
                ret_str = f'{m["ret_pct"]:+.1f}%'
                tr_str  = str(m['trades'])

                marker = ' ◀ BEST' if m['expectancy'] > 0 and m['trades'] >= 5 else ''

                print(f'  {combo["name"]:<22}  {tr_str:>6}  {wr_str:>7}  {rr_str:>6}  {ex_str:>8}  {dd_str:>6}  {pf_str:>10}  {ret_str:>7}{marker}')

                all_results.append({
                    'Symbol': sym, 'TF': tf,
                    'Config': combo['name'],
                    'Rules': combo['rules'], 'Quant': combo['quant'], 'ML': combo['ml'],
                    'TestBars': len(test_df),
                    **m
                })

    # ── Grand Summary ─────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print('  GRAND COMPARISON SUMMARY  (sorted by Expectancy)')
    print(f'{"="*80}')

    df_res = pd.DataFrame(all_results)
    if df_res.empty:
        print('No results.'); return

    with_trades = df_res[df_res['trades'] >= 5].sort_values('expectancy', ascending=False)
    print(f'\n  Top 20 configs by Expectancy (≥5 trades):')
    print(f'\n  {"Symbol":9} {"TF":4} {"Config":<22} {"Trades":>6} {"WinR":>5} {"Expect":>8} {"MaxDD":>6} {"Return":>7}')
    print(f'  {"─"*9} {"─"*4} {"─"*22} {"─"*6} {"─"*5} {"─"*8} {"─"*6} {"─"*7}')
    for _, r in with_trades.head(20).iterrows():
        print(f'  {r["Symbol"]:9} {r["TF"]:4} {r["Config"]:<22} {r["trades"]:>6} '
              f'{r["win_rate"]:>5.0%} {r["expectancy"]:>+8.1f} {r["max_dd"]:>6.1%} {r["ret_pct"]:>+7.1f}%')

    print(f'\n{"─"*80}')
    print('  BEST CONFIG PER TIMEFRAME:')
    print(f'{"─"*80}')
    for tf in TIMEFRAMES:
        sub = with_trades[with_trades['TF']==tf]
        if sub.empty: print(f'  {tf:4}  — no profitable configs with ≥5 trades'); continue
        best = sub.iloc[0]
        print(f'  {tf:4}  {best["Symbol"]:9}  {best["Config"]:<22}  '
              f'exp={best["expectancy"]:+.1f}  wr={best["win_rate"]:.0%}  ret={best["ret_pct"]:+.1f}%')

    print(f'\n{"─"*80}')
    print('  BEST CONFIG PER SYMBOL:')
    print(f'{"─"*80}')
    for sym in ['BTCUSDT','ETHUSDT']:
        sub = with_trades[with_trades['Symbol']==sym]
        if sub.empty: print(f'  {sym}  — no profitable configs'); continue
        best = sub.iloc[0]
        print(f'  {sym:9}  {best["TF"]:4}  {best["Config"]:<22}  '
              f'exp={best["expectancy"]:+.1f}  wr={best["win_rate"]:.0%}  ret={best["ret_pct"]:+.1f}%')

    print(f'\n{"─"*80}')
    print('  LAYER IMPACT ANALYSIS (avg expectancy by layer presence):')
    print(f'{"─"*80}')
    for layer in ['Rules','Quant','ML']:
        on  = df_res[df_res[layer]==True]['expectancy'].mean()
        off = df_res[df_res[layer]==False]['expectancy'].mean()
        delta = on - off
        verdict = 'HELPS' if delta > 0 else 'HURTS'
        print(f'  {layer:6}  ON avg exp={on:+.1f}   OFF avg exp={off:+.1f}   delta={delta:+.1f}  → {verdict}')

    print(f'\n{"─"*80}')
    print('  FUSION BENEFIT:')
    print(f'{"─"*80}')
    solo_exp   = df_res[df_res['Config'].isin(['Rules only','Quant only','ML only'])]['expectancy'].mean()
    dual_exp   = df_res[df_res['Config'].isin(['Rules + Quant','Rules + ML','Quant + ML'])]['expectancy'].mean()
    triple_exp = df_res[df_res['Config']=='Rules + Quant + ML']['expectancy'].mean()
    print(f'  Single-layer avg expectancy:   {solo_exp:+.2f}')
    print(f'  Dual-layer   avg expectancy:   {dual_exp:+.2f}')
    print(f'  Triple-layer avg expectancy:   {triple_exp:+.2f}')

    # Save
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/layer_comparison.csv', index=False)
    print(f'\n{"="*80}')
    print(f'  Full results saved → results/layer_comparison.csv')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
