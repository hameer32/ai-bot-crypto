#!/usr/bin/env python3
"""
Download historical OHLCV data for BTC and ETH across all timeframes,
then run the full backtest matrix and print a comprehensive results table.
"""
import sys, os, time, logging, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SYMBOLS    = ['BTC/USDT', 'ETH/USDT']
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
LOOKBACK   = {
    '1m':  30,    # 30 days
    '5m':  90,
    '15m': 365,
    '30m': 365,
    '1h':  730,
    '4h':  730,
    '1d':  1825,  # 5 years
    '1w':  1825,
}
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

TF_OFFSET = {
    '1m':'1min','5m':'5min','15m':'15min','30m':'30min',
    '1h':'1h','4h':'4h','1d':'1D','1w':'1W'
}

# ── Download ──────────────────────────────────────────────────────────────────

def fetch_ohlcv_full(exchange, symbol, timeframe, days):
    until_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_rows  = []
    current   = since_ms
    safe_sym  = symbol.replace('/', '')

    while current < until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        except Exception as e:
            logger.warning('Fetch error %s %s: %s', symbol, timeframe, e)
            time.sleep(5); continue
        if not batch: break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= current: break
        current = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')

    # Fill gaps
    offset = TF_OFFSET.get(timeframe)
    if offset:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=offset, tz='UTC')
        df = df.reindex(full_idx)
        for col in ['open','high','low','close']: df[col] = df[col].ffill()
        df['volume'] = df['volume'].fillna(0)

    df.index.name = 'timestamp'
    logger.info('  %s %s: %d rows  %s → %s', safe_sym, timeframe, len(df),
                df.index.min().date(), df.index.max().date())
    return df


def download_all(exchange):
    data = {}
    for sym in SYMBOLS:
        safe = sym.replace('/','')
        for tf in TIMEFRAMES:
            path = f'{CACHE_DIR}/{safe}_{tf}.parquet'
            if os.path.exists(path) and os.path.getmtime(path) > time.time() - 3600:
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                logger.info('  Cached: %s %s (%d rows)', safe, tf, len(df))
            else:
                logger.info('Downloading %s %s (%d days)...', sym, tf, LOOKBACK[tf])
                df = fetch_ohlcv_full(exchange, sym, tf, LOOKBACK[tf])
                if not df.empty:
                    df.to_parquet(path, compression='snappy')
            data[(safe, tf)] = df
    return data


# ── Indicators ────────────────────────────────────────────────────────────────

def build_all_indicators(df, n=20, atr_p=14, ema_p=50, vol_w=20):
    df = df.copy()
    # ATR (Wilder)
    prev_c = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'],(df['high']-prev_c).abs(),(df['low']-prev_c).abs()],axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/atr_p, min_periods=atr_p, adjust=False).mean()
    def rank_last(x):
        return float(np.sum(x[:-1]<=x[-1]))/max(len(x)-1,1)*100 if len(x)>1 else 50.
    df['atr_pct'] = df['atr'].rolling(100).apply(rank_last, raw=True)
    df['roll_high'] = df['high'].rolling(n).max()
    df['roll_low']  = df['low'].rolling(n).min()
    df['vol_pct']   = df['volume'].rolling(vol_w).apply(rank_last, raw=True)
    df['spread_pct']= (df['high']-df['low']).rolling(vol_w).apply(rank_last, raw=True)
    df['ema']       = df['close'].ewm(span=ema_p,adjust=False).mean()
    df['ema_slope'] = (df['ema']-df['ema'].shift(5))/5
    spread = (df['high']-df['low']).replace(0,np.nan)
    df['body_ratio']= ((df['close']-df['open']).abs()/spread).clip(0,1).fillna(0)
    df['spread']    = df['high']-df['low']
    # Swing points (vectorized approx)
    df['swing_high']= (df['high']==df['high'].rolling(5,center=True).max())
    df['swing_low'] = (df['low'] ==df['low'].rolling(5,center=True).min())
    return df


# ── Signals ───────────────────────────────────────────────────────────────────

def build_signals(df, vol_thresh=80, body_thresh=0.4, n=20):
    df = df.copy()
    # Liquidity sweep
    pr_low  = df['roll_low'].shift(1)
    pr_high = df['roll_high'].shift(1)
    df['sweep_bull'] = (df['low']<pr_low)&(df['close']>pr_low)&(df['vol_pct']>vol_thresh)
    df['sweep_bear'] = (df['high']>pr_high)&(df['close']<pr_high)&(df['vol_pct']>vol_thresh)
    df['sweep_low_level']  = pr_low.where(df['sweep_bull'])
    df['sweep_high_level'] = pr_high.where(df['sweep_bear'])

    # Trap
    nb_high = df['high'].rolling(n).max()
    nb_low  = df['low'].rolling(n).min()
    pr_nh   = nb_high.shift(2)
    pr_nl   = nb_low.shift(2)
    df['trap_bull'] = (df['low'].shift(1)<pr_nl) & (df['close']>df['open']) & (df['body_ratio']>body_thresh) & (df['close']>pr_nl)
    df['trap_bear'] = (df['high'].shift(1)>pr_nh) & (df['close']<df['open']) & (df['body_ratio']>body_thresh) & (df['close']<pr_nh)

    # Order flow absorption
    mid = (df['high']+df['low'])/2
    hv  = df['vol_pct']>vol_thresh
    ls  = df['spread_pct']<40
    df['of_bull'] = hv & ls & (df['close']>mid)
    df['of_bear'] = hv & ls & (df['close']<mid)

    # Market structure
    sh_vals = df['high'].where(df['swing_high']); sl_vals = df['low'].where(df['swing_low'])
    def prev_swing(series):
        swing_idx = series.dropna().index
        if len(swing_idx)<2: return pd.Series(np.nan, index=series.index)
        prev_v = pd.Series(np.nan, index=series.index)
        vals = series.dropna().values
        for i,idx in enumerate(swing_idx[1:],1):
            prev_v.loc[idx] = vals[i-1]
        return prev_v.ffill()
    sh_cur=sh_vals.ffill(); sh_prv=prev_swing(sh_vals)
    sl_cur=sl_vals.ffill(); sl_prv=prev_swing(sl_vals)
    valid = sh_prv.notna()&sl_prv.notna()
    bull_ms = valid&(sh_cur>sh_prv)&(sl_cur>sl_prv)
    bear_ms = valid&(sh_cur<sh_prv)&(sl_cur<sl_prv)
    df['structure'] = np.select([bull_ms,bear_ms],['bullish','bearish'],default='neutral')

    # Vol regime
    df['vol_regime'] = np.select([df['atr_pct']>=70,df['atr_pct']<=30],['high','low'],default='normal')
    return df


def build_htf_signals(df):
    return build_signals(df)


def align_htf(ltf, htf):
    htf_s = htf[['structure','vol_regime']].rename(columns={'structure':'htf_structure','vol_regime':'htf_vol_regime'})
    htf_s = htf_s.reset_index()
    ltf_r = ltf.reset_index()
    m = pd.merge_asof(ltf_r.sort_values('timestamp'), htf_s.sort_values('timestamp'),
                      on='timestamp', direction='backward').set_index('timestamp')
    return m


def generate_signals(df, sweep_lb=6):
    df = df.copy()
    htf = df.get('htf_structure', pd.Series('neutral',index=df.index))
    rec_bull = df['sweep_bull'].rolling(sweep_lb,min_periods=1).max().astype(bool)
    rec_bear = df['sweep_bear'].rolling(sweep_lb,min_periods=1).max().astype(bool)
    trig_bull = df['trap_bull']|df['of_bull']
    trig_bear = df['trap_bear']|df['of_bear']
    df['long_signal']  = (htf=='bullish')&rec_bull&trig_bull&df['vol_regime'].isin(['normal','high'])
    df['short_signal'] = (htf=='bearish')&rec_bear&trig_bear&df['vol_regime'].isin(['normal','high'])
    return df


def confidence(df, direction):
    w = {'htf':2,'sweep':2,'trap':3,'of':2,'vol':1}
    s = pd.Series(0.0, index=df.index)
    htf = df.get('htf_structure', pd.Series('neutral',index=df.index))
    if direction=='long':
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


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(ltf_df, symbol, capital=10000, risk_pct=0.01, atr_k=1.0,
             partial_rr=2.0, trail_mult=1.0, max_dd=0.10, conf_thresh=0.65):
    trades=[]; equity=[(ltf_df.index[0], capital)]
    open_trade=None; peak=capital; trade_id=0; kill=False

    for ts, row in ltf_df.iterrows():
        if kill: break

        # Update open trade
        if open_trade:
            hi=float(row['high']); lo=float(row['low'])
            t=open_trade
            if t['dir']=='long':
                if lo<=t['sl']:
                    pnl=(t['sl']-t['ep'])*t['sz']; t['pnl']+=pnl; t['state']='SL'
                    capital+=pnl; open_trade=None
                elif not t['partial'] and hi>=t['tp1']:
                    half=t['sz']*0.5; pnl=(t['tp1']-t['ep'])*half
                    t['pnl']+=pnl; t['sz']-=half; t['partial']=True
                    t['trail']=t['ep']; capital+=pnl
                elif t['partial'] and t['trail'] is not None:
                    new_trail=lo-trail_mult*t['atr']
                    if new_trail>t['trail']: t['trail']=new_trail
                    if lo<=t['trail']:
                        pnl=(t['trail']-t['ep'])*t['sz']; t['pnl']+=pnl
                        t['state']='TRAIL'; capital+=pnl; open_trade=None
            else:
                if hi>=t['sl']:
                    pnl=(t['ep']-t['sl'])*t['sz']; t['pnl']+=pnl; t['state']='SL'
                    capital+=pnl; open_trade=None
                elif not t['partial'] and lo<=t['tp1']:
                    half=t['sz']*0.5; pnl=(t['ep']-t['tp1'])*half
                    t['pnl']+=pnl; t['sz']-=half; t['partial']=True
                    t['trail']=t['ep']; capital+=pnl
                elif t['partial'] and t['trail'] is not None:
                    new_trail=hi+trail_mult*t['atr']
                    if new_trail<t['trail']: t['trail']=new_trail
                    if hi>=t['trail']:
                        pnl=(t['ep']-t['trail'])*t['sz']; t['pnl']+=pnl
                        t['state']='TRAIL'; capital+=pnl; open_trade=None

            if open_trade and not open_trade.get('state'):
                pass  # still open

        # Kill switch
        peak=max(peak,capital)
        if (peak-capital)/peak>=max_dd: kill=True; break

        if open_trade:
            equity.append((ts, capital))
            continue

        # New signal
        direction=None
        if row.get('long_signal',False): direction='long'
        elif row.get('short_signal',False): direction='short'
        if direction is None:
            equity.append((ts,capital)); continue

        conf = float(row.get(f'conf_{direction}',0))
        if conf < conf_thresh:
            equity.append((ts,capital)); continue

        ep=float(row['close'])
        atr_v=float(row.get('atr',ep*0.002))

        if direction=='long':
            level=float(row['sweep_low_level']) if not pd.isna(row.get('sweep_low_level',np.nan)) else float(row['roll_low'])
            sl=level-atr_k*atr_v
        else:
            level=float(row['sweep_high_level']) if not pd.isna(row.get('sweep_high_level',np.nan)) else float(row['roll_high'])
            sl=level+atr_k*atr_v

        risk_unit=abs(ep-sl)
        if risk_unit<=0:
            equity.append((ts,capital)); continue

        sz=(capital*risk_pct)/risk_unit
        if direction=='long':
            tp1=ep+(ep-sl)*partial_rr
        else:
            tp1=ep-(sl-ep)*partial_rr

        trade_id+=1
        open_trade={'id':trade_id,'dir':direction,'ep':ep,'sl':sl,'tp1':tp1,
                    'sz':sz,'atr':atr_v,'pnl':0.,'partial':False,'trail':None,
                    'state':None,'ts':ts}
        trades.append(open_trade)
        equity.append((ts,capital))

    # Close remaining
    if open_trade and open_trade.get('state') is None and len(ltf_df):
        lp=float(ltf_df['close'].iloc[-1])
        if open_trade['dir']=='long':
            pnl=(lp-open_trade['ep'])*open_trade['sz']
        else:
            pnl=(open_trade['ep']-lp)*open_trade['sz']
        open_trade['pnl']+=pnl; open_trade['state']='EOD'; capital+=pnl

    return trades, equity, capital


def metrics(trades, equity, init_cap):
    closed=[t for t in trades if t.get('state')]
    if not closed:
        return {'trades':0,'win_rate':0,'avg_rr':0,'expectancy':0,'max_dd':0,'pf':0,'final':init_cap,'ret_pct':0}
    pnls=[t['pnl'] for t in closed]
    wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
    wr=len(wins)/len(closed)
    aw=np.mean(wins) if wins else 0; al=np.mean(losses) if losses else 0
    exp=(wr*aw)+((1-wr)*al)
    gp=sum(wins); gl=abs(sum(losses))
    pf=gp/gl if gl>0 else float('inf')
    # RR per trade
    rr_list=[]
    for t in closed:
        risk=abs(t['ep']-t['sl'])*t.get('orig_sz',t['sz'])
    # simpler RR using entry-SL as 1R
    rr_list=[t['pnl']/abs((t['ep']-t['sl'])*max(t['sz'],0.0001)) for t in closed if abs(t['ep']-t['sl'])>0]
    avg_rr=np.mean(rr_list) if rr_list else 0
    # Max drawdown
    eq_vals=[v for _,v in equity]
    if eq_vals:
        arr=np.array(eq_vals); peak_arr=np.maximum.accumulate(arr)
        dd=(peak_arr-arr)/peak_arr; mdd=float(dd.max())
    else: mdd=0
    final=capital if not equity else equity[-1][1]
    return {
        'trades':len(closed),'win_rate':round(wr,3),'avg_rr':round(avg_rr,3),
        'expectancy':round(exp,2),'max_dd':round(mdd,4),
        'pf':round(pf,3),'final':round(final,2),
        'ret_pct':round((final-init_cap)/init_cap*100,2)
    }


# ── HTF lookup ────────────────────────────────────────────────────────────────

HTF_MAP = {
    '1m':'5m','5m':'1h','15m':'4h','30m':'4h',
    '1h':'4h','4h':'1d','1d':'1w','1w':'1w'
}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__=='__main__':
    print('='*70)
    print('  CRYPTO BACKTEST MATRIX — BTC & ETH — All Timeframes')
    print('='*70)

    exchange = ccxt.binance({'enableRateLimit':True})
    exchange.load_markets()

    print('\n[1/2] Downloading historical data...')
    data = download_all(exchange)

    print('\n[2/2] Running backtest matrix...')
    results=[]
    INIT_CAP=10000.0

    for sym in ['BTCUSDT','ETHUSDT']:
        for tf in TIMEFRAMES:
            ltf_df = data.get((sym,tf))
            if ltf_df is None or ltf_df.empty or len(ltf_df)<200:
                continue

            htf_tf = HTF_MAP.get(tf,tf)
            htf_df = data.get((sym,htf_tf))

            # Build indicators + signals on LTF
            n_period=20
            ltf = build_all_indicators(ltf_df.copy(), n=n_period)
            ltf = build_signals(ltf, n=n_period)

            # HTF
            if htf_df is not None and not htf_df.empty and htf_tf!=tf:
                htf = build_all_indicators(htf_df.copy(), n=n_period)
                htf = build_htf_signals(htf)
                ltf = align_htf(ltf, htf)
            else:
                ltf['htf_structure']=ltf['structure']
                ltf['htf_vol_regime']=ltf['vol_regime']

            ltf = generate_signals(ltf)
            ltf['conf_long']  = confidence(ltf,'long')
            ltf['conf_short'] = confidence(ltf,'short')

            ls=ltf['long_signal'].sum(); ss=ltf['short_signal'].sum()

            t_list, eq, final_cap = backtest(
                ltf, sym, capital=INIT_CAP,
                risk_pct=0.01, atr_k=1.0, partial_rr=2.0,
                trail_mult=1.0, max_dd=0.10, conf_thresh=0.5
            )
            m = metrics(t_list, eq, INIT_CAP)

            row={
                'Symbol':sym,'TF':tf,'Bars':len(ltf),
                'LongSig':int(ls),'ShortSig':int(ss),
                **m
            }
            results.append(row)
            print(f'  {sym:9s} {tf:4s}  bars={len(ltf):6d}  signals={ls+ss:4d}  '
                  f'trades={m["trades"]:3d}  wr={m["win_rate"]:.0%}  '
                  f'exp={m["expectancy"]:7.2f}  dd={m["max_dd"]:.1%}  '
                  f'ret={m["ret_pct"]:+.1f}%')

    # Print full table
    print('\n' + '='*70)
    print('  FULL RESULTS TABLE')
    print('='*70)
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        pd.set_option('display.max_columns',None,'display.width',200,'display.float_format','{:.3f}'.format)
        print(df_res.to_string(index=False))

        # Summary
        print('\n' + '='*70)
        print('  SUMMARY')
        print('='*70)
        print(f'  Total symbol/timeframe combos tested: {len(df_res)}')
        with_trades = df_res[df_res['trades']>0]
        if not with_trades.empty:
            print(f'  Combos with trades:      {len(with_trades)}')
            print(f'  Best return:             {with_trades["ret_pct"].max():+.2f}%  ({with_trades.loc[with_trades["ret_pct"].idxmax(),"Symbol"]} {with_trades.loc[with_trades["ret_pct"].idxmax(),"TF"]})')
            print(f'  Best win rate:           {with_trades["win_rate"].max():.0%}')
            print(f'  Best expectancy:         {with_trades["expectancy"].max():.2f}')
            print(f'  Avg win rate:            {with_trades["win_rate"].mean():.0%}')
            print(f'  Avg max drawdown:        {with_trades["max_dd"].mean():.1%}')
        print('='*70)

        # Save
        os.makedirs('results', exist_ok=True)
        df_res.to_csv('results/backtest_matrix.csv', index=False)
        print(f'\n  Full results saved → results/backtest_matrix.csv')
