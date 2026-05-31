#!/usr/bin/env python3
"""
Universal Strategy Runner
==========================
Runs any registered strategy against cached historical data.

Usage:
  python scripts/run_strategy.py                          # uses active_strategy from config.yaml
  python scripts/run_strategy.py --strategy plan_a_swing
  python scripts/run_strategy.py --strategy plan_b_scalp
  python scripts/run_strategy.py --strategy original_smc
  python scripts/run_strategy.py --list                  # show all available strategies
  python scripts/run_strategy.py --compare               # run all strategies and compare

Options:
  --strategy   Strategy name (overrides config)
  --symbols    Comma-separated symbols e.g. BTCUSDT,ETHUSDT
  --start      Start date YYYY-MM-DD (uses first 50% of data by default)
  --end        End date YYYY-MM-DD
  --capital    Initial capital (default 10000)
  --risk       Risk per trade as decimal (default 0.01)
  --list       List available strategies
  --compare    Run all strategies and print side-by-side comparison
"""
import sys, os, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.config import load_config
from strategies.registry import get_strategy, list_strategies


INIT_CAP = 10_000.0


# ── Data loader ───────────────────────────────────────────────────────────────

def load_cache(cache_dir: str = "cache") -> dict[tuple[str, str], pd.DataFrame]:
    data = {}
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".parquet"):
            continue
        parts = fname.replace(".parquet", "").split("_")
        if len(parts) < 2:
            continue
        sym, tf = parts[0], parts[1]
        try:
            df = pd.read_parquet(os.path.join(cache_dir, fname))
            df.index = pd.to_datetime(df.index)
            df.index.name = "timestamp"
            data[(sym, tf)] = df
        except Exception:
            pass
    return data


def filter_dates(
    data: dict, start, end
) -> dict:
    if not start and not end:
        return data
    result = {}
    for key, df in data.items():
        if start:
            ts = pd.Timestamp(start)
            if df.index.tz is not None:
                ts = ts.tz_localize("UTC")
            df = df[df.index >= ts]
        if end:
            ts = pd.Timestamp(end)
            if df.index.tz is not None:
                ts = ts.tz_localize("UTC")
            df = df[df.index <= ts]
        if len(df) > 0:
            result[key] = df
    return result


# ── Backtest engine (strategy-agnostic) ──────────────────────────────────────

def run_backtest(
    strategy,
    all_data: dict,
    symbols: list[str],
    ltf: str,
    capital: float = INIT_CAP,
    risk_pct: float = 0.01,
    partial_rr: float = 4.0,
    trail_mult: float = 1.0,
    max_dd: float = 0.10,
) -> tuple[list[dict], list[tuple], float]:
    """
    Event-driven backtest loop using the BaseStrategy interface.

    Returns (trades, equity_curve, final_capital).
    """
    strategy.prepare(all_data)

    trades     = []
    equity     = []
    cap        = capital
    peak       = capital
    open_trade = None
    kill       = False
    trade_id   = 0

    # iterate the LTF timeframe for each symbol (multi-symbol support)
    # For simplicity run symbols sequentially; portfolio mode can be added later
    for sym in symbols:
        ltf_df = all_data.get((sym, ltf))
        if ltf_df is None or len(ltf_df) < 50:
            print(f"  SKIP {sym} {ltf} — no data")
            continue

        open_trade = None
        kill = False

        for ts, row in ltf_df.iterrows():
            if kill:
                break

            # update open trade
            if open_trade:
                hi, lo = float(row["high"]), float(row["low"])
                t = open_trade

                if t["dir"] == "long":
                    if lo <= t["sl"]:
                        pnl = (t["sl"] - t["ep"]) * t["sz"]
                        t["pnl"] += pnl; t["state"] = "SL"
                        cap += pnl; open_trade = None
                    elif not t["partial"] and hi >= t["tp1"]:
                        half = t["sz"] * 0.5
                        pnl  = (t["tp1"] - t["ep"]) * half
                        t["pnl"] += pnl; t["sz"] -= half
                        t["partial"] = True; t["trail"] = t["ep"]
                        cap += pnl
                    elif t["partial"] and t["trail"] is not None:
                        atr_v = t["atr"]
                        new_trail = lo - trail_mult * atr_v
                        if new_trail > t["trail"]:
                            t["trail"] = new_trail
                        if lo <= t["trail"]:
                            pnl = (t["trail"] - t["ep"]) * t["sz"]
                            t["pnl"] += pnl; t["state"] = "TRAIL"
                            cap += pnl; open_trade = None
                    # TP2 check
                    if open_trade and t.get("tp2") and hi >= t["tp2"] and t["partial"]:
                        pnl = (t["tp2"] - t["ep"]) * t["sz"]
                        t["pnl"] += pnl; t["state"] = "TP2"
                        cap += pnl; open_trade = None

                else:  # short
                    if hi >= t["sl"]:
                        pnl = (t["ep"] - t["sl"]) * t["sz"]
                        t["pnl"] += pnl; t["state"] = "SL"
                        cap += pnl; open_trade = None
                    elif not t["partial"] and lo <= t["tp1"]:
                        half = t["sz"] * 0.5
                        pnl  = (t["ep"] - t["tp1"]) * half
                        t["pnl"] += pnl; t["sz"] -= half
                        t["partial"] = True; t["trail"] = t["ep"]
                        cap += pnl
                    elif t["partial"] and t["trail"] is not None:
                        atr_v = t["atr"]
                        new_trail = hi + trail_mult * atr_v
                        if new_trail < t["trail"]:
                            t["trail"] = new_trail
                        if hi >= t["trail"]:
                            pnl = (t["ep"] - t["trail"]) * t["sz"]
                            t["pnl"] += pnl; t["state"] = "TRAIL"
                            cap += pnl; open_trade = None
                    if open_trade and t.get("tp2") and lo <= t["tp2"] and t["partial"]:
                        pnl = (t["ep"] - t["tp2"]) * t["sz"]
                        t["pnl"] += pnl; t["state"] = "TP2"
                        cap += pnl; open_trade = None

            peak = max(peak, cap)
            if (peak - cap) / peak >= max_dd:
                kill = True
                break

            equity.append((ts, cap))
            if open_trade:
                continue

            # ask strategy for signals
            setups = strategy.on_bar(sym, ts, row, cap)
            if not setups:
                continue

            # take first setup (highest confidence)
            setup = max(setups, key=lambda s: s.confidence)
            ep  = setup.entry
            sl  = setup.sl
            atr = setup.atr if setup.atr > 0 else ep * 0.002

            risk_unit = abs(ep - sl)
            if risk_unit <= 0:
                continue

            sz = (cap * risk_pct) / risk_unit
            trade_id += 1
            open_trade = {
                "id": trade_id, "sym": sym, "dir": setup.direction,
                "ep": ep, "sl": sl, "tp1": setup.tp1, "tp2": setup.tp2,
                "sz": sz, "atr": atr, "pnl": 0.0,
                "partial": False, "trail": None, "state": None,
                "ts": ts, "conf": setup.confidence,
                "meta": setup.meta,
            }
            trades.append(open_trade)

        # close any open trade at EOD
        if open_trade and open_trade.get("state") is None and len(ltf_df):
            lp = float(ltf_df["close"].iloc[-1])
            if open_trade["dir"] == "long":
                pnl = (lp - open_trade["ep"]) * open_trade["sz"]
            else:
                pnl = (open_trade["ep"] - lp) * open_trade["sz"]
            open_trade["pnl"] += pnl
            open_trade["state"] = "EOD"
            cap += pnl

    return trades, equity, cap


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(trades, equity, init_cap):
    closed = [t for t in trades if t.get("state")]
    if not closed:
        return {"trades": 0, "win_rate": 0, "avg_rr": 0, "expectancy": 0,
                "max_dd": 0, "pf": 0, "final": init_cap, "ret_pct": 0}

    pnls   = [t["pnl"] for t in closed]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr     = len(wins) / len(closed)
    aw     = np.mean(wins) if wins else 0
    al     = np.mean(losses) if losses else 0
    exp    = (wr * aw) + ((1 - wr) * al)
    gp     = sum(wins)
    gl     = abs(sum(losses))
    pf     = gp / gl if gl > 0 else float("inf")

    rr_list = []
    for t in closed:
        risk = abs(t["ep"] - t["sl"]) * max(t["sz"], 1e-6)
        if risk > 0:
            rr_list.append(t["pnl"] / risk)
    avg_rr = np.mean(rr_list) if rr_list else 0

    eq_vals = [v for _, v in equity]
    max_dd = 0.0
    peak = eq_vals[0] if eq_vals else init_cap
    for v in eq_vals:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    final = eq_vals[-1] if eq_vals else init_cap
    return {
        "trades": len(closed), "win_rate": wr, "avg_rr": avg_rr,
        "expectancy": exp, "max_dd": max_dd, "pf": pf,
        "final": final, "ret_pct": (final - init_cap) / init_cap * 100,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_metrics(name: str, m: dict, init_cap: float):
    print(f"\n  ── {name} ─────────────────────────────────────────────")
    print(f"  Trades     : {m['trades']}")
    print(f"  Win Rate   : {m['win_rate']:.0%}")
    print(f"  Avg RR     : {m['avg_rr']:+.2f}")
    print(f"  Expectancy : {m['expectancy']:+.1f}")
    print(f"  Max DD     : {m['max_dd']:.1%}")
    print(f"  Prof Factor: {m['pf']:.2f}")
    print(f"  Return     : {m['ret_pct']:+.1f}%  (${m['final']:,.0f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal Strategy Runner")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--symbols",  default=None)
    parser.add_argument("--start",    default=None)
    parser.add_argument("--end",      default=None)
    parser.add_argument("--capital",  type=float, default=INIT_CAP)
    parser.add_argument("--risk",     type=float, default=0.01)
    parser.add_argument("--list",     action="store_true")
    parser.add_argument("--compare",  action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable strategies:")
        for s in list_strategies():
            print(f"  {s}")
        return

    cfg = load_config("config.yaml")

    # attach strategy config blocks as attributes if present
    import yaml
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    for key in ("plan_a_swing", "plan_b_scalp", "plan_c_ict"):
        if key in raw:
            setattr(cfg, key, raw[key])

    symbols = args.symbols.split(",") if args.symbols else cfg.symbols
    # plan_c_ict ltf is configurable (5m or 15m); others are fixed
    plan_c_ltf = raw.get("plan_c_ict", {}).get("ltf", "15m")
    ltf_map = {"plan_a_swing": "15m", "plan_b_scalp": "5m",
               "plan_c_ict": plan_c_ltf, "original_smc": cfg.ltf}

    all_data = load_cache()
    all_data = filter_dates(all_data, args.start, args.end)
    print(f"\n  Loaded {len(all_data)} cached datasets")

    strategies_to_run = list_strategies() if args.compare else [
        args.strategy or raw.get("active_strategy", "original_smc")
    ]

    print("\n" + "=" * 70)
    print(f"  STRATEGY BACKTEST COMPARISON" if args.compare else f"  STRATEGY BACKTEST")
    print("=" * 70)

    results = {}
    for strat_name in strategies_to_run:
        print(f"\n  Running: {strat_name} ...")
        try:
            strat = get_strategy(strat_name, cfg)
            ltf   = ltf_map.get(strat_name, cfg.ltf)
            trades, equity, final = run_backtest(
                strat, all_data, symbols, ltf,
                capital=args.capital,
                risk_pct=args.risk,
                max_dd=cfg.risk.max_drawdown_pct,
            )
            m = compute_metrics(trades, equity, args.capital)
            results[strat_name] = m
            print_metrics(strat_name, m, args.capital)
        except Exception as e:
            import traceback
            print(f"  ERROR in {strat_name}: {e}")
            traceback.print_exc()

    if args.compare and results:
        print("\n" + "=" * 70)
        print("  SIDE-BY-SIDE COMPARISON")
        print("=" * 70)
        print(f"\n  {'Strategy':<20} {'Trades':>7} {'WinR':>6} {'AvgRR':>7} "
              f"{'Expect':>8} {'MaxDD':>7} {'Return':>8}")
        print("  " + "─" * 68)
        for name, m in sorted(results.items(), key=lambda x: x[1]["expectancy"], reverse=True):
            print(f"  {name:<20} {m['trades']:>7} {m['win_rate']:>5.0%} "
                  f"{m['avg_rr']:>+7.2f} {m['expectancy']:>+8.1f} "
                  f"{m['max_dd']:>6.1%} {m['ret_pct']:>+7.1f}%")

    os.makedirs("results", exist_ok=True)
    if results:
        pd.DataFrame(results).T.to_csv("results/strategy_comparison.csv")
        print(f"\n  Results saved → results/strategy_comparison.csv")


if __name__ == "__main__":
    main()
