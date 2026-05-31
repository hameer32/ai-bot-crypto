#!/usr/bin/env python3
"""
Segment ML Trainer
==================
Trains four specialised ICT models, each optimised for its asset class:

  1. crypto       — Top-50 crypto (Binance), 5m LTF, London/NY killzones
  2. india_index  — NIFTY 50 / NIFTY 200 / SENSEX / NIFTY Bank, 1h LTF
  3. india_equity — Top-50 NIFTY constituent stocks, 1h LTF
  4. commodity    — Gold, Silver, Oil, NatGas, Copper, Platinum, 1h LTF

For 1h-LTF segments, the T2 zone hierarchy is shifted up:
  T1 : Weekly + Daily bias
  T2 : Daily (as h4_df) + 4H (as h1_df)  ← shifted vs standard 4H+1H
  T3 : 1h execution bars (CHoCH, OB precision)

The global model (models/ict_predictor.pkl) is preserved untouched.

Usage:
  python scripts/run_ml_train_segments.py               # all 4 segments
  python scripts/run_ml_train_segments.py --segment crypto
  python scripts/run_ml_train_segments.py --segment commodity --trials 30
  python scripts/run_ml_train_segments.py --skip-backtest

Model outputs:
  models/ict_predictor_crypto.pkl
  models/ict_predictor_india_index.pkl
  models/ict_predictor_india_equity.pkl
  models/ict_predictor_commodity.pkl
"""
import sys, os, argparse, subprocess, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Segment Definitions ────────────────────────────────────────────────────────

SEGMENTS = {

    "crypto": {
        "desc":       "Top-50+ Binance crypto (5m execution, 2-year history)",
        "plan_key":   "plan_c_ict_crypto",
        "symbol_filter": "usdt",
        "min_signals":   500,
        "trade_symbols": [
            "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
            "DOGEUSDT","SUIUSDT","NEARUSDT","ONDOUSDT","ZECUSDT",
        ],
    },

    "india_index": {
        "desc":       "Indian Indices — NIFTY50, NIFTY200, SENSEX, NIFTY Bank (1h LTF, IST killzone)",
        "plan_key":   "plan_c_ict_india_index",
        "symbols":    ["NSEI","BSESN","CNX200","NSEBANK"],
        "min_signals": 50,
        "trade_symbols": ["NSEI","BSESN","NSEBANK"],
    },

    "india_equity": {
        "desc":       "NIFTY 50 constituent stocks — 49 NSE stocks (1h LTF, IST killzone)",
        "plan_key":   "plan_c_ict_india_equity",
        "symbol_filter": "nse",
        "min_signals":  2000,
        "trade_symbols": [
            "RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS",
            "BHARTIARTL","KOTAKBANK","LT","AXISBANK","SBIN",
        ],
    },

    "commodity": {
        "desc":       "10 COMEX/NYMEX/ICE Futures — Gold,Silver,Oil,NatGas,Copper,Platinum,Palladium,Cocoa,Coffee,Wheat (1h LTF)",
        "plan_key":   "plan_c_ict_commodity",
        "symbols":    ["GOLD","SILVER","OIL","NATGAS","COPPER","PLATINUM","PALLADIUM","COCOA","COFFEE","WHEAT"],
        "min_signals": 2000,
        "trade_symbols": ["GOLD","SILVER","OIL","COPPER","NATGAS"],
    },

    "forex": {
        "desc":       "20 Forex pairs — majors + crosses (1h LTF, London/NY killzones)",
        "plan_key":   "plan_c_ict_forex",
        "symbols":    ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
                       "EURGBP","EURJPY","GBPJPY","EURAUD","EURCAD","GBPAUD","GBPCAD",
                       "AUDCAD","AUDNZD","NZDJPY","CADJPY","CHFJPY","AUDCHF"],
        "min_signals": 2000,
        "trade_symbols": ["EURUSD","GBPUSD","USDJPY","EURJPY","GBPJPY"],
    },
}

# ── Symbol helpers ─────────────────────────────────────────────────────────────

NSE_STOCKS = {
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJAJFINSV","BAJFINANCE","BHARTIARTL","BPCL","BRITANNIA","CIPLA",
    "COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
    "INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","MM",
    "MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBICARD",
    "SBIN","SHRIRAMFIN","SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL",
    "TCS","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO",
}

INDIA_INDICES = {"NSEI", "BSESN", "CNX200", "NSEBANK"}
COMMODITIES   = {"GOLD", "SILVER", "OIL", "NATGAS", "COPPER", "PLATINUM"}


def resolve_symbols(seg_cfg: dict, available: list[str], ltf: str) -> list[str]:
    """Return the symbols for this segment that have all required TFs in cache."""
    import glob

    # 15m and 30m are available for full 730-day history on Binance
    # For yfinance assets, 30m/15m limited to 59 days but 1h/4h/1d/1w are long
    # Require only the stable long-history TFs for symbol eligibility
    stable_req = {"1w", "1d", "4h", "1h"}
    required_tfs = stable_req | ({ltf} if ltf not in ("30m", "15m") else set())
    cached_by_sym: dict[str, set] = {}
    for f in glob.glob("cache/*.parquet"):
        name = os.path.basename(f).replace(".parquet", "")
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            sym, tf = parts
            cached_by_sym.setdefault(sym, set()).add(tf)

    def has_all_tfs(sym):
        return required_tfs.issubset(cached_by_sym.get(sym, set()))

    # Explicit list
    if "symbols" in seg_cfg:
        return [s for s in seg_cfg["symbols"] if s in available and has_all_tfs(s)]

    # Filter-based
    filt = seg_cfg.get("symbol_filter", "")
    if filt == "usdt":
        return [s for s in available if s.endswith("USDT") and has_all_tfs(s)]
    elif filt == "nse":
        return [s for s in available if s in NSE_STOCKS and has_all_tfs(s)]

    return [s for s in available if has_all_tfs(s)]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_segment(seg_name: str, seg_cfg: dict, args) -> bool:
    """Run training for one segment by calling run_ml_train.py as a subprocess."""
    import glob

    print(f"\n{'═'*70}")
    print(f"  SEGMENT: {seg_name.upper()} — {seg_cfg['desc']}")
    print(f"{'═'*70}")

    # Resolve which symbols have the data
    all_available = sorted({
        os.path.basename(f).replace(".parquet", "").rsplit("_", 1)[0]
        for f in glob.glob("cache/*.parquet")
    })
    # Get LTF from plan_key config or fall back to segment default
    plan_key = seg_cfg.get("plan_key", "")
    ltf = "5m"   # default
    if plan_key:
        import yaml
        with open("config.yaml") as _f:
            _raw = yaml.safe_load(_f)
        if plan_key in _raw:
            ltf = _raw[plan_key].get("ltf", "5m")

    symbols = resolve_symbols(seg_cfg, all_available, ltf)

    if not symbols:
        print(f"  SKIP — no symbols with all required TFs for {seg_name}")
        return False

    model_path = _raw.get(plan_key, {}).get("model_path", f"models/ict_predictor_{seg_name}.pkl") if plan_key else f"models/ict_predictor_{seg_name}.pkl"
    print(f"  Symbols ({len(symbols)}): {symbols}")
    min_sig = seg_cfg.get("min_signals", 100)
    print(f"  LTF: {ltf}  |  Target: ≥{min_sig} signals  |  Model: {model_path}")

    # Build subprocess command — use --plan-key to load segment-specific config
    plan_key = seg_cfg.get("plan_key")
    cmd = [
        sys.executable, "scripts/run_ml_train.py",
        "--training-symbols", ",".join(symbols),
        "--segment",    seg_name,
        "--trials",     str(args.trials),
    ]
    if plan_key:
        cmd += ["--plan-key", plan_key]
    if args.threshold:
        cmd += ["--threshold", str(args.threshold)]
    if args.skip_backtest:
        cmd += ["--no-save"]
    if args.dry_run:
        print(f"  DRY RUN — would run: {' '.join(cmd)}")
        return True

    os.makedirs("models", exist_ok=True)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train all 4 segment ICT models")
    parser.add_argument("--segment",    default=None,
                        choices=list(SEGMENTS.keys()),
                        help="Train only this segment (default: all)")
    parser.add_argument("--trials",     type=int, default=50)
    parser.add_argument("--threshold",  type=float, default=0.42)
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip end-of-training backtest comparison (faster)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    segs_to_run = ([args.segment] if args.segment
                   else list(SEGMENTS.keys()))

    results = {}
    for seg in segs_to_run:
        ok = run_segment(seg, SEGMENTS[seg], args)
        results[seg] = "✓" if ok else "✗"

    print(f"\n{'═'*70}")
    print(f"  SEGMENT TRAINING SUMMARY")
    print(f"{'═'*70}")
    for seg, status in results.items():
        plan_key = SEGMENTS[seg].get("plan_key", "")
        try:
            import yaml
            with open("config.yaml") as _f:
                _raw = yaml.safe_load(_f)
            model = _raw.get(plan_key, {}).get("model_path", f"models/ict_predictor_{seg}.pkl")
        except Exception:
            model = f"models/ict_predictor_{seg}.pkl"
        exists = os.path.exists(model)
        print(f"  {status}  {seg:<16} → {model}  {'[saved]' if exists else '[missing]'}")
    print(f"\n  Global model preserved: models/ict_predictor.pkl")


if __name__ == "__main__":
    main()
