#!/usr/bin/env python3
"""
Paper Trading Runner
====================
Runs both models every 5 minutes on live Binance data.
Generates dashboard.html after each cycle.

Usage:
  python live/run_paper.py              # run continuously
  python live/run_paper.py --once       # run one cycle then exit (for testing)
  python live/run_paper.py --dashboard  # just regenerate dashboard from existing trades

Dashboard:  results/paper_trading/dashboard.html
Trade logs: results/paper_trading/model_global_trades.jsonl
            results/paper_trading/model_crypto_trades.jsonl
"""
import sys, os, json, logging, argparse, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/paper_trading/paper_trader.log"),
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open("live/config.json") as f:
        return json.load(f)


def setup_strategies(cfg, raw_yaml):
    """Create strategies for each model config."""
    from config.config import load_config as load_app_cfg
    from strategies.registry import get_strategy

    strategies = {}
    for model_key, model_cfg in cfg["models"].items():
        app_cfg = load_app_cfg("config.yaml")
        # Inject the right plan_c_ict config for this model
        import yaml
        seg_cfg = dict(raw_yaml.get(model_cfg["config_key"], raw_yaml.get("plan_c_ict", {})))
        base = dict(raw_yaml.get("plan_c_ict", {}))
        base.update(seg_cfg)
        base["symbols"] = model_cfg["symbols"]
        base["training_symbols"] = []
        setattr(app_cfg, "plan_c_ict", base)

        strat = get_strategy("plan_c_ict", app_cfg)

        # Load ML model
        from ml.supervised import ICTTradePredictor
        pred = ICTTradePredictor()
        if pred.load(model_cfg["model_path"]):
            strat._ml_model = pred
            strat._ml_threshold = model_cfg["ml_threshold"]
            logger.info("[%s] ML model loaded (thr=%.2f)", model_key, model_cfg["ml_threshold"])
        else:
            logger.warning("[%s] ML model NOT found: %s", model_key, model_cfg["model_path"])

        strategies[model_key] = (strat, model_cfg)
    return strategies


def run_cycle(strategies, accounts, cfg, all_data_store):
    """One 5-minute cycle: fetch data → build signals → check entries → update positions → dashboard."""
    from live.data_feed import update_all_data, get_current_price
    from live.dashboard import generate_dashboard

    logger.info("=" * 60)
    logger.info("Starting cycle at %s", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

    # ── 1. Collect all symbols across both models ─────────────────────────────
    all_symbols = set()
    for _, model_cfg in strategies.values():
        all_symbols.update(model_cfg["symbols"])
    all_symbols = sorted(all_symbols)

    logger.info("Fetching live data for %d symbols...", len(all_symbols))
    all_data = update_all_data(
        symbols=list(all_symbols),
        timeframes=cfg["timeframes"],
        bars_per_tf=cfg["live_bars_per_tf"],
        existing_data=all_data_store,
    )
    all_data_store.update(all_data)

    # ── 2. Run each model ─────────────────────────────────────────────────────
    for model_key, (strat, model_cfg) in strategies.items():
        account = accounts[model_key]
        symbols = model_cfg["symbols"]
        ltf = "5m"

        logger.info("[%s] Building signals for %d symbols...", model_key, len(symbols))

        try:
            strat.prepare(all_data)
        except Exception as e:
            logger.error("[%s] prepare() failed: %s", model_key, e)
            continue

        signals_found = 0
        for sym in symbols:
            sig_df = strat._signals.get(sym)
            if sig_df is None or sig_df.empty:
                continue

            # Get the latest bar timestamp that's in our signal df
            latest_ts = sig_df.index[-1]
            latest_bar = sig_df.iloc[-1]

            try:
                setups = strat.on_bar(sym, latest_ts, latest_bar,
                                      capital=account.capital)
                for setup in setups:
                    placed = account.process_signal(setup)
                    if placed:
                        signals_found += 1
                        logger.info("[%s] Signal: %s %s @ %.6f (ML=%.3f)",
                                    model_key, setup.direction.upper(), sym,
                                    setup.entry, setup.meta.get("ml_proba", 0))
            except Exception as e:
                logger.warning("[%s] on_bar error for %s: %s", model_key, sym, e)

        logger.info("[%s] %d new signal(s) this cycle", model_key, signals_found)

    # ── 3. Update open positions with current prices ──────────────────────────
    current_prices = {}
    for sym in all_symbols:
        price = get_current_price(sym)
        if price > 0:
            current_prices[sym] = price

    for model_key, account in accounts.items():
        if account.open_positions:
            account.update_positions(current_prices)

    # ── 4. Generate HTML dashboard ────────────────────────────────────────────
    from live.dashboard import generate_dashboard
    generate_dashboard(accounts, cfg["dashboard_path"])
    logger.info("Dashboard updated → %s", cfg["dashboard_path"])

    # Print quick summary
    for model_key, account in accounts.items():
        s = account.stats()
        logger.info("[%s] Capital=$%.2f (%+.1f%%) | Open=%d | Trades=%d | WR=%.0f%%",
                    model_key, s["capital"], s["return_pct"],
                    s["open_trades"], s["total_trades"], s["win_rate"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",      action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dashboard", action="store_true", help="Regenerate dashboard only")
    args = parser.parse_args()

    os.makedirs("results/paper_trading", exist_ok=True)

    cfg = load_config()
    import yaml
    with open("config.yaml") as f:
        raw_yaml = yaml.safe_load(f)
    for key in ("plan_a_swing","plan_b_scalp","plan_c_ict"):
        if key in raw_yaml:
            pass  # keep as-is

    # ── Setup accounts ────────────────────────────────────────────────────────
    from live.paper_engine import PaperAccount
    accounts = {}
    for model_key, model_cfg in cfg["models"].items():
        accounts[model_key] = PaperAccount(
            model_name=f"model_{model_key}",
            starting_capital=model_cfg["starting_capital"],
            risk_pct=model_cfg["risk_pct"],
            trades_dir="results/paper_trading",
            max_dd_pct=cfg.get("max_drawdown_pct", 0.10),
        )

    if args.dashboard:
        from live.dashboard import generate_dashboard
        generate_dashboard(accounts, cfg["dashboard_path"])
        print(f"Dashboard regenerated → {cfg['dashboard_path']}")
        return

    # ── Setup strategies ──────────────────────────────────────────────────────
    strategies = setup_strategies(cfg, raw_yaml)
    all_data_store = {}

    logger.info("=" * 60)
    logger.info("PAPER TRADER STARTED")
    logger.info("Models: %s", list(strategies.keys()))
    logger.info("Poll interval: %d seconds", cfg["poll_interval_seconds"])
    logger.info("Dashboard: %s", cfg["dashboard_path"])
    logger.info("=" * 60)

    # Update config with start time
    cfg["started"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    if args.once:
        run_cycle(strategies, accounts, cfg, all_data_store)
        print(f"\n✅ Dashboard → {os.path.abspath(cfg['dashboard_path'])}")
        return

    # ── Continuous loop ───────────────────────────────────────────────────────
    while True:
        try:
            run_cycle(strategies, accounts, cfg, all_data_store)
        except KeyboardInterrupt:
            logger.info("Paper trader stopped by user.")
            break
        except Exception as e:
            logger.error("Cycle error: %s", e, exc_info=True)

        logger.info("Sleeping %d seconds until next cycle...", cfg["poll_interval_seconds"])
        time.sleep(cfg["poll_interval_seconds"])


if __name__ == "__main__":
    main()
