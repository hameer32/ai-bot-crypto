# AI Bot Crypto

A multi-strategy algorithmic trading bot for crypto, Indian equities, commodities, and forex. Combines ICT (Inner Circle Trader) structure analysis with ML-based signal filtering.

**Active strategy:** Plan C ICT — multi-timeframe order blocks, FVG, CHoCH/BOS, killzones, and ML win-probability gate.

---

## Requirements

- Python 3.10+
- Binance account (for crypto live/paper trading — read-only API key)
- No broker API needed for backtesting

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/hameer32/ai-bot-crypto.git
cd ai-bot-crypto
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` file

Create a `.env` file in the root directory:

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret

# Optional — for LLM trade reports
ANTHROPIC_API_KEY=your_anthropic_key
```

> Binance API only needs **read** permissions. Never enable withdrawals.

---

## Data

Historical OHLCV data is cached in `cache/` as `.parquet` files.  
The repo ships with pre-fetched data — skip fetching if cache is already present.

To re-fetch or update:

```bash
# Fetch top-20 crypto + Gold/Silver + Indian indices & stocks
python scripts/fetch_universe.py

# Or fetch just top-10 crypto
python scripts/fetch_top10.py
```

---

## Running

### Paper Trading (recommended first step)

Runs the ICT + ML model on live Binance prices every 5 minutes. No real money involved.

```bash
bash start_paper_trading.sh
```

- Logs: `results/paper_trading/paper_trader.log`
- Dashboard: `results/paper_trading/dashboard.html` (open in browser)
- Stop: `pkill -f run_paper.py`

Or run a single cycle to test:

```bash
python live/run_paper.py --once
```

### Backtest

```bash
python scripts/run_backtest.py
```

Results saved to `results/`.

### Strategy signal test (quick check)

```bash
python scripts/run_strategy.py
```

### Optimize parameters

```bash
python scripts/run_optimize.py
```

---

## ML Models

Pre-trained models are included in `models/`:

| File | Segment |
|------|---------|
| `ict_predictor.pkl` | Global (all assets) |
| `ict_predictor_crypto.pkl` | Top-20 crypto |
| `ict_predictor_commodity.pkl` | Gold / Silver (COMEX) |
| `ict_predictor_india_equity.pkl` | NIFTY 50 stocks |
| `ict_predictor_india_index.pkl` | NIFTY / SENSEX indices |
| `ict_predictor_forex.pkl` | 20 Forex pairs |

To retrain models on your local cache:

```bash
# All segments
python scripts/run_ml_train_segments.py

# Single segment
python scripts/run_ml_train_segments.py --segment crypto
```

---

## Project Structure

```
├── strategies/
│   ├── plan_c_ict/       # Active strategy — ICT multi-TF
│   ├── plan_a_swing/     # Swing strategy
│   ├── plan_b_scalp/     # Scalp strategy
│   └── plan_d_hybrid/    # Hybrid ML strategy
├── live/                 # Paper trading engine + dashboard
├── backtest/             # Backtesting engine
├── ml/                   # Supervised, RL, LLM reporter
├── signals/              # Liquidity, market structure, order flow
├── indicators/           # ATR, Ichimoku, structure, volume
├── quant/                # Regime, momentum, mean reversion
├── risk/                 # Position sizer, drawdown guard
├── scripts/              # CLI entry points
├── cache/                # Cached OHLCV parquet files
├── models/               # Trained ML models (.pkl)
├── results/              # Backtest + paper trade results
└── config.yaml           # All strategy parameters
```

---

## Config

All parameters are in [config.yaml](config.yaml). Key settings:

```yaml
active_strategy: plan_c_ict

plan_c_ict:
  symbols: [BTCUSDT, ETHUSDT, BNBUSDT, ...]
  ltf: 5m
  tp1_rr: 4.0               # 1:4 risk/reward minimum
  ml_threshold: 0.42        # ML win-probability gate (0.0 = disabled)
  confidence_threshold: 0.45

risk:
  initial_capital: 10000
  risk_pct_per_trade: 0.01  # 1% risk per trade
  max_drawdown_pct: 0.1     # 10% max drawdown
```

---

## Supported Markets

| Market | Source | Timeframes |
|--------|--------|------------|
| Top-20 crypto (Binance) | ccxt | 5m, 15m, 30m, 1h, 4h, 1d, 1w |
| Gold / Silver (COMEX) | yfinance | 1h, 4h, 1d, 1w |
| NIFTY 50 stocks (NSE) | yfinance | 1h, 1d, 1w |
| NIFTY / SENSEX indices | yfinance | 1h, 1d, 1w |
| Forex (20 pairs) | yfinance | 1h, 1d |
