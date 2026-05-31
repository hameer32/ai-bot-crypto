#!/bin/bash
# Start paper trading on server
# Usage: bash start_paper_trading.sh

set -e
cd "$(dirname "$0")"

echo "=== Paper Trading Startup ==="
echo "Working dir: $(pwd)"

# Check models exist
if [ ! -f "models/ict_predictor.pkl" ]; then
    echo "ERROR: models/ict_predictor.pkl not found. Push models to server first."
    exit 1
fi
echo "✓ Models found"

# Create output dirs
mkdir -p results/paper_trading

# Kill any existing paper trader
pkill -f "live/run_paper.py" 2>/dev/null && echo "Stopped existing paper trader" || true
sleep 1

# Start paper trader in background
nohup python3 live/run_paper.py \
    > results/paper_trading/paper_trader.log 2>&1 &

PID=$!
echo "✓ Paper trader started (PID=$PID)"
echo "  Log:       tail -f results/paper_trading/paper_trader.log"
echo "  Dashboard: results/paper_trading/dashboard.html"
echo "  Stop:      kill $PID  OR  pkill -f run_paper.py"
echo ""
echo "Waiting 10s to confirm startup..."
sleep 10
if kill -0 $PID 2>/dev/null; then
    echo "✓ Still running. First cycle in progress."
    tail -20 results/paper_trading/paper_trader.log
else
    echo "✗ Process died. Check log:"
    cat results/paper_trading/paper_trader.log
fi
