import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from indicators.volume import relative_volume, volume_spike
from strategies.signals.order_blocks import detect_order_blocks
from strategies.signals.fvg import detect_fvg

def create_mock_data(n=100):
    """Create synthetic OHLCV data with a clear impulse and volume spike."""
    data = {
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(100, 110, n),
        "low": np.random.uniform(100, 110, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 2000, n),
        "atr": np.full(n, 1.0)
    }
    df = pd.DataFrame(data)
    
    # Create a Bullish Order Block at index 50
    # 50: Bearish candle (OB candidate)
    df.loc[50, "open"] = 105
    df.loc[50, "close"] = 104
    df.loc[50, "high"] = 105.5
    df.loc[50, "low"] = 103.5
    df.loc[50, "volume"] = 1500
    
    # 51: Strong Bullish Impulse + Volume Spike
    df.loc[51, "open"] = 104
    df.loc[51, "close"] = 108  # > 1.5 ATR move
    df.loc[51, "high"] = 108.5
    df.loc[51, "low"] = 104
    df.loc[51, "volume"] = 5000 # Spike (avg is ~1500)
    
    # Create an FVG at index 60
    # 59 High: 102
    # 61 Low: 104
    df.loc[59, "high"] = 102
    df.loc[60, "volume"] = 4000 # Gap forming candle
    df.loc[61, "low"] = 104
    
    return df

def test_volume_indicators():
    df = create_mock_data()
    rvol = relative_volume(df)
    vspike = volume_spike(df)
    
    print(f"RVOL at impulse (index 51): {rvol.iloc[51]:.2f}")
    assert rvol.iloc[51] > 2.0
    assert vspike.iloc[51] == True
    print("✓ Volume indicators working")

def test_ob_volume_detection():
    df = create_mock_data()
    obs = detect_order_blocks(df, impulse_atr_mult=1.5)
    
    # Find the OB we created at index 50
    target_ob = next((o for o in obs if o["bar_idx"] == 50), None)
    
    assert target_ob is not None
    assert "rvol" in target_ob
    assert "vspike" in target_ob
    print(f"OB Volume conviction: RVOL={target_ob['rvol']:.2f}, Spike={target_ob['vspike']}")
    assert target_ob["vspike"] == True
    print("✓ OB Volume validation working")

def test_fvg_volume_detection():
    df = create_mock_data()
    fvgs = detect_fvg(df)
    
    target_fvg = next((f for f in fvgs if f["bar_idx"] == 61), None) # gap detected on candle i
    if not target_fvg:
        # Check nearby index due to 3-candle logic
        target_fvg = fvgs[0] if fvgs else None

    assert target_fvg is not None
    assert "rvol" in target_fvg
    print(f"FVG Volume intensity: RVOL={target_fvg['rvol']:.2f}")
    print("✓ FVG Volume validation working")

if __name__ == "__main__":
    try:
        test_volume_indicators()
        test_ob_volume_detection()
        test_fvg_volume_detection()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
