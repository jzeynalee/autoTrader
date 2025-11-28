#run_backtest.py
import os
# Fix for Windows KMeans memory leak warning - MUST be set before sklearn import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from .backtest import WalkForwardValidator
from .features import WindowConfig

def generate_regime_data(n_bars=2000):
    """
    Generates a random walk where High/Low ranges EXPAND during high volatility.
    This ensures Parkinson Volatility features can detect the regimes.
    """
    np.random.seed(42)
    
    # Define Volatility Regimes (Standard Deviation)
    vol_schedule = np.zeros(n_bars)
    vol_schedule[:500] = 0.005      # Low Vol
    vol_schedule[500:1000] = 0.02   # High Vol
    vol_schedule[1000:1500] = 0.005 # Low Vol
    vol_schedule[1500:] = 0.04      # Extreme Vol
    
    # 1. Generate Returns & Prices
    returns = np.random.normal(0, vol_schedule, n_bars)
    price = 100 * np.exp(np.cumsum(returns))
    
    # 2. Generate High/Low based on CURRENT Volatility
    # In reality, H-L range is proportional to volatility
    # We add some random noise to the range itself so it's not perfect
    range_noise = np.random.uniform(0.8, 1.2, n_bars)
    # The 'true' range for this bar is roughly proportional to the vol_schedule
    bar_range = price * vol_schedule * range_noise * 2.0  # Multiplier to ensure range covers open/close drift
    
    # Ensure H/L encompass the Open/Close
    # Open for t is roughly Close for t-1 (simplified)
    open_p = np.roll(price, 1)
    open_p[0] = price[0]
    
    # High is max(Open, Close) + some upward wick
    # Low is min(Open, Close) - some downward wick
    max_oc = np.maximum(open_p, price)
    min_oc = np.minimum(open_p, price)
    
    high = max_oc + (bar_range * 0.5)
    low = min_oc - (bar_range * 0.5)
    
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='h')
    
    df = pd.DataFrame({
        'close': price,
        'open': open_p,
        'high': high,
        'low': low,
        'volume': np.random.rand(n_bars) * 1000 + (vol_schedule * 100000) # More vol = often more volume
    }, index=dates)
    
    return df

def main():
    # 1. Generate Synthetic Data
    print("Generating synthetic market data (Fixed Volatility Scaling)...")
    df = generate_regime_data()
    
    # 2. Configure Validator
    # Train on 300 bars, Predict forward 50 bars
    # Using 3 components to catch Low/High/Extreme
    validator = WalkForwardValidator(
        window_config=WindowConfig(window_size=32),
        train_window=300,
        test_window=50,
        n_components=3
    )
    
    # 3. Run Validation
    results = validator.run(df)
    
    # 4. Analyze
    print("\nanalysis running...")
    stats = validator.evaluate(results)
    
    # 5. Check Dwell Times (Stability)
    results['switch'] = results['regime'].diff().fillna(0) != 0
    switches = results['switch'].sum()
    print(f"\nTotal Regime Switches: {switches} over {len(results)} bars")
    print(f"Average Dwell Time: {len(results)/switches:.1f} bars")
    
    # Sanity Check for User
    regime_0_vol = stats.loc[0, 'Mean_Vol']
    regime_2_vol = stats.loc[2, 'Mean_Vol']
    
    if regime_2_vol > regime_0_vol * 1.5:
        print("\n[PASS] Strong separation detected: Regime 2 is >1.5x more volatile than Regime 0.")
    else:
        print("\n[FAIL] Separation is still weak. Check feature extraction logic.")

if __name__ == "__main__":
    main()