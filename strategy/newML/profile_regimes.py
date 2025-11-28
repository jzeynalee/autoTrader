# autoTrader.strategy.newML.profile_regimes.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from tqdm import tqdm
from ...db_connector import DatabaseConnector
from .pipeline import RegimePipeline
from .features import FeatureRegistry, WindowConfig

def profile_regimes():
    # CONFIG
    # Ensure this matches your directory structure
    DB_PATH = "./data/auto_trader_db.sqlite"
    PAIR_TF = "btc_usdt_1h"
    
    # 1. Load Data
    print(f"Loading {PAIR_TF}...")
    db = DatabaseConnector(DB_PATH)
    df = db.load_raw_ohlcv(PAIR_TF)
    if df is None or df.empty:
        print(f"❌ No data found for {PAIR_TF}. Please check the pair name in your DB.")
        return

    # 2. Train Model on ALL data (to analyze structure)
    print("Training extraction model on full history...")
    registry = FeatureRegistry()
    config = WindowConfig(window_size=32)
    
    X = []
    # Skip first window_size
    valid_indices = range(config.window_size, len(df))
    
    for i in tqdm(valid_indices):
        feat = registry.build_window(df, i, config)
        X.append(feat)
    
    X = np.array(X)
    
    pipeline = RegimePipeline(n_components=3, seed=42)
    pipeline.fit(X)
    
    # 3. Predict Regimes
    # We use the internal components directly for fast batch prediction
    print("Classifying history...")
    X_pca = pipeline.pca.transform(X)
    raw_labels = pipeline.gmm.predict(X_pca)
    aligned_labels = pipeline.aligner.transform(raw_labels)
    
    # 4. Attach to DataFrame
    # Align indices (df is longer than X by window_size)
    analysis_df = df.iloc[config.window_size:].copy()
    analysis_df['regime'] = aligned_labels
    
    # Calculate Forward Returns (1-bar and 4-bar)
    analysis_df['ret_1h'] = analysis_df['close'].pct_change().shift(-1)
    analysis_df['ret_4h'] = analysis_df['close'].pct_change(4).shift(-4)
    analysis_df['vol_range'] = (analysis_df['high'] - analysis_df['low']) / analysis_df['open']
    
    # 5. Generate Stat Card
    print("\n" + "="*50)
    print(f" REGIME IDENTITY CARD ({PAIR_TF})")
    print("="*50)
    
    stats = analysis_df.groupby('regime').agg({
        'ret_1h': ['mean', 'std', lambda x: (x > 0).mean()], # Mean Ret, Volatility, Win Rate
        'vol_range': 'mean',  # Average Candle Size
        'close': 'count'      # Frequency
    })
    
    stats.columns = ['Avg_Return', 'Std_Dev (Risk)', 'Bull_Prob', 'Avg_Candle_Size', 'Count']
    stats['Frequency %'] = (stats['Count'] / len(analysis_df) * 100).round(1)
    
    print(stats)
    print("\nINTERPRETATION GUIDE:")
    print("- High 'Bull_Prob' (>0.52)? -> BULLISH Regime (Go Long)")
    print("- Low 'Bull_Prob' (<0.48)? -> BEARISH Regime (Go Short/Cash)")
    print("- High 'Std_Dev'? -> VOLATILE (Widen Stops)")
    print("- Low 'Std_Dev'? -> QUIET (Tighten Stops)")

if __name__ == "__main__":
    profile_regimes()