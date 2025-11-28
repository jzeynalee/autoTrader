# train.py
import os
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm

from ...db_connector import DatabaseConnector
from .features import FeatureRegistry, WindowConfig
from .pipeline import RegimePipeline

def train_and_save():
    # CONFIGURATION
    DB_PATH = "./data/auto_trader_db.sqlite"
    PAIR_TF = "btc_usdt_1h" # Training Pair
    MODEL_PATH = "./models/regime_pipeline_v1.pkl"
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print(f"1. Loading Data for {PAIR_TF}...")
    db = DatabaseConnector(DB_PATH)
    df = db.load_raw_ohlcv(PAIR_TF)
    
    if df is None or df.empty:
        print("❌ No data found. Cannot train.")
        return

    print("2. Building Feature Set...")
    registry = FeatureRegistry()
    config = WindowConfig(window_size=32)
    
    X = []
    # Skip the first 'window_size' bars
    valid_indices = range(config.window_size, len(df))
    
    for i in tqdm(valid_indices):
        feat = registry.build_window(df, i, config)
        X.append(feat)
        
    X_train = np.array(X)
    
    print(f"3. Training ML Pipeline on {len(X_train)} samples...")
    # 3 Components: 0=Chop, 1=Bear, 2=Bull
    pipeline = RegimePipeline(n_components=3, seed=42) 
    pipeline.fit(X_train)
    
    print(f"4. Saving Model to {MODEL_PATH}...")
    pipeline.save(MODEL_PATH)
    print("✅ Training Complete.")

if __name__ == "__main__":
    train_and_save()