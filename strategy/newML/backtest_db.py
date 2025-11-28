#backtest_db.py
import os
# Fix for TensorFlow/OneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Fix for Windows KMeans memory leak
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from tqdm import tqdm
from ...db_connector import DatabaseConnector
from .pipeline import RegimePipeline
from .features import FeatureRegistry, WindowConfig
from .interpretation import InterpretationRegistry

class DatabaseBacktester:
    def __init__(self, db_path: str, pair_tf: str, train_ratio: float = 0.3):
        self.db = DatabaseConnector(db_path)
        self.pair_tf = pair_tf
        self.train_ratio = train_ratio
        self.feature_registry = FeatureRegistry()
        self.window_config = WindowConfig(window_size=32)
        self.interpreter = InterpretationRegistry()
        self.pipeline = None

    def load_data(self):
        print(f"Loading data for {self.pair_tf}...")
        df = self.db.load_raw_ohlcv(self.pair_tf)
        if df is None or df.empty:
            raise ValueError(f"No data found for {self.pair_tf}")
        
        # Baseline Trend Indicator
        df['sma_50'] = df['close'].rolling(50).mean()
        df.dropna(inplace=True)
        return df

    def train_model(self, df_train: pd.DataFrame):
        print(f"Training on {len(df_train)} bars...")
        X_train = []
        # Skip first window_size to avoid index errors
        for i in range(self.window_config.window_size, len(df_train)):
            feat = self.feature_registry.build_window(df_train, i, self.window_config)
            X_train.append(feat)
        
        self.pipeline = RegimePipeline(n_components=3, seed=42)
        self.pipeline.fit(np.array(X_train))

    def run_backtest(self):
        df = self.load_data()
        split_idx = int(len(df) * self.train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:].copy()
        
        self.train_model(df_train)
        
        print(f"Simulating strategy on {len(df_test)} bars...")
        
        regimes = []
        # Create buffer for feature extraction
        buffer = pd.concat([df_train.iloc[-self.window_config.window_size:], df_test])
        
        # We need to process test data bar by bar
        for i in tqdm(range(len(df_test))):
            # Calculate the index in the buffer corresponding to the current test bar
            current_buffer_idx = self.window_config.window_size + i
            try:
                feat = self.feature_registry.build_window(buffer, current_buffer_idx, self.window_config)
                pred = self.pipeline.predict(feat.reshape(1, -1))
                regimes.append(pred['regime'])
            except Exception as e:
                # Default to Chop if error
                regimes.append(0)

        df_test['regime'] = regimes
        self._calculate_metrics(df_test)
        
    def _calculate_metrics(self, df: pd.DataFrame):
        # Log Returns
        df['ret'] = np.log(df['close'].shift(-1) / df['close'])
        
        # --- Strategy A: Naive Trend ---
        # Long if Price > SMA 50
        df['signal_base'] = np.where(df['close'] > df['sma_50'], 1, 0)
        
        # --- Strategy B: Smart Regime ---
        # Logic derived from 'profile_regimes.py' findings:
        # 1. Regime 2 (High Vol): Positive Expectancy -> Aggressive Long
        # 2. Regime 1 (Med Vol): Negative Expectancy -> Cash (Exit)
        # 3. Regime 0 (Low Vol): Neutral -> Follow Trend (SMA)
        
        conditions = [
            (df['regime'] == 2), # Bull/Pump -> Force Long
            (df['regime'] == 1), # Bear/Dump -> Force Cash
            (df['regime'] == 0) & (df['close'] > df['sma_50']) # Chop -> SMA Trend
        ]
        choices = [1, 0, 1]
        
        df['signal_smart'] = np.select(conditions, choices, default=0)
        
        # Calculate PnL
        df['pnl_base'] = df['signal_base'] * df['ret']
        df['pnl_smart'] = df['signal_smart'] * df['ret']
        
        metrics = {}
        for strategy in ['base', 'smart']:
            pnl = df[f'pnl_{strategy}']
            total_ret = pnl.sum()
            
            # Max DD
            equity = pnl.cumsum()
            dd = equity - equity.cummax()
            max_dd = dd.min()
            
            # Win Rate
            trades = pnl[pnl != 0]
            wins = len(trades[trades > 0])
            total_trades = len(trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Profit Factor
            gross_win = pnl[pnl > 0].sum()
            gross_loss = abs(pnl[pnl < 0].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else 0
            
            metrics[strategy] = {
                'Total Return': round(total_ret, 4),
                'Max Drawdown': round(max_dd, 4),
                'Win Rate': round(win_rate, 2),
                'Profit Factor': round(pf, 2)
            }

        print("\n" + "="*40)
        print(f" FINAL VERIFICATION RESULTS ({self.pair_tf})")
        print("="*40)
        print(pd.DataFrame(metrics))
        print("\nNote: 'Smart' strategy ignores SMA in Regime 2 to catch early pumps, and hard-exits Regime 1.")

if __name__ == "__main__":
    # Ensure this matches your DB
    DB_PATH = "./data/auto_trader_db.sqlite" 
    PAIR_TF = "btc_usdt_1h"
    
    DatabaseBacktester(DB_PATH, PAIR_TF).run_backtest()