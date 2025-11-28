# Run: python -m autoTrader.strategy.newML.backtest_strategy_class
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ...db_connector import DatabaseConnector
from .pipeline import RegimePipeline
from .features import FeatureRegistry, WindowConfig
from .strategy import RegimeAwareStrategy

class StrategyClassBacktester:
    def __init__(self, db_path: str, pair_tf: str, initial_balance: float = 10000.0):
        self.db = DatabaseConnector(db_path)
        self.pair_tf = pair_tf
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Performance Tracking
        self.equity_curve = []
        self.trades = []
        
    def load_and_prep(self):
        print(f"Loading data for {self.pair_tf}...")
        df = self.db.load_raw_ohlcv(self.pair_tf)
        if df is None or df.empty:
            raise ValueError(f"No data found for {self.pair_tf}")
        
        # Calculate ATR for stop loss logic (needed for position sizing simulation)
        # Simple ATR implementation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Add SMA for Trend Logic (to mimic production signal)
        df['sma_50'] = df['close'].rolling(50).mean()
        
        df.dropna(inplace=True)
        return df

    def run(self, train_ratio=0.3):
        df = self.load_and_prep()
        
        # 1. Split Data
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:].copy()
        
        # 2. Train Pipeline (Bootstrap)
        print("Bootstrapping ML Pipeline...")
        registry = FeatureRegistry()
        config = WindowConfig(window_size=32)
        X_train = []
        for i in range(config.window_size, len(df_train)):
            feat = registry.build_window(df_train, i, config)
            X_train.append(feat)
            
        pipeline = RegimePipeline(n_components=3, seed=42)
        pipeline.fit(np.array(X_train))
        
        # 3. Initialize Strategy Class
        print("Initializing RegimeAwareStrategy...")
        strategy = RegimeAwareStrategy(pipeline=pipeline)
        
        # 4. Pre-fill Strategy Buffer with Training Data
        # This prevents the 'warming_up' state at the start of the test
        print("Warming up strategy buffer...")
        warmup_data = df_train.iloc[-50:] # Feed last 50 bars
        for idx, row in warmup_data.iterrows():
            strategy.on_bar(idx, row['open'], row['high'], row['low'], row['close'], row['volume'])

        # 5. Run Event Loop
        print(f"Running Event-Driven Backtest on {len(df_test)} bars...")
        
        current_position = None # { 'entry': float, 'qty': float, 'stop': float }
        
        for i in tqdm(range(len(df_test))):
            bar = df_test.iloc[i]
            ts = df_test.index[i]
            
            # A. Update Strategy
            instruction = strategy.on_bar(ts, bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
            
            # B. Manage Existing Position
            if current_position:
                # Check Stop Loss
                if bar['low'] <= current_position['stop']:
                    # Stop Hit
                    exit_price = current_position['stop']
                    pnl = (exit_price - current_position['entry']) * current_position['qty']
                    self.balance += pnl
                    self.trades.append({'res': 'loss', 'pnl': pnl, 'regime': instruction.regime_name})
                    current_position = None
                
                # Check Regime Exit (Force Close if Bearish)
                elif not instruction.can_trade:
                    # ML says GET OUT
                    exit_price = bar['close']
                    pnl = (exit_price - current_position['entry']) * current_position['qty']
                    self.balance += pnl
                    self.trades.append({'res': 'regime_exit', 'pnl': pnl, 'regime': instruction.regime_name})
                    current_position = None
                    
                # Take Profit / Re-eval (Simplified: Close after 1 bar to isolate alpha, 
                # OR hold. Let's hold until Stop or Regime Exit for realism)
                else:
                    # Mark-to-Market Equity
                    unrealized_pnl = (bar['close'] - current_position['entry']) * current_position['qty']
                    self.equity_curve.append(self.balance + unrealized_pnl)
                    continue

            # C. Entry Logic
            # Only enter if flat
            if current_position is None:
                # Mock Signal: Trend Following (Close > SMA)
                trend_signal = bar['close'] > bar['sma_50']
                
                # SPECIAL RULE: If Regime is 'Aggressive' (2), ignore SMA and force entry
                if instruction.mode == 'aggressive':
                    trend_signal = True
                
                if trend_signal and instruction.can_trade:
                    # Calculate Stop Loss
                    # Use the multiplier from the instruction
                    stop_dist = bar['atr'] * instruction.stop_loss_multiplier
                    stop_price = bar['close'] - stop_dist
                    
                    # Calculate Position Size (Dynamic Risk)
                    qty = strategy.calculate_position_size(
                        balance=self.balance, 
                        entry=bar['close'], 
                        stop=stop_price, 
                        instr=instruction
                    )
                    
                    if qty > 0:
                        current_position = {
                            'entry': bar['close'],
                            'qty': qty,
                            'stop': stop_price
                        }
            
            self.equity_curve.append(self.balance)

        self._report()

    def _report(self):
        eq = np.array(self.equity_curve)
        
        # Metrics
        total_ret = (self.balance - self.initial_balance) / self.initial_balance * 100
        max_dd = 0
        peak = eq[0]
        for val in eq:
            if val > peak: peak = val
            dd = (val - peak) / peak
            if dd < max_dd: max_dd = dd
            
        wins = len([t for t in self.trades if t['pnl'] > 0])
        total = len(self.trades)
        win_rate = (wins/total * 100) if total > 0 else 0
        
        print("\n" + "="*40)
        print(" STRATEGY CLASS VERIFICATION RESULTS")
        print("="*40)
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Return:   {total_ret:.2f}%")
        print(f"Max Drawdown:   {max_dd*100:.2f}%")
        print(f"Total Trades:   {total}")
        print(f"Win Rate:       {win_rate:.2f}%")
        print("="*40)
        
        if total_ret > 0 and abs(max_dd) < 0.25:
             print("✅ PASS: Strategy logic is profitable and risk-controlled.")
        elif total_ret > 0:
             print("⚠️ PASS/WARN: Profitable but High Drawdown. Check risk params.")
        else:
             print("❌ FAIL: Strategy lost money.")

if __name__ == "__main__":
    DB_PATH = "./data/auto_trader_db.sqlite" 
    PAIR_TF = "btc_usdt_1h"
    
    bt = StrategyClassBacktester(DB_PATH, PAIR_TF)
    bt.run()