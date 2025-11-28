# Run: python -m autoTrader.strategy.newML.backtest_strategy_class
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from autoTrader.db_connector import DatabaseConnector
from autoTrader.strategy.newML.pipeline import RegimePipeline
from autoTrader.strategy.newML.strategy import RegimeAwareStrategy

class StrategyClassBacktester:
    def __init__(self, db_path: str, pair_tf: str, initial_balance: float = 10000.0):
        self.db = DatabaseConnector(db_path)
        self.pair_tf = pair_tf
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve = []
        self.trades = []
        
    def load_data(self):
        print(f"Loading FULL feature data for {self.pair_tf}...")
        df = self.db.load_full_features(self.pair_tf)
        if df is None or df.empty:
            raise ValueError(f"No data found for {self.pair_tf}")
        
        # Ensure standard columns exist
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # FIX: Only drop NaNs for columns strictly needed for the backtest
        # A blanket df.dropna() might wipe data if 'rsi' column exists in DB but is NULL
        df.dropna(subset=['close', 'sma_50', 'atr'], inplace=True)
        
        return df

    def run(self, train_ratio=0.3):
        df = self.load_data()
        
        if len(df) < 100:
            print("❌ Insufficient data to run backtest.")
            return

        # Split Data
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:].copy()
        
        if df_test.empty:
            print("❌ Test set is empty. Check train_ratio or data size.")
            return
        
        # Use existing model file if present, or None
        model_path = "./models/regime_pipeline_v1.pkl"
        
        print("Initializing RegimeAwareStrategy with DB Logic...")
        strategy = RegimeAwareStrategy(
            model_path=model_path, 
            db_path=self.db.db_path,
            pair_tf=self.pair_tf
        )
        
        # Warmup
        print("Warming up strategy buffer...")
        warmup = df_train.iloc[-50:]
        for idx, row in warmup.iterrows():
            feats = row.to_dict()
            strategy.on_bar(idx, row['open'], row['high'], row['low'], row['close'], row['volume'], extra_features=feats)

        print(f"Running Logic-Driven Backtest on {len(df_test)} bars...")
        
        current_position = None 
        
        for i in tqdm(range(len(df_test))):
            bar = df_test.iloc[i]
            ts = df_test.index[i]
            feats = bar.to_dict()
            
            # --- 1. GET STRATEGY INSTRUCTION ---
            instruction = strategy.on_bar(ts, bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'], extra_features=feats)
            
            # --- 2. EXIT LOGIC ---
            if current_position:
                # Stop Loss
                if bar['low'] <= current_position['stop']:
                    pnl = (current_position['stop'] - current_position['entry']) * current_position['qty']
                    self.balance += pnl
                    self.trades.append({'res': 'loss', 'pnl': pnl})
                    current_position = None
                
                # Logic Exit (If strategy flipped to Bearish)
                elif instruction.signal_type == 'BEARISH':
                    pnl = (bar['close'] - current_position['entry']) * current_position['qty']
                    self.balance += pnl
                    self.trades.append({'res': 'logic_exit', 'pnl': pnl})
                    current_position = None
                else:
                    self.equity_curve.append(self.balance)
                    continue

            # --- 3. ENTRY LOGIC ---
            if current_position is None:
                should_enter = False
                
                # NEW LOGIC: Use the formulated signal!
                if instruction.signal_type == 'BULLISH':
                    should_enter = True # Statistical Mining says BUY
                
                # Fallback: Conservative Trend
                elif instruction.mode == 'conservative' and bar['close'] > bar['sma_50']:
                    should_enter = True 
                
                if should_enter and instruction.can_trade:
                    stop_dist = bar['atr'] * instruction.stop_loss_multiplier
                    stop_price = bar['close'] - stop_dist
                    
                    qty = strategy.calculate_position_size(self.balance, bar['close'], stop_price, instruction)
                    
                    if qty > 0:
                        current_position = {'entry': bar['close'], 'qty': qty, 'stop': stop_price}
            
            self.equity_curve.append(self.balance)

        self._report()

    def _report(self):
        if not self.equity_curve:
            print("⚠️ No equity curve data generated (Loop didn't run or data empty).")
            return

        eq = np.array(self.equity_curve)
        total_ret = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        max_dd = 0
        peak = eq[0]
        for val in eq:
            if val > peak: peak = val
            if peak > 0:
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
        print(f"Win Rate:       {win_rate:.2f}% ({wins}/{total})")
        print("="*40)

if __name__ == "__main__":
    DB_PATH = "./data/auto_trader_db.sqlite" 
    PAIR_TF = "btc_usdt_1h"
    bt = StrategyClassBacktester(DB_PATH, PAIR_TF)
    bt.run()