#backtest.py
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from .features import FeatureRegistry, WindowConfig
from .pipeline import RegimePipeline

class WalkForwardValidator:
    def __init__(self, 
                 window_config: WindowConfig, 
                 train_window: int = 500, 
                 test_window: int = 100,
                 n_components: int = 3):
        """
        Args:
            train_window: Number of bars to train the GMM on.
            test_window: Number of bars to predict forward before retraining.
        """
        self.feature_registry = FeatureRegistry()
        self.config = window_config
        self.train_window = train_window
        self.test_window = test_window
        self.n_components = n_components

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the Walk-Forward Validation.
        """
        results = []
        n_bars = len(df)
        
        # We need enough data for the first training set + lookback
        start_index = self.train_window + self.config.window_size
        
        if n_bars < start_index + self.test_window:
            raise ValueError(f"Not enough data. Need at least {start_index + self.test_window} bars.")

        print(f"Starting Walk-Forward Analysis on {n_bars} bars...")
        
        # Stepping through data
        for i in tqdm(range(start_index, n_bars, self.test_window)):
            # 1. Define Training Slice (Rolling Window)
            # Train on [i - train_window : i]
            train_start = i - self.train_window
            train_end = i
            
            # 2. Build Training Features
            X_train = self._build_batch_features(df, train_start, train_end)
            
            # 3. Fit Pipeline
            # We create a new pipeline instance every step to simulate "Retraining"
            # This tests if the Aligner correctly maps regimes despite random initialization.
            pipeline = RegimePipeline(n_components=self.n_components, seed=42 + i)
            pipeline.fit(X_train)
            
            # 4. Predict on Test Slice (Forward)
            test_end = min(i + self.test_window, n_bars)
            
            # Iterate through test window bar-by-bar (simulating live feed)
            # Optimization: In real backtest, we might vectorise, but loop is safer for logic check
            for t in range(i, test_end):
                # Build feature for time t
                # Note: build_window slices strictly up to t (inclusive)
                feat_vector = self.feature_registry.build_window(df, t, self.config)
                
                # Predict
                pred = pipeline.predict(feat_vector.reshape(1, -1))
                
                # Calculate Forward Metrics (Target)
                # We want to see if Regime predicted at T predicts Volatility at T+1
                if t + 1 < n_bars:
                    next_close = df['close'].iloc[t+1]
                    curr_close = df['close'].iloc[t]
                    fwd_ret = np.log(next_close / curr_close)
                    
                    # Store Result
                    results.append({
                        'timestamp': df.index[t],
                        'regime': pred['regime'],
                        'confidence': pred['confidence'],
                        'fwd_ret': fwd_ret,
                        'abs_fwd_ret': abs(fwd_ret) # Proxy for realized volatility
                    })
                    
        return pd.DataFrame(results)

    def _build_batch_features(self, df, start_idx, end_idx):
        """Helper to build a batch of features for training."""
        features = []
        # For training, we need valid windows. 
        # The build_window function looks back `config.window_size`.
        # So we iterate from start_idx to end_idx.
        for t in range(start_idx, end_idx):
            feat = self.feature_registry.build_window(df, t, self.config)
            features.append(feat)
        return np.array(features)

    def evaluate(self, results_df: pd.DataFrame):
        """
        Generates performance report of the regimes.
        """
        print("\n--- Regime Performance Report ---")
        
        # Group by Regime
        stats = results_df.groupby('regime').agg({
            'abs_fwd_ret': ['mean', 'count'], # Mean Forward Volatility
            'fwd_ret': 'mean',                # Mean Directional Drift
            'confidence': 'mean'              # Average Confidence
        })
        
        stats.columns = ['Mean_Vol', 'Count', 'Mean_Return', 'Avg_Conf']
        
        # Calculate Information Coefficient (IC)
        # Correlation between Regime ID and Absolute Forward Return
        # If Alignment works, correlation should be Positive (Higher ID = Higher Vol)
        ic = results_df['regime'].corr(results_df['abs_fwd_ret'])
        
        print(stats)
        print(f"\nInformation Coefficient (Regime vs Vol): {ic:.4f}")
        
        if ic > 0.1:
            print("SUCCESS: Positive correlation implies correct Regime Alignment (High ID = High Vol).")
        else:
            print("WARNING: Low or Negative correlation. Alignment might be failing or signal is weak.")
            
        return stats