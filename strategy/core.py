import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict
import json
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from .analysis_trend import TrendAnalysisSystem
from .analysis_pullback import PullbackAnalysisSystem
from .analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .analysis_confluence_scoring import ConfluenceScoringSystem

try:
    from .discovery_mapping import (
        map_indicator_state,
        BULLISH_PATTERNS,
        BEARISH_PATTERNS,
        NEUTRAL_PATTERNS,
        CHART_BULLISH,
        CHART_BEARISH,
        CHART_NEUTRAL,
        TIMEFRAME_GROUPS,
        TIMEFRAME_HIERARCHY,
        validate_timeframe_group
    )
    MAPPER_AVAILABLE = True
except ImportError:
    # CRITICAL: Define DUMMY placeholders to prevent NameErrors in core.py methods
    # We exit gracefully later in strategy_main.py, but core.py must not crash its imports.
    def map_indicator_state(*args, **kwargs):
        raise NotImplementedError("map_indicator_state not available. Check discovery_mapping import.")

    # Define minimal placeholders for constants needed by other core methods (e.g., categorizing)
    BULLISH_PATTERNS = set()
    BEARISH_PATTERNS = set()
    NEUTRAL_PATTERNS = set()
    CHART_BULLISH = set()
    CHART_BEARISH = set()
    CHART_NEUTRAL = set()
    TIMEFRAME_GROUPS = {} # Must exist for StrategyDiscoverySystem to run
    
    MAPPER_AVAILABLE = False
    # Do NOT exit(1) here. Let strategy_main.py handle the graceful exit.
    print("FATAL WARNING: discovery_mapping.py failed to load. Strategy core will lack crucial constant definitions.")

class StrategyDiscoverySystem:
    """
    Discovers trading strategies by analyzing correlations between indicator states,
    candlestick patterns, chart patterns and price movements.
    Reports ALL confirming signals when a trend is detected.
    """

    # --- Global strict timeframe order ---
    TIMEFRAME_PRIORITY = {"4h": 1, "1h": 2, "15m": 3, "5m": 4, "1m": 5}
    
    def __init__(self, data_dir='lbank_data', lookforward_periods=5, price_threshold=0.005, n_jobs=-1):
        """
        Initialize the strategy discovery system with enhanced trend and pullback analysis.
        
        Args:
            data_dir: Directory containing CSV files from calculator.py
            lookforward_periods: Number of periods to look forward for price movement
            price_threshold: Minimum price change % to consider bullish/bearish (default 2%)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.data_dir = data_dir
        self.lookforward_periods = lookforward_periods
        self.price_threshold = price_threshold
        self.strategy_pool = {}
        self.all_dataframes = {}
        self.state_cache = {} 
        self.use_categorical_encoding = True
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

        # Cache attributes
        self.structure_cache = {}
        self.swing_cache = {}

        self.pattern_cache = {}
        self.alignment_cache = {}
        self.state_cache = {}
        self.max_cache_size = 1000
            
        # Phase 6: Performance Optimization Caches
        self.structure_cache = {}
        self.swing_cache = {}
        self.pattern_cache = {}
        self.alignment_cache = {}
        
        # Performance tracking
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'vectorized_operations': 0,
            'computation_time_saved': 0.0
        }

    def load_data(self):
        """Load all CSV files from the data directory and calculate price states"""
        print("Loading CSV files...")
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory {self.data_dir} not found!")
            return False
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return False
        
        # Load all CSV files
        for filename in csv_files:
            if not filename.lower().startswith("btc_usdt"):
                print(f"Skipping {filename} (non-BTC pair)")    
                continue
                
            filepath = os.path.join(self.data_dir, filename)
            pair_tf = filename.replace('.csv', '')
            
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 🔧 CALCULATE PRICE STATES IMMEDIATELY AFTER LOADING
                print(f"  Calculating price states for {pair_tf}...")
                df_with_states = self.identify_price_states(df)
                self.all_dataframes[pair_tf] = df_with_states
                
                # Debug info
                self.debug_price_states(self.all_dataframes[pair_tf], pair_tf)
                
                print(f"  ✓ Loaded {pair_tf}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {str(e)}")
        
        print(f"\n✓ Successfully loaded {len(self.all_dataframes)} datasets with price states")
        
        # Show summary of what was loaded
        for pair_tf, df in self.all_dataframes.items():
            future_return_exists = 'future_return' in df.columns
            price_state_exists = 'price_state' in df.columns
            print(f"  • {pair_tf}: {len(df)} rows | Future Return: {future_return_exists} | Price State: {price_state_exists}")
        
        return len(self.all_dataframes) > 0
    
    def get_mtf_dataframes(self, pair, htf_tf, ttf_tf, ltf_tf):
        """Enhanced version with price structure synchronization"""
        htf_key = f"{pair}_{htf_tf}"
        ttf_key = f"{pair}_{ttf_tf}" 
        ltf_key = f"{pair}_{ltf_tf}"
        
        if htf_key not in self.all_dataframes:
            return None, None, None
        if ttf_key not in self.all_dataframes:
            return None, None, None
        if ltf_key not in self.all_dataframes:
            return None, None, None
        
        htf_df = self.all_dataframes[htf_key].copy()
        ttf_df = self.all_dataframes[ttf_key].copy()
        ltf_df = self.all_dataframes[ltf_key].copy()
        
        # Ensure all have price states AND price action features calculated
        htf_df = self.identify_price_states(htf_df)
        ttf_df = self.identify_price_states(ttf_df) 
        ltf_df = self.identify_price_states(ltf_df)
        
        return self.align_mtf_with_price_structure(htf_df, ttf_df, ltf_df)

    def optimize_data_loading(self):
        """Optimize memory usage for all loaded dataframes"""
        print("Optimizing memory usage...")
        
        for pair_tf, df in self.all_dataframes.items():
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Downcast numeric types
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # Convert low cardinality strings to categories
            obj_cols = df.select_dtypes(include=['object']).columns
            for col in obj_cols:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            print(f"  {pair_tf}: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% reduction)")

    def get_or_compute_states(self, df, column_name, use_cache=True):
        """Get cached indicator states or compute if not cached."""
        cache_key = f"{id(df)}_{column_name}"
        
        if use_cache and cache_key in self.state_cache:
            return self.state_cache[cache_key]
        
        states = map_indicator_state(df, column_name)

        if use_cache and states is not None:
            if self.use_categorical_encoding:
                expected_cats = ['bullish', 'bearish', 'neutral', 'sideways', 0, 1, -1]
                states = pd.Categorical(states, categories=expected_cats)
                states = pd.Series(states, index=df.index)
            self.state_cache[cache_key] = states
        
        return states

    def identify_price_states(self, df, horizons=None):
        """FIXED: Vectorized price state identification with proper future return calculation"""
        if len(df) == 0:
            return pd.DataFrame()
            
        df = df.copy()
        horizons = horizons or [self.lookforward_periods]
        
        # Handle single-row case
        if len(df) <= max(horizons):
            # For small dataframes, return basic structure without future calculations
            for h in horizons:
                df[f'future_return_{h}'] = np.nan
                df[f'price_state_{h}'] = 'unknown'
            df['price_state'] = 'unknown'
            return df
        
        close_values = df['close'].values
        
        for h in horizons:
            # Calculate future returns correctly
            future_returns = np.full(len(close_values), np.nan)
            
            # Calculate future price changes
            for i in range(len(close_values) - h):
                future_price = close_values[i + h]
                current_price = close_values[i]
                future_returns[i] = (future_price / current_price - 1) * 100
            
            df[f'future_return_{h}'] = future_returns
            
            # Calculate price states
            price_states = []
            for ret in future_returns:
                if pd.isna(ret):
                    price_states.append('unknown')
                elif ret > self.price_threshold:
                    price_states.append('bullish')
                elif ret < -self.price_threshold:
                    price_states.append('bearish')
                else:
                    price_states.append('neutral')
            
            df[f'price_state_{h}'] = price_states
        
        # Overall price state (use shortest horizon)
        df['price_state'] = df[f'price_state_{horizons[0]}']
        df['future_return'] = df[f'future_return_{horizons[0]}']
        
        return df
    
    def _calculate_max_streak(self, bool_array):
        """Vectorized maximum consecutive True values"""
        if len(bool_array) == 0:
            return 0
        
        bool_array = np.asarray(bool_array, dtype=bool)
        diff = np.diff(np.concatenate(([False], bool_array, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
        
        return int((ends - starts).max())
