import pandas as pd
import numpy as np
import os
import time
import json
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .reporting import ReportsMixin
from .patterns import PatternsMixin
from .backtesting import BacktestingMixin
from .analysis_helpers import AnalysisHelpersMixin
from .discovery_modes import DiscoveryModesMixin
from .analysis_trend import TrendAnalysisSystem
from .analysis_pullback import PullbackAnalysisSystem
from .analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .analysis_confluence_scoring import ConfluenceScoringSystem
from autoTrader.db_connector import DatabaseConnector

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
    print("FATAL WARNING: discovery_mapping.py failed to load. Strategy core will lack crucial constant definitions.")

class StrategyDiscoverySystem(DiscoveryModesMixin, ReportsMixin, PatternsMixin, BacktestingMixin, AnalysisHelpersMixin):
    """
    Discovers trading strategies by analyzing correlations between indicator states,
    candlestick patterns, chart patterns and price movements.
    Reports ALL confirming signals when a trend is detected.
    """

    # --- Global strict timeframe order ---
    TIMEFRAME_PRIORITY = {"4h": 1, "1h": 2, "15m": 3, "5m": 4, "1m": 5}
    
    def __init__(self, data_dir='lbank_data', lookforward_periods=5, price_threshold=0.005, n_jobs=-1, db_connector=None):
        """
        Initialize the strategy discovery system with enhanced trend and pullback analysis.
        
        Args:
            data_dir: Directory containing CSV files from calculator.py
            lookforward_periods: Number of periods to look forward for price movement
            price_threshold: Minimum price change % to consider bullish/bearish (default 2%)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.data_dir = data_dir
        self.db = db_connector
        self.regime_detector = None
        self.trend_analyzer = None
        self.pullback_analyzer = None
        self.confluence_scorer = None
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

    def load_data_from_db(self):
        """Loads all calculated features from the database into all_dataframes."""
        if self.db is None:
            print("Error: DB Connector not provided.")
            return False
        
        # Assumes these methods are now fully implemented in db_connector.py:
        pair_tfs = self.db.get_all_calculated_pair_tfs() 
        
        if not pair_tfs:
            print("No calculated feature data found in the database.")
            return False
            
        loaded_count = 0
        for pair_tf in pair_tfs:
            # Load the fully featured DataFrame (OHLCV + 200+ features)
            df = self.db.load_full_features(pair_tf) # Assumes load_full_features method in DBConnector
            
            if df is not None and not df.empty:
                self.all_dataframes[pair_tf] = df
                # We skip memory optimization and regime detection here, as they are 
                # now handled globally in the orchestrator loop.
                loaded_count += 1
                
        print(f"✅ Loaded {loaded_count} feature sets from Database.")
        return loaded_count > 0
    
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
        if not isinstance(df, pd.DataFrame):
            print(f"ERROR: Expected DataFrame, got {type(df)}. Aborting price state identification.")
            return pd.DataFrame()
        
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
        
        if 'close' not in df.columns:
            print(f"CRITICAL ERROR: 'close' column missing in DataFrame. Columns found: {df.columns.tolist()}")
            
            # Attempt to find common alternatives if 'close' is missing 
            if 'CLOSE' in df.columns:
                df = df.rename(columns={'CLOSE': 'close'})
            elif 'Close' in df.columns:
                df = df.rename(columns={'Close': 'close'})
            else:
                # If still missing after renaming, we must stop and return the incomplete DataFrame
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
    
    def _analyze_single_column(self, args):
        """Worker function for parallel column analysis"""
        col, df_id, pair_tf, categories = args
        
        try:
            df = self.all_dataframes[pair_tf]
            col_states = self.get_or_compute_states(df, col, use_cache=True)
            
            if col_states is None:
                return None
            
            col_states = self.debounce_signal_states(col_states, k=2, method='consecutive')
            
            analysis_df = pd.DataFrame({
                'signal_state': col_states.values,
                'price_state': df['price_state'].values,
                'future_return': df['future_return'].values
            })
            
            mask = ~pd.isna(analysis_df['signal_state']) & ~pd.isna(analysis_df['price_state'])
            analysis_df = analysis_df[mask]
            
            if len(analysis_df) < 50:
                return None
            
            if col in categories['candlestick_patterns']:
                signal_type = 'candlestick'
            elif col in categories['chart_patterns']:
                signal_type = 'chart_pattern'
            else:
                signal_type = 'indicator'
            
            results = []
            
            # Bullish analysis
            bullish_mask = analysis_df['signal_state'] == 'bullish'
            if bullish_mask.sum() > 20:
                bullish_correct = ((analysis_df['signal_state'] == 'bullish') & 
                                 (analysis_df['price_state'] == 'bullish')).sum()
                bullish_accuracy = bullish_correct / bullish_mask.sum()
                
                if bullish_accuracy > 0.55:
                    results.append({
                        'pair_tf': pair_tf,
                        'signal_type': signal_type,
                        'signal_name': col,
                        'direction': 'bullish',
                        'trade_direction': 'long',
                        'discovered_accuracy': bullish_accuracy,
                        'sample_size': int(bullish_mask.sum()),
                        'type': 'single_signal'
                    })
            
            # Bearish analysis
            bearish_mask = analysis_df['signal_state'] == 'bearish'
            if bearish_mask.sum() > 20:
                bearish_correct = ((analysis_df['signal_state'] == 'bearish') & 
                                 (analysis_df['price_state'] == 'bearish')).sum()
                bearish_accuracy = bearish_correct / bearish_mask.sum()
                
                if bearish_accuracy > 0.55:
                    results.append({
                        'pair_tf': pair_tf,
                        'signal_type': signal_type,
                        'signal_name': col,
                        'direction': 'bearish',
                        'trade_direction': 'short',
                        'discovered_accuracy': bearish_accuracy,
                        'sample_size': int(bearish_mask.sum()),
                        'type': 'single_signal'
                    })
            
            return results
            
        except Exception as e:
            return None
        
    def categorize_columns(self, df):
        """Categorize columns into indicators, candlestick patterns, and chart patterns."""
        exclude_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'future_close', 'future_return', 'price_state', 'returns'}
        
        all_candlestick_patterns = BULLISH_PATTERNS | BEARISH_PATTERNS | NEUTRAL_PATTERNS
        all_chart_patterns = CHART_BULLISH | CHART_BEARISH | CHART_NEUTRAL
        
        indicators = []
        candlestick_patterns = []
        chart_patterns = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if col in all_candlestick_patterns:
                candlestick_patterns.append(col)
            elif col in all_chart_patterns:
                chart_patterns.append(col)
            elif col.startswith('pattern_'):
                col_lower = col.lower()
                if any(p in col_lower for p in ['double', 'triple', 'head_shoulder', 
                                                  'flag', 'pennant', 'wedge', 'cup_handle',
                                                  'rectangle', 'triangle']):
                    chart_patterns.append(col)
                else:
                    candlestick_patterns.append(col)
            else:
                indicators.append(col)
        
        return {
            'indicators': indicators,
            'candlestick_patterns': candlestick_patterns,
            'chart_patterns': chart_patterns
        }
    
    def discover_multi_signal_strategies(self):
        """Parallel strategy discovery"""
        print("\n" + "="*80)
        print("DISCOVERING MULTI-SIGNAL STRATEGIES")
        print("="*80)
        
        strategy_id = 0
        
        # Enforce strict timeframe order when iterating all_dataframes:
        # 4h -> 1h -> 15m -> 5m -> 1m
        timeframe_priority = {'4h': 1, '1h': 2, '15m': 3, '5m': 4, '1m': 5}

        def _pair_tf_priority(pair_tf):
            # Expect pair_tf like "BTC_USDT_15m" or "btc_usdt_15m"
            try:
                tf = pair_tf.split('_')[-1].lower()
                return timeframe_priority.get(tf, 999)
            except Exception:
                return 999

        sorted_pair_tfs = sorted(list(self.all_dataframes.keys()), key=_pair_tf_priority)

        for pair_tf in sorted_pair_tfs:
            df = self.all_dataframes[pair_tf]
            # CRITICAL FIX 1: Validate the retrieved item type
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"⚠️ Skipping {pair_tf}: Data structure invalid or empty. Expected DataFrame.")
                continue
            print(f"\nAnalyzing {pair_tf}.")

            # ensure price states present (cheap idempotent op)
            df = self.identify_price_states(df)
            self.all_dataframes[pair_tf] = df
            categories = self.categorize_columns(df)

            all_columns = (categories['indicators'] +
                          categories['candlestick_patterns'] +
                          categories['chart_patterns'])

            # Prepare arguments for parallel processing per-timeframe
            args_list = [(col, id(df), pair_tf, categories) for col in all_columns]

            # Process columns in parallel but keep timeframe ordering deterministic
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(self._analyze_single_column, args_list))

            # Collect results
            for result in results:
                if result:
                    for strategy_data in result:
                        strategy_id += 1
                        strategy_key = f"STRAT_{strategy_id:04d}"
                        strategy_data['id'] = strategy_id
                        self.strategy_pool[strategy_key] = strategy_data

        print(f"\n✓ Discovered {len(self.strategy_pool)} signal-price correlations")
        return self.strategy_pool
    
    # Delegation for Regime Detection (Fixes the last run's AttributeError)
    def enhanced_market_regime_detection(self, df):
        """Routes the initial regime calculation request to the specialized module."""
        if self.regime_detector is None:
            # Fallback if the module wasn't injected (or call simple internal function)
            return self.identify_price_states(df) 
        
        # Delegate to the specialized class
        regime_results = self.regime_detector.detect_advanced_market_regimes(df)

        # --- ROBUSTNESS FIX ---
        # Check if the detector (incorrectly) returned a dict instead of a DataFrame.
        # This can happen if it's designed to return the *current* state as a dict.
        if isinstance(regime_results, dict):
            # This assumes the dict is a summary, e.g., {'primary_regime': 'trending'}
            # We will add these as new columns to the original DataFrame.
            df_with_regimes = df.copy()
            for key, value in regime_results.items():
                # Add the regime info as a new column, applied to all rows
                # This makes the data available for historical analysis if needed
                df_with_regimes[key] = value 
            return df_with_regimes
        
        # If it's a DataFrame (the correct behavior), return it
        elif isinstance(regime_results, pd.DataFrame):
            return regime_results
        
        # Fallback if it returned something else weird (e.g., None)
        else:
            print(f"⚠️  Regime detector returned unexpected type: {type(regime_results)}. Falling back.")
            return self.identify_price_states(df)

    # Delegation for Trend Analysis (Used by Modes H, I)
    def classify_mtf_trend_strength(self, htf_df, ttf_df, ltf_df):
        """Routes the MTF trend request to the specialized module."""
        if self.trend_analyzer is None:
            raise AttributeError("Trend analyzer not initialized.")
        return self.trend_analyzer.classify_mtf_trend_strength(htf_df, ttf_df, ltf_df)

    # Delegation for Pullback Analysis (Used by Modes H, I)
    def analyze_pullback_quality(self, df, trend_direction='bullish'):
        """Routes the Pullback quality request to the specialized module."""
        if self.pullback_analyzer is None:
            raise AttributeError("Pullback analyzer not initialized.")
        return self.pullback_analyzer.analyze_pullback_quality(df, trend_direction)
    
    def align_mtf_with_price_structure(self, htf_df, ttf_df, ltf_df):
            """
            Enhanced MTF alignment that synchronizes swing points and pullback stages
            across timeframes to prevent structural mismatches
            """
            
            # --- FIX: Ensure 'timestamp' is a column for merge_asof ---
            # DataFrames are loaded with 'timestamp' as the index.
            # We must reset it to be a column for this function's logic.
            if 'timestamp' not in ltf_df.columns:
                ltf_df = ltf_df.reset_index()
            if 'timestamp' not in ttf_df.columns:
                ttf_df = ttf_df.reset_index()
            if 'timestamp' not in htf_df.columns:
                htf_df = htf_df.reset_index()

            # Use LTF as base for alignment - preserve ALL LTF columns
            aligned_data = ltf_df.copy()
            
            # Define key price action columns to synchronize
            price_action_columns = [
                'swing_high', 'swing_low', 'last_swing_high', 'last_swing_low',
                'pullback_stage', 'pullback_complete_bull', 'pullback_complete_bear',
                'higher_lows_pattern', 'lower_highs_pattern', 'trend_medium'
            ]
            
            # Merge HTF data with forward fill for swing points - PRESERVE OHLC COLUMNS
            htf_aligned = pd.merge_asof(
                aligned_data[['timestamp']],  # <-- This will now work
                htf_df,  # ✅ Merge ALL HTF columns including OHLC
                on='timestamp', 
                direction='backward'
            )
            
            # Merge TTF data - PRESERVE OHLC COLUMNS  
            ttf_aligned = pd.merge_asof(
                aligned_data[['timestamp']],  # <-- This will now work
                ttf_df,  # ✅ Merge ALL TTF columns including OHLC
                on='timestamp',
                direction='backward'
            )
            
            # Use LTF data as-is (most granular) - already has all columns
            ltf_aligned = ltf_df.copy()
            
            # Ensure swing point consistency - HTF swing points override lower timeframes
            swing_columns = ['swing_high', 'swing_low']
            for col in swing_columns:
                # HTF swing points are more significant
                htf_swing_mask = htf_aligned[col] == 1
                ttf_swing_mask = ttf_aligned[col] == 1
                
                # Remove LTF swing points that conflict with HTF structure
                conflicting_ltf_swings = (
                    ltf_aligned[col] == 1
                ) & (
                    (htf_swing_mask & (col == 'swing_high')) | 
                    (htf_swing_mask & (col == 'swing_low'))
                )
                
                ltf_aligned.loc[conflicting_ltf_swings, col] = 0
            
            print(f"  ✓ Price structure aligned: {len(htf_aligned)} bars")

            # --- FINAL FIX: Set index back to timestamp for consistency ---
            # The rest of the system expects 'timestamp' to be the index.
            # We must set it back before returning.
            try:
                if 'timestamp' in htf_aligned.columns:
                    htf_aligned = htf_aligned.set_index('timestamp')
                if 'timestamp' in ttf_aligned.columns:
                    ttf_aligned = ttf_aligned.set_index('timestamp')
                if 'timestamp' in ltf_aligned.columns:
                    ltf_aligned = ltf_aligned.set_index('timestamp')
            except Exception as e:
                print(f"Warning: Could not reset index to 'timestamp' after alignment: {e}")

            return htf_aligned, ttf_aligned, ltf_aligned
    
    def generate_pattern_effectiveness_report(self):
        """Generate report on the effectiveness of candlestick and chart patterns"""
        print("\n" + "="*80)
        print("PATTERN EFFECTIVENESS REPORT")
        print("="*80)
        
        pattern_stats = defaultdict(lambda: {
            'bullish_occurrences': 0,
            'bullish_successes': 0,
            'bearish_occurrences': 0,
            'bearish_successes': 0
        })
        
        for pair_tf, df in self.all_dataframes.items():
            df = self.identify_price_states(df)
            categories = self.categorize_columns(df)
            
            all_patterns = categories['candlestick_patterns'] + categories['chart_patterns']
            
            for pattern in all_patterns:
                pattern_states = map_indicator_state(df, pattern)
                
                if pattern_states is None:
                    continue
                
                pattern_values = pattern_states.values
                price_state_values = df['price_state'].values
                future_return_values = df['future_return'].values
                
                valid_mask = ~pd.isna(pattern_values) & ~pd.isna(price_state_values)
                
                bullish_mask = (pattern_values == 'bullish') & valid_mask
                if bullish_mask.sum() > 0:
                    pattern_stats[pattern]['bullish_occurrences'] += bullish_mask.sum()
                    pattern_stats[pattern]['bullish_successes'] += ((pattern_values == 'bullish') & 
                                                                    (future_return_values > 0) & 
                                                                    valid_mask).sum()
                
                bearish_mask = (pattern_values == 'bearish') & valid_mask
                if bearish_mask.sum() > 0:
                    pattern_stats[pattern]['bearish_occurrences'] += bearish_mask.sum()
                    pattern_stats[pattern]['bearish_successes'] += ((pattern_values == 'bearish') & 
                                                                    (future_return_values < 0) & 
                                                                    valid_mask).sum()
        
        pattern_effectiveness = []
        for pattern, stats in pattern_stats.items():
            total_occurrences = stats['bullish_occurrences'] + stats['bearish_occurrences']
            if total_occurrences >= 10:
                total_successes = stats['bullish_successes'] + stats['bearish_successes']
                overall_accuracy = total_successes / total_occurrences if total_occurrences > 0 else 0
                
                bullish_accuracy = (stats['bullish_successes'] / stats['bullish_occurrences'] 
                                   if stats['bullish_occurrences'] > 0 else 0)
                bearish_accuracy = (stats['bearish_successes'] / stats['bearish_occurrences'] 
                                   if stats['bearish_occurrences'] > 0 else 0)
                
                pattern_effectiveness.append({
                    'pattern': pattern,
                    'overall_accuracy': overall_accuracy,
                    'total_occurrences': total_occurrences,
                    'bullish_accuracy': bullish_accuracy,
                    'bullish_count': stats['bullish_occurrences'],
                    'bearish_accuracy': bearish_accuracy,
                    'bearish_count': stats['bearish_occurrences']
                })
        
        pattern_effectiveness.sort(key=lambda x: x['overall_accuracy'], reverse=True)
        
        print("\nTOP 20 MOST EFFECTIVE PATTERNS:")
        print("─"*80)
        
        for i, p in enumerate(pattern_effectiveness[:20], 1):
            print(f"\n#{i} {p['pattern']}")
            print(f"  Overall Accuracy: {p['overall_accuracy']:.2%} ({p['total_occurrences']} occurrences)")
            if p['bullish_count'] > 0:
                print(f"  Bullish: {p['bullish_accuracy']:.2%} ({p['bullish_count']} times)")
            if p['bearish_count'] > 0:
                print(f"  Bearish: {p['bearish_accuracy']:.2%} ({p['bearish_count']} times)")
        
        return pattern_effectiveness
