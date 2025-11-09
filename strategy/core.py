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
from ..db_connector import DatabaseConnector

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
    # CRITICAL: Define DUMMY placeholders
    def map_indicator_state(*args, **kwargs):
        raise NotImplementedError("map_indicator_state not available. Check discovery_mapping import.")

    BULLISH_PATTERNS, BEARISH_PATTERNS, NEUTRAL_PATTERNS = set(), set(), set()
    CHART_BULLISH, CHART_BEARISH, CHART_NEUTRAL = set(), set(), set()
    TIMEFRAME_GROUPS = {} 
    MAPPER_AVAILABLE = False
    print("FATAL WARNING: discovery_mapping.py failed to load. Strategy core will lack crucial constant definitions.")

class StrategyDiscoverySystem(DiscoveryModesMixin, ReportsMixin, PatternsMixin, BacktestingMixin, AnalysisHelpersMixin):
    """
    Discovers trading strategies by analyzing correlations between indicator states,
    candlestick patterns, chart patterns and price movements.
    """

    TIMEFRAME_PRIORITY = {"4h": 1, "1h": 2, "15m": 3, "5m": 4, "1m": 5}
    
    def __init__(self, data_dir='lbank_data', lookforward_periods=5, price_threshold=0.005, n_jobs=-1, db_connector=None):
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
        self.cache_hits = 0
        self.cache_misses = 0

        # Cache attributes
        self.structure_cache = {}
        self.swing_cache = {}
        self.pattern_cache = {}
        self.alignment_cache = {}
        self.aligned_df_cache = {}
        self.state_cache = {}
        self.max_cache_size = 1000
            
        self.performance_stats = {
            'cache_hits': 0, 'cache_misses': 0,
            'vectorized_operations': 0, 'computation_time_saved': 0.0
        }

    def load_data_from_db(self):
        if self.db is None:
            print("Error: DB Connector not provided.")
            return False
        
        pair_tfs = self.db.get_all_calculated_pair_tfs() 
        if not pair_tfs:
            print("No calculated feature data found in the database.")
            return False
            
        loaded_count = 0
        for pair_tf in pair_tfs:
            df = self.db.load_full_features(pair_tf) 
            if df is not None and not df.empty:
                # CRITICAL: Ensure keys are lowercase for consistency
                self.all_dataframes[pair_tf.lower()] = df
                loaded_count += 1
                
        print(f"✅ Loaded {loaded_count} feature sets from Database.")
        return loaded_count > 0
    
    def get_mtf_dataframes(self, pair, htf_tf, ttf_tf, ltf_tf):
        """
        Enhanced version with price structure synchronization
        FIXED: Now uses lowercase keys to match loaded data.
        FIXED: Caches aligned dataframes to prevent re-computation in backtesting.
        """
        pair_lower = pair.lower()
        cache_key = f"{pair_lower}_{htf_tf}_{ttf_tf}_{ltf_tf}"
        if cache_key in self.aligned_df_cache:
            self.cache_hits += 1
            return self.aligned_df_cache[cache_key]
        self.cache_misses += 1

        # Keys are loaded as lowercase (e.g., 'btc_usdt_4h')
        # The 'pair' argument must also be lowercase.
        htf_key = f"{pair_lower}_{htf_tf}"
        ttf_key = f"{pair_lower}_{ttf_tf}" 
        ltf_key = f"{pair_lower}_{ltf_tf}"
        # --- END OF FIX ---
        
        if htf_key not in self.all_dataframes:
            print(f"Warning: HTF data {htf_key} not found.")
            return None, None, None
        if ttf_key not in self.all_dataframes:
            print(f"Warning: TTF data {ttf_key} not found.")
            return None, None, None
        if ltf_key not in self.all_dataframes:
            print(f"Warning: LTF data {ltf_key} not found.")
            return None, None, None
        
        htf_df = self.all_dataframes[htf_key].copy()
        ttf_df = self.all_dataframes[ttf_key].copy()
        ltf_df = self.all_dataframes[ltf_key].copy()
        
        htf_df = self.identify_price_states(htf_df)
        ttf_df = self.identify_price_states(ttf_df) 
        ltf_df = self.identify_price_states(ltf_df)
                
        aligned_dfs = self.align_mtf_with_price_structure(htf_df, ttf_df, ltf_df)
        self.aligned_df_cache[cache_key] = aligned_dfs  # Store in cache
        return aligned_dfs

    def optimize_data_loading(self):
        print("Optimizing memory usage...")
        for pair_tf, df in self.all_dataframes.items():
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            obj_cols = df.select_dtypes(include=['object']).columns
            for col in obj_cols:
                if df[col].nunique() / len(df) < 0.5 and not pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            reduction = (original_memory - optimized_memory) / original_memory * 100
            print(f"  {pair_tf}: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% reduction)")

    def get_or_compute_states(self, df, column_name, pair_tf, use_cache=True):
        """
        Get cached indicator states or compute if not cached.
        FIXED: Uses a robust cache key including pair_tf to prevent IndexError.
        """
        # --- START OF CACHE KEY FIX ---
        # The bug "IndexError: Boolean index has wrong length" is caused by
        # cache collisions. This key is now 100% unique.
        try:
            cache_key = f"{pair_tf}_{len(df)}_{df.index[0]}_{df.index[-1]}_{column_name}"
        except IndexError:
            cache_key = f"empty_df_{column_name}"
        # --- END OF CACHE KEY FIX ---
        
        if use_cache and cache_key in self.state_cache:
            self.cache_hits += 1
            return self.state_cache[cache_key]
        
        self.cache_misses += 1
        states = map_indicator_state(df, column_name, pair_tf=pair_tf)

        if use_cache and states is not None:
            if self.use_categorical_encoding:
                expected_cats = ['bullish', 'bearish', 'neutral', 'sideways', 0, 1, -1]
                states_values = states.unique()
                new_cats = list(set(expected_cats) | set(states_values))
                states = pd.Categorical(states, categories=new_cats)
                states = pd.Series(states, index=df.index)
            self.state_cache[cache_key] = states
        
        return states

    def identify_price_states(self, df, horizons=None):
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        if 'close' not in df.columns: # Guard clause
            return df
            
        df = df.copy()
        horizons = horizons or [self.lookforward_periods]
        
        if len(df) <= max(horizons):
            for h in horizons:
                df[f'future_return_{h}'] = np.nan
                df[f'price_state_{h}'] = 'unknown'
            df['price_state'] = 'unknown'
            return df
        
        close_values = df['close'].values
        
        for h in horizons:
            future_returns = np.full(len(close_values), np.nan)
            for i in range(len(close_values) - h):
                future_price = close_values[i + h]
                current_price = close_values[i]
                if current_price != 0:
                    future_returns[i] = (future_price / current_price - 1) * 100
                else:
                    future_returns[i] = 0.0
            
            df[f'future_return_{h}'] = future_returns
            
            price_states = np.where(future_returns > self.price_threshold, 'bullish',
                           np.where(future_returns < -self.price_threshold, 'bearish', 'neutral'))
            price_states[pd.isna(future_returns)] = 'unknown'
            
            df[f'price_state_{h}'] = pd.Categorical(price_states)
        
        df['price_state'] = df[f'price_state_{horizons[0]}']
        df['future_return'] = df[f'future_return_{horizons[0]}']
        
        return df
    
    def _analyze_single_column(self, args):
        """
        Worker function for parallel column analysis
        FIXED: Applies debounce *only* to indicators, not to patterns.
        """
        col, df_id_tuple, pair_tf, categories = args
        
        try:
            df = self.all_dataframes[pair_tf]
            # Pass pair_tf for correct caching
            col_states = self.get_or_compute_states(df, col, pair_tf, use_cache=True)
            
            if col_states is None:
                return None
            
            # --- START OF DEBOUNCE FIX ---
            if col in categories['candlestick_patterns']:
                signal_type = 'candlestick'
            elif col in categories['chart_patterns']:
                signal_type = 'chart_pattern'
            else:
                # ONLY apply debounce to non-pattern indicators
                col_states = self.debounce_signal_states(col_states, k=2, method='consecutive')
                signal_type = 'indicator'
            # --- END OF DEBOUNCE FIX ---
            
            analysis_df = pd.DataFrame({
                'signal_state': col_states.values,
                'price_state': df['price_state'].values,
                'future_return': df['future_return'].values
            })
            
            mask = ~pd.isna(analysis_df['signal_state']) & ~pd.isna(analysis_df['price_state'])
            analysis_df = analysis_df[mask]
            
            if len(analysis_df) < 50:
                return None
            
            results = []
            
            # Bullish analysis
            bullish_mask = analysis_df['signal_state'] == 'bullish'
            if bullish_mask.sum() > 20:
                bullish_correct = ((analysis_df['signal_state'] == 'bullish') & 
                                 (analysis_df['price_state'] == 'bullish')).sum()
                bullish_accuracy = bullish_correct / bullish_mask.sum()
                
                # Use the 0.51 threshold
                if bullish_accuracy > 0.51:
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
                
                # Use the 0.51 threshold
                if bearish_accuracy > 0.51:
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
        exclude_cols = {
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'future_close', 'future_return', 'price_state', 'returns',
            'historical_regime', 'regime_volatility', 'regime_trend',
        }
        # Add any future_return/price_state columns to exclude
        exclude_cols.update([col for col in df.columns if 'future_return_' in col or 'price_state_' in col])
        
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
            elif any(s in col for s in ['_horizon', '_window', '_level']):
                continue
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
        total_found = 0 # <-- FIX for logging
        
        timeframe_priority = {'4h': 1, '1h': 2, '15m': 3, '5m': 4, '1m': 5}

        def _pair_tf_priority(pair_tf):
            try:
                tf = pair_tf.split('_')[-1].lower()
                return timeframe_priority.get(tf, 999)
            except Exception:
                return 999

        sorted_pair_tfs = sorted(list(self.all_dataframes.keys()), key=_pair_tf_priority)

        for pair_tf in sorted_pair_tfs:
            df = self.all_dataframes[pair_tf]
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"⚠️ Skipping {pair_tf}: Data structure invalid or empty.")
                continue
            print(f"\nAnalyzing {pair_tf}.")

            df = self.identify_price_states(df)
            self.all_dataframes[pair_tf] = df
            categories = self.categorize_columns(df)

            all_columns = (categories['indicators'] +
                          categories['candlestick_patterns'] +
                          categories['chart_patterns'])

            # Create a stable, hashable ID tuple for the df
            df_id_tuple = (len(df), df.index[0], df.index[-1])
            # Pass the pair_tf for correct caching
            args_list = [(col, df_id_tuple, pair_tf, categories) for col in all_columns]

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(self._analyze_single_column, args_list))

            # Collect results
            for result in results:
                if result:
                    for strategy_data in result:
                        strategy_id += 1
                        total_found += 1 # <-- FIX for logging
                        strategy_key = f"STRAT_{strategy_id:04d}"
                        strategy_data['id'] = strategy_id
                        self.strategy_pool[strategy_key] = strategy_data

        print(f"\n✓ Discovered {total_found} signal-price correlations") # <-- FIX for logging
        return self.strategy_pool
    
    def discover_combination_strategies(self):
        """
        COMPLETE REWRITE: Creates multi-indicator combination strategies
        by analyzing rows where price moved significantly and finding
        ALL confirming indicators/patterns active at that moment.
        """
        print("\n" + "="*80)
        print("DISCOVERING COMBINATION STRATEGIES")
        print("="*80)
        
        combination_strategies = []
        strategy_id = len(self.strategy_pool) + 1
        
        # Process each timeframe
        for pair_tf, df in self.all_dataframes.items():
            print(f"\nAnalyzing {pair_tf} for combinations...")
            
            if 'price_state' not in df.columns:
                df = self.identify_price_states(df)
            
            # Get all available signals
            categories = self.categorize_columns(df)
            all_signals = (categories['indicators'] + 
                        categories['candlestick_patterns'] + 
                        categories['chart_patterns'])
            
            # === FIND STRONG PRICE MOVEMENTS ===
            strong_bullish = df['price_state'] == 'bullish'
            strong_bearish = df['price_state'] == 'bearish'
            
            # === ANALYZE EACH STRONG MOVEMENT BAR ===
            for direction, mask in [('bullish', strong_bullish), ('bearish', strong_bearish)]:
                if mask.sum() < 10:  # Need minimum occurrences
                    continue
                
                # Get indices of strong movements
                movement_indices = df[mask].index
                
                # Dictionary to track signal co-occurrences
                signal_combinations = defaultdict(lambda: {
                    'occurrences': 0,
                    'successful': 0,
                    'signals': set()
                })
                
                # === ANALYZE EACH BAR ===
                for idx in movement_indices:
                    try:
                        # Get all active signals at this bar
                        active_signals = []
                        
                        for signal in all_signals:
                            if signal not in df.columns:
                                continue
                            
                            states = self.get_or_compute_states(df, signal, pair_tf)
                            if states is None:
                                continue
                            
                            # Check if signal was active at this bar
                            try:
                                if states.loc[idx] == direction:
                                    active_signals.append(signal)
                            except KeyError:
                                continue
                        
                        # Only consider bars with 3+ confirming signals
                        if len(active_signals) >= 3:
                            # Create a unique key for this combination
                            combo_key = tuple(sorted(active_signals[:5]))  # Limit to top 5
                            
                            # Track this combination
                            signal_combinations[combo_key]['occurrences'] += 1
                            signal_combinations[combo_key]['signals'].update(active_signals)
                            
                            # Check if price movement was successful
                            future_return = df.loc[idx, 'future_return']
                            if (direction == 'bullish' and future_return > 0) or \
                            (direction == 'bearish' and future_return < 0):
                                signal_combinations[combo_key]['successful'] += 1
                    
                    except Exception as e:
                        continue
                
                # === CREATE STRATEGIES FROM HIGH-PERFORMING COMBINATIONS ===
                for combo_key, stats in signal_combinations.items():
                    if stats['occurrences'] < 10:  # Minimum sample size
                        continue
                    
                    win_rate = stats['successful'] / stats['occurrences']
                    
                    if win_rate > 0.55:  # Combination threshold (lower than single signals)
                        combination_strategies.append({
                            'id': strategy_id,
                            'type': 'combination',
                            'pair_tf': pair_tf,
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'signals': list(stats['signals']),
                            'primary_signals': list(combo_key),  # The core combo
                            'signal_count': len(stats['signals']),
                            'discovered_accuracy': win_rate,
                            'sample_size': stats['occurrences'],
                            'successful_trades': stats['successful'],
                            'performance_score': win_rate * (1 + len(stats['signals']) * 0.05),  # Bonus for more confirmation
                            'strategy_class': 'multi_signal_combination'
                        })
                        
                        strategy_id += 1
        
        # === ADD TO STRATEGY POOL ===
        for strategy in combination_strategies:
            strategy_key = f"COMBO_{strategy['id']:04d}"
            self.strategy_pool[strategy_key] = strategy
        
        print(f"\n✅ Discovered {len(combination_strategies)} combination strategies")
        return combination_strategies
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION: Delegations & Mixin-Called Functions
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def enhanced_market_regime_detection(self, df):
        """Routes the regime calculation request to the specialized module."""
        if self.regime_detector is None:
            print("  ⚠️  Regime Detector not initialized. Skipping regime detection.")
            df['historical_regime'] = 'unknown' # Add fallback column
            return df
        
        return self.regime_detector.detect_advanced_market_regimes(df)

    def classify_mtf_trend_strength(self, htf_df, ttf_df, ltf_df):
        if self.trend_analyzer is None:
            raise AttributeError("Trend analyzer not initialized.")
        return self.trend_analyzer.classify_mtf_trend_strength(htf_df, ttf_df, ltf_df)

    def analyze_pullback_quality(self, df, trend_direction='bullish'):
        if self.pullback_analyzer is None:
            raise AttributeError("Pullback analyzer not initialized.")
        return self.pullback_analyzer.analyze_pullback_quality(df, trend_direction)
    
    def align_mtf_with_price_structure(self, htf_df, ttf_df, ltf_df):
            if 'timestamp' not in ltf_df.columns:
                ltf_df = ltf_df.reset_index()
            if 'timestamp' not in ttf_df.columns:
                ttf_df = ttf_df.reset_index()
            if 'timestamp' not in htf_df.columns:
                htf_df = htf_df.reset_index()

            aligned_data = ltf_df.copy()
            
            htf_aligned = pd.merge_asof(
                aligned_data[['timestamp']], htf_df, on='timestamp', direction='backward'
            )
            ttf_aligned = pd.merge_asof(
                aligned_data[['timestamp']], ttf_df, on='timestamp', direction='backward'
            )
            ltf_aligned = ltf_df.copy()
            
            print(f"  ✓ Price structure aligned: {len(htf_aligned)} bars")

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
        print("\n" + "="*80)
        print("PATTERN EFFECTIVENESS REPORT")
        print("="*80)
        
        pattern_stats = defaultdict(lambda: {
            'bullish_occurrences': 0, 'bullish_successes': 0,
            'bearish_occurrences': 0, 'bearish_successes': 0
        })
        
        for pair_tf, df in self.all_dataframes.items():
            if 'price_state' not in df.columns:
                df = self.identify_price_states(df)
            
            categories = self.categorize_columns(df)
            all_patterns = categories['candlestick_patterns'] + categories['chart_patterns']
            
            for pattern in all_patterns:
                pattern_states = map_indicator_state(df, pattern, pair_tf=pair_tf) # Pass pair_tf
                if pattern_states is None: continue
                
                pattern_values = pattern_states.values
                price_state_values = df['price_state'].values
                future_return_values = df['future_return'].values
                
                valid_mask = ~pd.isna(pattern_values) & ~pd.isna(price_state_values) & ~pd.isna(future_return_values)
                
                bullish_mask = (pattern_values == 'bullish') & valid_mask
                if bullish_mask.sum() > 0:
                    pattern_stats[pattern]['bullish_occurrences'] += bullish_mask.sum()
                    pattern_stats[pattern]['bullish_successes'] += ((future_return_values > 0) & bullish_mask).sum()
                
                bearish_mask = (pattern_values == 'bearish') & valid_mask
                if bearish_mask.sum() > 0:
                    pattern_stats[pattern]['bearish_occurrences'] += bearish_mask.sum()
                    pattern_stats[pattern]['bearish_successes'] += ((future_return_values < 0) & bearish_mask).sum()
        
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
                    'pattern': pattern, 'overall_accuracy': overall_accuracy,
                    'total_occurrences': total_occurrences,
                    'bullish_accuracy': bullish_accuracy, 'bullish_count': stats['bullish_occurrences'],
                    'bearish_accuracy': bearish_accuracy, 'bearish_count': stats['bearish_occurrences']
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
