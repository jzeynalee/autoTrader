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


# =========================================================================
# 6.1 VECTORIZED PRICE ACTION CALCULATIONS
# =========================================================================
class PatternsMixin:
    def vectorized_swing_analysis(self, df, lookback=5):
        """
        6.1 Optimized swing point detection using NumPy
        Replaces iterative swing detection with vectorized operations
        """
        # Check for required columns
        if 'high' not in df.columns or 'low' not in df.columns:
            return {
                'swing_high': np.zeros(len(df), dtype=np.int8),
                'swing_low': np.zeros(len(df), dtype=np.int8)
            }
        

        cache_key = f"swing_{len(df)}_{df.index[-1]}_{lookback}"
        if cache_key in self.swing_cache:
            self.performance_stats['cache_hits'] += 1
            return self.swing_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        high_values = df['high'].values
        low_values = df['low'].values
        n = len(df)
        
        # Initialize arrays
        swing_highs = np.zeros(n, dtype=np.int8)
        swing_lows = np.zeros(n, dtype=np.int8)
        
        # Vectorized swing high detection
        for i in range(lookback, n - lookback):
            # Check if current high is higher than lookback periods before and after
            if (high_values[i] > np.max(high_values[i-lookback:i]) and 
                high_values[i] > np.max(high_values[i+1:i+lookback+1])):
                swing_highs[i] = 1
        
        # Vectorized swing low detection
        for i in range(lookback, n - lookback):
            # Check if current low is lower than lookback periods before and after
            if (low_values[i] < np.min(low_values[i-lookback:i]) and 
                low_values[i] < np.min(low_values[i+1:i+lookback+1])):
                swing_lows[i] = 1
        
        result = {
            'swing_high': swing_highs,
            'swing_low': swing_lows
        }
        
        # Cache the result
        self.swing_cache[cache_key] = result
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        self.performance_stats['vectorized_operations'] += 1
        
        return result

    def vectorized_trend_structure_analysis(self, df):
        """
        Vectorized trend structure analysis using NumPy
        """
        cache_key = f"trend_structure_{len(df)}_{df.index[-1]}"
        if cache_key in self.structure_cache:
            self.performance_stats['cache_hits'] += 1
            return self.structure_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        high_values = df['high'].values
        low_values = df['low'].values
        close_values = df['close'].values
        n = len(df)
        
        # Initialize result arrays
        higher_highs = np.zeros(n, dtype=np.int8)
        higher_lows = np.zeros(n, dtype=np.int8)
        lower_highs = np.zeros(n, dtype=np.int8)
        lower_lows = np.zeros(n, dtype=np.int8)
        
        # Vectorized HH/LL detection using rolling windows
        for i in range(10, n):
            # Look at last 5 bars for structure
            window_start = max(0, i-5)
            window_highs = high_values[window_start:i+1]
            window_lows = low_values[window_start:i+1]
            
            # Higher Highs/Lower Lows detection
            if len(window_highs) >= 3:
                # Check if we have consecutive higher highs
                high_increasing = (
                    all(window_highs[j] > window_highs[j-1] for j in range(1, len(window_highs)))
                    if len(window_highs) > 1
                    else False
                )
                                
                low_increasing = (
                    all(window_lows[j] > window_lows[j-1] for j in range(1, len(window_lows)))
                    if len(window_lows) > 1
                    else False
                )

                if high_increasing and low_increasing:
                    higher_highs[i] = 1
                    higher_lows[i] = 1
            
            # Lower Highs/Lower Lows detection
            if len(window_highs) >= 3:
                high_decreasing = (
                    all(window_highs[j] < window_highs[j-1] for j in range(1, len(window_highs)))
                    if len(window_highs) > 1 
                    else False
                    )
                low_decreasing = (
                    all(window_lows[j] < window_lows[j-1] for j in range(1, len(window_lows)))
                    if len(window_lows) > 1 
                    else False
                    )
                
                if high_decreasing and low_decreasing:
                    lower_highs[i] = 1
                    lower_lows[i] = 1
        
        result = {
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'lower_highs': lower_highs,
            'lower_lows': lower_lows,
            'trend_strength': (higher_highs + higher_lows - lower_highs - lower_lows) / 4.0
        }
        
        # Cache the result
        self.structure_cache[cache_key] = result
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        self.performance_stats['vectorized_operations'] += 1
        
        return result

    def vectorized_momentum_analysis(self, df):
        """
        Vectorized momentum analysis using NumPy
        """
        cache_key = f"momentum_{len(df)}_{df.index[-1]}"
        if cache_key in self.pattern_cache:
            self.performance_stats['cache_hits'] += 1
            return self.pattern_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        close_values = df['close'].values
        high_values = df['high'].values
        low_values = df['low'].values
        
        if 'rsi' not in df.columns:
            return {}
        
        rsi_values = df['rsi'].values
        n = len(df)
        
        # Initialize result arrays
        momentum_divergence_bullish = np.zeros(n, dtype=np.int8)
        momentum_divergence_bearish = np.zeros(n, dtype=np.int8)
        
        # Vectorized momentum divergence detection
        for i in range(10, n):
            # Look for bullish divergence (price lower low, RSI higher low)
            if i >= 10:
                price_window = low_values[i-10:i+1]
                rsi_window = rsi_values[i-10:i+1]
                
                # Find significant lows in price and RSI
                price_low_idx = np.argmin(price_window)
                rsi_low_idx = np.argmin(rsi_window)
                
                if (price_low_idx == len(price_window)-1 and  # Price made new low
                    rsi_low_idx != len(rsi_window)-1 and      # RSI didn't make new low
                    rsi_values[i] > np.min(rsi_window[:-1])): # RSI is higher
                    momentum_divergence_bullish[i] = 1
            
            # Look for bearish divergence (price higher high, RSI lower high)
            if i >= 10:
                price_window = high_values[i-10:i+1]
                rsi_window = rsi_values[i-10:i+1]
                
                # Find significant highs in price and RSI
                price_high_idx = np.argmax(price_window)
                rsi_high_idx = np.argmax(rsi_window)
                
                if (price_high_idx == len(price_window)-1 and  # Price made new high
                    rsi_high_idx != len(rsi_window)-1 and      # RSI didn't make new high
                    rsi_values[i] < np.max(rsi_window[:-1])):  # RSI is lower
                    momentum_divergence_bearish[i] = 1
        
        result = {
            'momentum_divergence_bullish': momentum_divergence_bullish,
            'momentum_divergence_bearish': momentum_divergence_bearish
        }
        
        # Cache the result
        self.pattern_cache[cache_key] = result
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        self.performance_stats['vectorized_operations'] += 1
        
        return result

    def vectorized_volume_analysis(self, df):
        """
        Vectorized volume analysis using NumPy
        """
        cache_key = f"volume_{len(df)}_{df.index[-1]}"
        if cache_key in self.pattern_cache:
            self.performance_stats['cache_hits'] += 1
            return self.pattern_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        if 'volume' not in df.columns:
            return {}
        
        volume_values = df['volume'].values
        close_values = df['close'].values
        n = len(df)
        
        # Initialize result arrays
        volume_breakout = np.zeros(n, dtype=np.int8)
        volume_divergence = np.zeros(n, dtype=np.int8)
        
        # Vectorized volume analysis
        volume_sma = pd.Series(volume_values).rolling(20).mean().values
        
        for i in range(20, n):
            # Volume breakout detection
            if volume_values[i] > volume_sma[i] * 1.5:
                volume_breakout[i] = 1
            
            # Volume divergence detection (decreasing volume in downtrend)
            if i >= 10:
                recent_volume_avg = np.mean(volume_values[i-5:i+1])
                previous_volume_avg = np.mean(volume_values[i-10:i-5])
                price_trend = close_values[i] < close_values[i-5]  # Price declining
                
                if price_trend and recent_volume_avg < previous_volume_avg * 0.8:
                    volume_divergence[i] = 1
        
        result = {
            'volume_breakout': volume_breakout,
            'volume_divergence': volume_divergence
        }
        
        # Cache the result
        self.pattern_cache[cache_key] = result
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        self.performance_stats['vectorized_operations'] += 1
        
        return result

    # =========================================================================
    # 6.2 CACHING STRATEGY FOR PRICE STRUCTURE
    # =========================================================================

    def get_cached_price_structure(self, df, lookback=20):
        """
        6.2 Cache expensive price structure calculations
        """
        cache_key = f"price_structure_{len(df)}_{df.index[-1]}"
        if cache_key in self.structure_cache:
            self.performance_stats['cache_hits'] += 1
            return self.structure_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        # Perform expensive price structure calculations
        price_structure = self._calculate_price_structure(df, lookback)
        
        # Cache the result
        self.structure_cache[cache_key] = price_structure
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        
        return price_structure

    def _calculate_price_structure(self, df, lookback):
        """
        Calculate comprehensive price structure (expensive operation)
        """
        structure = {
            'support_levels': [],
            'resistance_levels': [],
            'trend_lines': [],
            'key_levels': [],
            'structure_score': 0
        }
        
        # Get swing points
        swing_analysis = self.vectorized_swing_analysis(df, lookback)
        swing_high_indices = np.where(swing_analysis['swing_high'] == 1)[0]
        swing_low_indices = np.where(swing_analysis['swing_low'] == 1)[0]
        
        # Identify support and resistance levels
        if len(swing_high_indices) >= 2:
            recent_highs = swing_high_indices[-min(5, len(swing_high_indices)):]
            resistance_prices = [df.iloc[idx]['high'] for idx in recent_highs]
            structure['resistance_levels'] = self._cluster_price_levels(resistance_prices)
        
        if len(swing_low_indices) >= 2:
            recent_lows = swing_low_indices[-min(5, len(swing_low_indices)):]
            support_prices = [df.iloc[idx]['low'] for idx in recent_lows]
            structure['support_levels'] = self._cluster_price_levels(support_prices)
        
        # Calculate structure score
        structure_score = self._calculate_structure_score(df, swing_analysis)
        structure['structure_score'] = structure_score
        
        return structure

    def _cluster_price_levels(self, prices, threshold=0.005):
        """
        Cluster nearby price levels to identify key areas
        """
        if not prices:
            return []
        
        clusters = []
        sorted_prices = sorted(prices)
        
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if abs(price - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(price)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters

    def _calculate_structure_score(self, df, swing_analysis):
        """
        Calculate quantitative structure quality score
        """
        swing_high_count = np.sum(swing_analysis['swing_high'])
        swing_low_count = np.sum(swing_analysis['swing_low'])
        
        total_bars = len(df)
        swing_density = (swing_high_count + swing_low_count) / total_bars
        
        # Ideal swing density is between 0.1 and 0.3
        if 0.1 <= swing_density <= 0.3:
            density_score = 1.0
        elif swing_density < 0.1:
            density_score = swing_density / 0.1
        else:
            density_score = 0.3 / swing_density
        
        # Trend consistency score
        trend_analysis = self.vectorized_trend_structure_analysis(df)
        trend_strength = np.mean(np.abs(trend_analysis['trend_strength'][-20:]))
        trend_score = min(trend_strength * 2, 1.0)  # Normalize to 0-1
        
        # Combine scores
        structure_score = (density_score * 0.4 + trend_score * 0.6) * 100
        
        return structure_score


    def get_cached_mtf_alignment(self, htf_df, ttf_df, ltf_df):
        """Get cached MTF alignment or compute it"""
        cache_key = self._create_mtf_alignment_cache_key(htf_df, ttf_df, ltf_df)
        
        if cache_key in self.alignment_cache:
            return self.alignment_cache[cache_key]
        
        # Compute and cache (not recursive)
        alignment = self._perform_mtf_alignment_analysis(htf_df, ttf_df, ltf_df)
        self.alignment_cache[cache_key] = alignment
        
        return alignment


    def get_cached_regime_analysis(self, df):
        """
        Cache expensive regime analysis calculations
        """
        cache_key = f"regime_{len(df)}_{df.index[-1]}"
        if cache_key in self.structure_cache:
            self.performance_stats['cache_hits'] += 1
            return self.structure_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        # Perform regime analysis
        regime = self.regime_detector.detect_advanced_market_regimes(df)
        
        # Cache the result
        self.structure_cache[cache_key] = regime
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        
        return regime



    # =========================================================================
    # 6.3 OPTIMIZED PATTERN DETECTION
    # =========================================================================

    def optimized_detect_advanced_price_patterns(self, df):
        """
        Optimized version of advanced price pattern detection
        Uses vectorized operations and caching
        """
        # Existing Implementation
        swing_analysis = self.vectorized_swing_analysis(df)

        cache_key = f"patterns_{len(df)}_{df.index[-1]}"
        if cache_key in self.pattern_cache:
            self.performance_stats['cache_hits'] += 1
            return self.pattern_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        df = df.copy()
        
        # Use vectorized analysis methods
        swing_analysis = self.vectorized_swing_analysis(df)
        trend_analysis = self.vectorized_trend_structure_analysis(df)
        momentum_analysis = self.vectorized_momentum_analysis(df)
        volume_analysis = self.vectorized_volume_analysis(df)
        
        # Add results to dataframe efficiently
        for key, values in swing_analysis.items():
            df[key] = values
        
        for key, values in trend_analysis.items():
            df[key] = values
        
        for key, values in momentum_analysis.items():
            if key in df.columns:
                df[key] = np.maximum(df[key].values, values)  # Combine with existing
            else:
                df[key] = values
        
        for key, values in volume_analysis.items():
            if key in df.columns:
                df[key] = np.maximum(df[key].values, values)  # Combine with existing
            else:
                df[key] = values
        
        # Cache the result
        self.pattern_cache[cache_key] = df
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        self.performance_stats['vectorized_operations'] += 4  # 4 vectorized operations
        
        return df



    # =========================================================================
    # 6.4 MEMORY MANAGEMENT AND CACHE OPTIMIZATION
    # =========================================================================

    def optimize_memory_usage(self):
        """
        Optimize memory usage for all caches and dataframes
        """
        print("Optimizing memory usage...")
        
        # Downcast numeric types in all dataframes
        for pair_tf, df in self.all_dataframes.items():
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Downcast numeric columns
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            print(f"  {pair_tf}: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({reduction:.1f}% reduction)")
        
        # Optimize cache memory usage
        self._optimize_cache_memory()


    def _optimize_cache_memory(self):
        """Optimize cache memory usage more aggressively"""
        max_size = self.max_cache_size
        
        if len(self.structure_cache) > max_size:
            # Remove oldest entries
            keys_to_remove = list(self.structure_cache.keys())[:len(self.structure_cache) - max_size]
            for key in keys_to_remove:
                del self.structure_cache[key]
        
        # Apply same logic to other caches
        for cache_name in ['swing_cache', 'pattern_cache', 'alignment_cache', 'state_cache']:
            cache = getattr(self, cache_name)
            if len(cache) > max_size:
                keys_to_remove = list(cache.keys())[:len(cache) - max_size]
                for key in keys_to_remove:
                    del cache[key]

    def clear_old_cache_entries(self, older_than_hours=24):
        """
        Clear cache entries older than specified hours
        """
        current_time = time.time()
        cleared_count = 0
        
        for cache_name in ['structure_cache', 'swing_cache', 'pattern_cache', 'alignment_cache']:
            cache = getattr(self, cache_name)
            keys_to_remove = []
            
            for key in list(cache.keys()):
                # Simple heuristic: remove entries with old dataframe IDs
                if '_' in key:
                    try:
                        # Extract timestamp or use creation order
                        if 'timestamp' in str(key):
                            keys_to_remove.append(key)
                    except:
                        pass
            
            for key in keys_to_remove:
                del cache[key]
                cleared_count += 1
        
        print(f"Cleared {cleared_count} old cache entries")
    
    # patterns.py (inside PatternsMixin class)

    def analyze_mtf_structure_alignment(self, htf_df, ttf_df, ltf_df):
        """
        Analyzes the structural alignment of trends across three timeframes.
        This is a helper function for Mode L.
        """
        # Use caching
        cache_key = f"alignment_{len(htf_df)}_{htf_df.index[-1]}_{len(ltf_df)}_{ltf_df.index[-1]}"
        if cache_key in self.alignment_cache:
            self.performance_stats['cache_hits'] += 1
            return self.alignment_cache[cache_key]
            
        self.performance_stats['cache_misses'] += 1
        
        def get_trend_direction(df):
            # Use 'trend_structure' if available, else simple MA
            if 'trend_structure' in df.columns:
                # Use .iloc[-1] on the *Series* (df['col']), not the DataFrame (df)
                last_trend = df['trend_structure'].iloc[-1]
                if last_trend in ['strong_uptrend', 'uptrend']: return 1
                if last_trend in ['strong_downtrend', 'downtrend']: return -1
                return 0
            elif 'ema_50' in df.columns:
                # Fallback to simple MA cross
                if df['close'].iloc[-1] > df['ema_50'].iloc[-1]: return 1
                if df['close'].iloc[-1] < df['ema_50'].iloc[-1]: return -1
                return 0
            return 0

        # Get the trend direction for the *last available bar*
        htf_trend = get_trend_direction(htf_df)
        ttf_trend = get_trend_direction(ttf_df)
        ltf_trend = get_trend_direction(ltf_df)

        trends = [htf_trend, ttf_trend, ltf_trend]
        
        # Calculate alignment score
        alignment_score = 0
        if htf_trend == ttf_trend == ltf_trend and htf_trend != 0:
            alignment_score = 1.0  # Perfect alignment
            quality = 'excellent'
        elif (htf_trend == ttf_trend and htf_trend != 0) or (htf_trend == ltf_trend and htf_trend != 0):
            alignment_score = 0.6  # HTF aligns with one other
            quality = 'good'
        elif ttf_trend == ltf_trend and ttf_trend != 0:
            alignment_score = 0.3  # Lower TFs align, HTF conflicts
            quality = 'poor'
        else:
            alignment_score = 0.0 # No alignment
            quality = 'conflicting'
            
        result = {
            'htf_trend': htf_trend,
            'ttf_trend': ttf_trend,
            'ltf_trend': ltf_trend,
            'alignment_quality': quality,
            'overall_alignment_score': alignment_score
        }
        
        self.alignment_cache[cache_key] = result
        return result
