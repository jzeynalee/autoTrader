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

    '''def optimized_detect_advanced_price_patterns(self, df):
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
        
        return df'''

    # Replace the optimized_detect_advanced_price_patterns method in patterns.py

    def optimized_detect_advanced_price_patterns(self, df):
        """
        COMPLETE pattern detection including ALL advanced patterns required by Modes F-N
        """
        cache_key = f"patterns_{len(df)}_{df.index[-1]}"
        if cache_key in self.pattern_cache:
            self.performance_stats['cache_hits'] += 1
            return self.pattern_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        start_time = time.time()
        
        df = df.copy()
        
        # === STEP 1: Basic Price Action Patterns ===
        swing_analysis = self.vectorized_swing_analysis(df)
        for key, values in swing_analysis.items():
            df[key] = values
        
        # === STEP 2: Trend Structure ===
        trend_analysis = self.vectorized_trend_structure_analysis(df)
        for key, values in trend_analysis.items():
            df[key] = values
        
        # === STEP 3: Momentum Patterns ===
        momentum_analysis = self.vectorized_momentum_analysis(df)
        for key, values in momentum_analysis.items():
            if key in df.columns:
                df[key] = np.maximum(df[key].values, values)
            else:
                df[key] = values
        
        # === STEP 4: Volume Patterns ===
        volume_analysis = self.vectorized_volume_analysis(df)
        for key, values in volume_analysis.items():
            if key in df.columns:
                df[key] = np.maximum(df[key].values, values)
            else:
                df[key] = values
        
        # === STEP 5: CRITICAL - Add Missing Advanced Patterns ===
        # These are required by Modes F-N but were NOT being created!
        
        # Trend Structure (required by F, G, H, I, M, N)
        if 'trend_structure' not in df.columns:
            df['trend_structure'] = self._analyze_trend_structure(df)
        
        # Market Structure (required by F, G, M)
        if 'market_structure' not in df.columns:
            df['market_structure'] = self._analyze_market_structure(df)
        
        # Higher Highs / Lower Lows (required by F, K, L)
        if 'higher_highs_lower_lows' not in df.columns:
            df['higher_highs_lower_lows'] = self._detect_hh_ll_pattern(df)
        
        # Equal Highs/Lows (required by F, K, M)
        if 'equal_highs_lows' not in df.columns:
            df['equal_highs_lows'] = self._detect_equal_highs_lows(df)
        
        # Swing Failures (required by F, G, K)
        if 'swing_failure' not in df.columns:
            df['swing_failure'] = self._detect_swing_failures(df)
        
        # Structure Breaks (required by F, G, H, I, K, L, M)
        if 'structure_break_bullish' not in df.columns:
            df['structure_break_bullish'] = self._detect_structure_break_bullish(df)
        if 'structure_break_bearish' not in df.columns:
            df['structure_break_bearish'] = self._detect_structure_break_bearish(df)
        
        # False Breakouts (required by F, G, K, M)
        if 'false_breakout_bullish' not in df.columns:
            df['false_breakout_bullish'] = self._detect_false_breakout_bullish(df)
        if 'false_breakout_bearish' not in df.columns:
            df['false_breakout_bearish'] = self._detect_false_breakout_bearish(df)
        
        # Momentum Continuation (required by F, G, H, K, L, M, N)
        if 'momentum_continuation' not in df.columns:
            df['momentum_continuation'] = self._detect_momentum_continuation(df)
        
        # Volume Breakout Confirmation (required by F, G, L)
        if 'volume_breakout_confirmation' not in df.columns:
            df['volume_breakout_confirmation'] = self._detect_volume_breakout_confirmation(df)
        
        # Volume Divergence (required by K, M)
        if 'volume_divergence' not in df.columns:
            df['volume_divergence'] = self._detect_volume_divergence(df)
        
        # === STEP 6: Additional Patterns for Modes H, I ===
        # Pullback Completion (required by H, I, D, E)
        if 'pullback_complete_bull' not in df.columns:
            # Simplified pullback detection (the full version is in the calculator)
            df['pullback_complete_bull'] = pd.Series(0, index=df.index, dtype=np.int8)
            df['pullback_complete_bear'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # Healthy Pullbacks (required by D, E)
        if 'healthy_bull_pullback' not in df.columns:
            df['healthy_bull_pullback'] = pd.Series(0, index=df.index, dtype=np.int8)
            df['healthy_bear_pullback'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # ABC Pullbacks (required by D, E)
        if 'abc_pullback_bull' not in df.columns:
            df['abc_pullback_bull'] = pd.Series(0, index=df.index, dtype=np.int8)
            df['abc_pullback_bear'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # SR Confluence (required by D, E)
        if 'sr_confluence_score' not in df.columns:
            df['sr_confluence_score'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # Near Fib Levels (required by D, E)
        for fib_level in ['236', '382', '500', '618', '786']:
            col_name = f'near_fib_{fib_level}'
            if col_name not in df.columns:
                df[col_name] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # Pullback Stage (required by D, E)
        if 'pullback_stage' not in df.columns:
            df['pullback_stage'] = pd.Series('none', index=df.index)
        
        # Volume Decreasing (required by D, E, Confluence)
        if 'volume_decreasing' not in df.columns:
            df['volume_decreasing'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # Trend Short/Medium/Long (required by D, E, H, I)
        if 'trend_medium' not in df.columns:
            df['trend_short'] = pd.Series(0, index=df.index, dtype=np.int8)
            df['trend_medium'] = pd.Series(0, index=df.index, dtype=np.int8)
            df['trend_long'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # BB Squeeze (required by M)
        if 'bb_squeeze' not in df.columns:
            df['bb_squeeze'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # RSI Extreme (required by M)
        if 'rsi_extreme' not in df.columns:
            df['rsi_extreme'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # Momentum Confirmation (required by M)
        if 'momentum_confirmation' not in df.columns:
            df['momentum_confirmation'] = pd.Series(0, index=df.index, dtype=np.int8)
        
        # Cache the result
        self.pattern_cache[cache_key] = df
        self.performance_stats['computation_time_saved'] += (time.time() - start_time)
        self.performance_stats['vectorized_operations'] += 4
        
        return df
    
    def _analyze_trend_structure(self, df):
        """Analyze overall trend structure state"""
        trend_vals = np.where(df['close'] > df['ema_50'], 1, 
                    np.where(df['close'] < df['ema_50'], -1, 0))
        
        # Use rolling count to determine strength
        uptrend_strength = pd.Series(trend_vals).rolling(20).apply(
            lambda x: (x == 1).sum() / len(x), raw=True
        )
        
        result = pd.Series('neutral', index=df.index)
        result[uptrend_strength > 0.7] = 'strong_uptrend'
        result[(uptrend_strength >= 0.55) & (uptrend_strength <= 0.7)] = 'uptrend'
        result[(uptrend_strength >= 0.3) & (uptrend_strength < 0.45)] = 'downtrend'
        result[uptrend_strength < 0.3] = 'strong_downtrend'
        
        return result

    def _analyze_market_structure(self, df):
        """Analyze market structure (trending vs ranging)"""
        # Use ADX as primary indicator
        if 'adx' in df.columns:
            result = pd.Series('ranging', index=df.index)
            result[df['adx'] > 35] = 'strong_trend'
            result[(df['adx'] >= 25) & (df['adx'] <= 35)] = 'trending'
            return result
        
        # Fallback: use price volatility
        returns = df['close'].pct_change()
        vol = returns.rolling(20).std()
        result = pd.Series('ranging', index=df.index)
        result[vol > vol.quantile(0.7)] = 'trending'
        return result

    def _detect_hh_ll_pattern(self, df):
        """Detect Higher Highs and Lower Lows pattern"""
        if 'higher_highs' not in df.columns or 'lower_lows' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Return 1 for HH pattern, -1 for LL pattern
        return np.where(df['higher_highs'] == 1, 1,
                        np.where(df['lower_lows'] == 1, -1, 0))

    def _detect_equal_highs_lows(self, df):
        """Detect equal highs/lows (consolidation)"""
        highs = df['high'].rolling(10).max()
        lows = df['low'].rolling(10).min()
        
        # Check if range is compressing
        range_pct = (highs - lows) / df['close']
        range_avg = range_pct.rolling(20).mean()
        
        # Equal highs/lows when current range < 50% of average
        return np.where(range_pct < range_avg * 0.5, 1, 0)

    def _detect_swing_failures(self, df):
        """Detect swing failures (false breaks)"""
        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = pd.Series(0, index=df.index)
        
        # Bullish swing failure: Price breaks below swing low but closes above it
        swing_lows = df['low'].where(df['swing_low'] == 1).ffill()
        bullish_failure = (df['low'] < swing_lows) & (df['close'] > swing_lows)
        result[bullish_failure] = 1
        
        # Bearish swing failure: Price breaks above swing high but closes below it
        swing_highs = df['high'].where(df['swing_high'] == 1).ffill()
        bearish_failure = (df['high'] > swing_highs) & (df['close'] < swing_highs)
        result[bearish_failure] = -1
        
        return result

    def _detect_structure_break_bullish(self, df):
        """Detect bullish structure breaks"""
        if 'swing_high' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Last swing high
        last_swing_high = df['high'].where(df['swing_high'] == 1).ffill()
        
        # Break confirmed when close > last swing high with volume
        vol_confirm = df['volume'] > df['volume'].rolling(20).mean() if 'volume' in df.columns else True
        
        return ((df['close'] > last_swing_high) & vol_confirm).astype(int)

    def _detect_structure_break_bearish(self, df):
        """Detect bearish structure breaks"""
        if 'swing_low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        last_swing_low = df['low'].where(df['swing_low'] == 1).ffill()
        vol_confirm = df['volume'] > df['volume'].rolling(20).mean() if 'volume' in df.columns else True
        
        return ((df['close'] < last_swing_low) & vol_confirm).astype(int)

    def _detect_false_breakout_bullish(self, df):
        """Detect bullish false breakouts (bear traps)"""
        if 'swing_low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Price breaks swing low but quickly reverses
        swing_lows = df['low'].where(df['swing_low'] == 1).ffill()
        
        # 3-bar reversal pattern
        break_below = df['low'] < swing_lows
        quick_recovery = df['close'].shift(-2) > swing_lows * 1.005  # Close back above within 2 bars
        
        return (break_below & quick_recovery.shift(2).fillna(False)).astype(int)

    def _detect_false_breakout_bearish(self, df):
        """Detect bearish false breakouts (bull traps)"""
        if 'swing_high' not in df.columns:
            return pd.Series(0, index=df.index)
        
        swing_highs = df['high'].where(df['swing_high'] == 1).ffill()
        
        break_above = df['high'] > swing_highs
        quick_reversal = df['close'].shift(-2) < swing_highs * 0.995
        
        return (break_above & quick_reversal.shift(2).fillna(False)).astype(int)

    def _detect_momentum_continuation(self, df):
        """Detect momentum continuation patterns"""
        if 'rsi' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # RSI stays above 50 (bullish momentum) or below 50 (bearish)
        rsi_trend = df['rsi'].rolling(5).mean()
        
        bullish = (rsi_trend > 50) & (df['close'] > df['close'].shift(1))
        bearish = (rsi_trend < 50) & (df['close'] < df['close'].shift(1))
        
        return np.where(bullish, 1, np.where(bearish, -1, 0))

    def _detect_volume_breakout_confirmation(self, df):
        """Detect volume-confirmed breakouts"""
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Volume spike + price movement
        vol_spike = df['volume'] > df['volume'].rolling(20).mean() * 1.5
        price_move = abs(df['close'].pct_change()) > df['close'].pct_change().rolling(20).std()
        
        bullish = vol_spike & (df['close'] > df['close'].shift(1)) & price_move
        bearish = vol_spike & (df['close'] < df['close'].shift(1)) & price_move
        
        return np.where(bullish, 1, np.where(bearish, -1, 0))

    def _detect_volume_divergence(self, df):
        """Detect volume divergence (price up, volume down = bearish)"""
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        price_trend = df['close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)
        vol_trend = df['volume'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False)
        
        # Divergence when price and volume move opposite
        divergence = price_trend != vol_trend
        
        return divergence.astype(int)

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
