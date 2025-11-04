import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict
import json
from datetime import datetime
from functools import lru_cache

# Try to import the planner, but create a dummy if it fails
try:
    from sl_tp_planner import SLTPPlanner
except ImportError:
    print("⚠️  Warning: 'sl_tp_planner' module not found. SL/TP calculations will be stubbed.")
    # Create a dummy class to prevent crashes
    class SLTPPlanner:
        def __init__(self, *args, **kwargs): pass
        def set_mtf_levels(self, *args, **kwargs): pass
        def set_by_regime(self, *args, **kwargs): pass
        def set_by_atr(self, *args, **kwargs): pass
        def validate_risk_reward(self, *args, **kwargs): pass
        def get_plan(self):
            return {"dummy_plan": {"valid": False, "reason": "Module not found"}}
        def get_trailing_config(self):
            return None


class AnalysisHelpersMixin:
    """
    Mixin containing the advanced analysis helper methods (Modes K, L, M, N),
    SL/TP logic, and other helpers that were left out of the main refactor.
    """

    def _analyze_htf_price_movements(self, htf_df):
        """Detect main HTF price movements and tag confirming indicators."""
        if htf_df is None or len(htf_df) == 0:
            return htf_df
        htf_df["htf_signal_core"] = np.sign(htf_df["close"].diff())
        return htf_df

    def _confirm_ttf_within_htf_intervals(self, htf_df, ttf_df):
        """Align TTF with HTF movement windows."""
        if ttf_df is None or len(ttf_df) == 0 or "htf_signal_core" not in htf_df.columns:
            return ttf_df
        htf_last_dir = htf_df["htf_signal_core"].iloc[-1]
        ttf_df["ttf_confirms_htf"] = np.where(
            np.sign(ttf_df["close"].diff()) == htf_last_dir, 1, 0
        )
        return ttf_df

    def _confirm_ltf_within_ttf_intervals(self, ttf_df, ltf_df):
        """Align LTF with TTF-confirmed regions."""
        if ltf_df is None or len(ltf_df) == 0 or "ttf_confirms_htf" not in ttf_df.columns:
            return ltf_df
        ttf_last_dir = ttf_df["ttf_confirms_htf"].iloc[-1]
        ltf_df["ltf_confirms_ttf"] = np.where(
            np.sign(ltf_df["close"].diff()) == ttf_last_dir, 1, 0
        )
        return ltf_df

    def _identify_regime_contexts(self, htf_regime, ttf_regime, ltf_regime):
        """Identify predominant regime contexts across timeframes"""
        contexts = []
        regimes = [htf_regime['primary_regime'], ttf_regime['primary_regime'], ltf_regime['primary_regime']]
        
        # Categorize regimes into broader contexts
        for regime in regimes:
            if 'trend' in regime:
                contexts.append('trending')
            elif 'ranging' in regime:
                contexts.append('ranging')
            elif 'transition' in regime:
                contexts.append('transition')
        
        # Return unique contexts
        return list(set(contexts)) if contexts else ['unknown']

    def _build_adaptive_signal_set(self, strategy_config, regime_context, htf_df, ttf_df, ltf_df):
        """Build adaptive signal set based on regime context"""
        adaptive_signals = {
            'htf': strategy_config['core_signals']['htf'].copy(),
            'ttf': strategy_config['core_signals']['ttf'].copy(),
            'ltf': strategy_config['core_signals']['ltf'].copy()
        }
        
        # Add regime-specific signals
        regime_adaptation = self.get_regime_adaptation(strategy_config, regime_context)
        additional_signals = regime_adaptation.get('additional_signals', [])
            
        # Add signals that exist in the dataframes
        for tf in ['htf', 'ttf', 'ltf']:
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            for signal in additional_signals:
                if signal in df.columns and signal not in adaptive_signals[tf]:
                    adaptive_signals[tf].append(signal)
        
        return adaptive_signals

    def _create_adaptive_signal_mask(self, adaptive_signals, htf_df, ttf_df, ltf_df, direction):
        """Create signal mask using adaptive signal set"""
        adaptive_mask = pd.Series(True, index=ltf_df.index)
        
        # Apply signals from each timeframe
        for tf, signals in adaptive_signals.items():
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            
            for signal in signals:
                if signal in df.columns:
                    states = self.get_or_compute_states(df, signal)
                    if states is not None:
                        if direction == 'bullish':
                            adaptive_mask &= (states == 'bullish')
                        else:
                            adaptive_mask &= (states == 'bearish')
        
        return adaptive_mask
    
    def _create_mtf_alignment_cache_key(self, htf_df, ttf_df, ltf_df):
        """Create cache key for MTF alignment analysis"""
        htf_key = f"{len(htf_df)}_{htf_df.index.iloc[-1]}_{htf_df['close'].iloc[-10]:.2f}"
        ttf_key = f"{len(ttf_df)}_{ttf_df.index.iloc[-1]}_{ttf_df['close'].iloc[-10]:.2f}"
        ltf_key = f"{len(ltf_df)}_{ltf_df.index.iloc[-1]}_{ltf_df['close'].iloc[-10]:.2f}"
        return f"mtf_align_{hash(htf_key + ttf_key + ltf_key) & 0xFFFFFFFF}"

    def analyze_mtf_structure_alignment(self, htf_df, ttf_df, ltf_df):
        """Analyze MTF structure alignment without recursion"""
        if not hasattr(self, 'alignment_cache'): self.alignment_cache = {}
        cache_key = self._create_mtf_alignment_cache_key(htf_df, ttf_df, ltf_df)
        
        if cache_key in self.alignment_cache:
            return self.alignment_cache[cache_key]
        
        # Perform actual analysis (not recursive call)
        alignment = self._perform_mtf_alignment_analysis(htf_df, ttf_df, ltf_df)
        
        # Cache the result
        self.alignment_cache[cache_key] = alignment
        if len(self.alignment_cache) > getattr(self, 'max_cache_size', 1000):
            self._optimize_cache_memory()
        
        return alignment

    def _perform_mtf_alignment_analysis(self, htf_df, ttf_df, ltf_df):
        """Actual MTF alignment analysis implementation"""
        # Perform all alignment analyses
        trend_alignment = self._check_trend_alignment(htf_df, ttf_df, ltf_df)
        swing_alignment = self._check_swing_alignment(htf_df, ttf_df, ltf_df)
        pullback_sync = self._check_pullback_synchronization(htf_df, ttf_df, ltf_df)
        structure_strength = self._analyze_structure_strength(htf_df, ttf_df, ltf_df)
        momentum_alignment = self._check_momentum_alignment(htf_df, ttf_df, ltf_df)
        volume_confirmation = self._check_volume_confirmation(htf_df, ttf_df, ltf_df)
        
        # Calculate overall scores
        overall_alignment_score = self._calculate_overall_alignment_score(
            trend_alignment, swing_alignment, pullback_sync, 
            structure_strength, momentum_alignment, volume_confirmation
        )
        
        alignment_quality = self._classify_alignment_quality(overall_alignment_score)
        
        return {
            'trend_alignment': trend_alignment,
            'swing_alignment': swing_alignment,
            'pullback_sync': pullback_sync,
            'structure_strength': structure_strength,
            'momentum_alignment': momentum_alignment,
            'volume_confirmation': volume_confirmation,
            'overall_alignment_score': overall_alignment_score,
            'alignment_quality': alignment_quality
        }

    def _check_pullback_synchronization(self, htf_df, ttf_df, ltf_df):
        """Check if pullbacks are synchronized across timeframes"""
        # Analyze pullback timing and depth alignment
        htf_pullbacks = self._identify_pullbacks(htf_df)
        ttf_pullbacks = self._identify_pullbacks(ttf_df)
        ltf_pullbacks = self._identify_pullbacks(ltf_df)
        
        sync_score = self._calculate_pullback_sync_score(
            htf_pullbacks, ttf_pullbacks, ltf_pullbacks
        )
        
        return {
            'score': sync_score,
            'quality': self._score_to_quality(sync_score),
            'htf_pullbacks': len(htf_pullbacks),
            'ttf_pullbacks': len(ttf_pullbacks),
            'ltf_pullbacks': len(ltf_pullbacks),
            'sync_ratio': self._calculate_sync_ratio(htf_pullbacks, ttf_pullbacks, ltf_pullbacks)
        }

    def _analyze_structure_strength(self, htf_df, ttf_df, ltf_df):
        """Analyze the strength of price structure across timeframes"""
        htf_strength = self._analyze_single_tf_structure_strength(htf_df)
        ttf_strength = self._analyze_single_tf_structure_strength(ttf_df)
        ltf_strength = self._analyze_single_tf_structure_strength(ltf_df)
        
        avg_strength = (htf_strength['score'] + ttf_strength['score'] + ltf_strength['score']) / 3
        consistency = self._calculate_structure_consistency(htf_strength, ttf_strength, ltf_strength)
        
        return {
            'score': avg_strength,
            'quality': self._score_to_quality(avg_strength),
            'consistency_score': consistency,
            'htf_strength': htf_strength,
            'ttf_strength': ttf_strength,
            'ltf_strength': ltf_strength
        }
    
    def _calculate_overall_alignment_score(self, trend_alignment, swing_alignment, pullback_sync, 
                                        structure_strength, momentum_alignment, volume_confirmation):
        """Calculate weighted overall alignment score"""
        weights = {
            'trend': 0.25,      # Most important - trend alignment
            'swing': 0.20,      # Swing structure alignment
            'pullback': 0.15,   # Pullback synchronization
            'structure': 0.15,  # Structure strength
            'momentum': 0.15,   # Momentum alignment
            'volume': 0.10      # Volume confirmation
        }
        
        weighted_score = (
            trend_alignment['score'] * weights['trend'] +
            swing_alignment['score'] * weights['swing'] +
            pullback_sync['score'] * weights['pullback'] +
            structure_strength['score'] * weights['structure'] +
            momentum_alignment['score'] * weights['momentum'] +
            volume_confirmation['score'] * weights['volume']
        )
        
        return min(1.0, max(0.0, weighted_score))

    def _classify_alignment_quality(self, score):
        """Classify alignment quality based on score"""
        if score >= 0.8:
            return 'perfect'
        elif score >= 0.7:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.5:
            return 'moderate'
        elif score >= 0.4:
            return 'partial'
        elif score >= 0.3:
            return 'weak'
        else:
            return 'conflicting'

    # Helper methods
    def _identify_pullbacks(self, df):
        """Identify pullback periods in dataframe"""
        pullbacks = []
        if len(df) < 10:
            return pullbacks
        
        close_prices = df['close'].values
        for i in range(10, len(close_prices) - 5):
            if (close_prices[i] > close_prices[i-5:i].max() and 
                close_prices[i+5] < close_prices[i] * 0.98):
                pullbacks.append(i)
        
        return pullbacks

    def _calculate_pullback_sync_score(self, htf_pullbacks, ttf_pullbacks, ltf_pullbacks):
        """Calculate pullback synchronization score"""
        if not htf_pullbacks or not ttf_pullbacks or not ltf_pullbacks:
            return 0.0
        
        min_pullbacks = min(len(htf_pullbacks), len(ttf_pullbacks), len(ltf_pullbacks))
        max_pullbacks = max(len(htf_pullbacks), len(ttf_pullbacks), len(ltf_pullbacks))
        
        if max_pullbacks == 0:
            return 0.0
        
        return min_pullbacks / max_pullbacks

    def _analyze_single_tf_structure_strength(self, df):
        """Analyze structure strength for a single timeframe"""
        if len(df) < 20 or 'swing_high' not in df.columns or 'swing_low' not in df.columns:
            return {'score': 0.0, 'quality': 'weak', 'swing_count': 0}
        
        swing_highs = len(df[df['swing_high'] == 1])
        swing_lows = len(df[df['swing_low'] == 1])
        total_swings = swing_highs + swing_lows
        expected_swings = max(1, len(df) // 20)
        
        swing_score = min(1.0, total_swings / expected_swings)
        
        return {
            'score': swing_score,
            'quality': self._score_to_quality(swing_score),
            'swing_count': total_swings,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }

    def _calculate_structure_consistency(self, htf_strength, ttf_strength, ltf_strength):
        """Calculate consistency of structure strength across timeframes"""
        scores = [htf_strength['score'], ttf_strength['score'], ltf_strength['score']]
        avg_score = sum(scores) / len(scores)
        
        if avg_score == 0:
            return 0.0
        
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        cv = std_dev / avg_score if avg_score > 0 else 1.0
        
        return max(0.0, 1.0 - cv)

    def _check_momentum_alignment(self, htf_df, ttf_df, ltf_df):
        """Check momentum alignment across timeframes"""
        momentum_indicators = ['rsi', 'macd', 'stoch_k', 'momentum_continuation']
        
        alignment_score = 0
        total_checks = 0
        
        for indicator in momentum_indicators:
            htf_states = self.get_or_compute_states(htf_df, indicator) if indicator in htf_df.columns else None
            ttf_states = self.get_or_compute_states(ttf_df, indicator) if indicator in ttf_df.columns else None
            ltf_states = self.get_or_compute_states(ltf_df, indicator) if indicator in ltf_df.columns else None
            
            if all(states is not None for states in [htf_states, ttf_states, ltf_states]):
                recent_htf = htf_states.iloc[-1]
                recent_ttf = ttf_states.iloc[-1]
                recent_ltf = ltf_states.iloc[-1]
                
                if recent_htf == recent_ttf == recent_ltf:
                    alignment_score += 1
                total_checks += 1
        
        score = alignment_score / total_checks if total_checks > 0 else 0
        
        return {
            'score': score,
            'quality': 'excellent' if score >= 0.8 else 'good' if score >= 0.6 else 'fair' if score >= 0.4 else 'poor',
            'aligned_indicators': alignment_score,
            'total_indicators': total_checks
        }

    def _check_volume_confirmation(self, htf_df, ttf_df, ltf_df):
        """Check volume confirmation across timeframes"""
        volume_metrics = ['volume_breakout_confirmation', 'volume_divergence', 'volume_trend']
        
        confirmation_score = 0
        total_checks = 0
        
        for metric in volume_metrics:
            htf_value = htf_df[metric].iloc[-1] if metric in htf_df.columns else 0
            ttf_value = ttf_df[metric].iloc[-1] if metric in ttf_df.columns else 0
            ltf_value = ltf_df[metric].iloc[-1] if metric in ltf_df.columns else 0
            
            # Check if volume confirms price action (all positive or all negative)
            if (htf_value > 0 and ttf_value > 0 and ltf_value > 0) or \
            (htf_value < 0 and ttf_value < 0 and ltf_value < 0):
                confirmation_score += 1
            total_checks += 1
        
        score = confirmation_score / total_checks if total_checks > 0 else 0
        
        return {
            'score': score,
            'quality': 'excellent' if score >= 0.8 else 'good' if score >= 0.6 else 'fair' if score >= 0.4 else 'poor',
            'volume_confirmations': confirmation_score,
            'total_volume_checks': total_checks
        }

    def _check_trend_alignment(self, htf_df, ttf_df, ltf_df):
        """Check trend direction alignment across timeframes"""
        # Use trend structure analysis
        htf_trend = self._analyze_trend_structure(htf_df).iloc[-1]
        ttf_trend = self._analyze_trend_structure(ttf_df).iloc[-1]
        ltf_trend = self._analyze_trend_structure(ltf_df).iloc[-1]
        
        # Convert to numerical scores
        trend_scores = []
        for trend in [htf_trend, ttf_trend, ltf_trend]:
            if 'strong_uptrend' in trend:
                trend_scores.append(2)
            elif 'uptrend' in trend:
                trend_scores.append(1)
            elif 'downtrend' in trend:
                trend_scores.append(-1)
            elif 'strong_downtrend' in trend:
                trend_scores.append(-2)
            else:
                trend_scores.append(0)
        
        # Calculate alignment (standard deviation)
        alignment_std = np.std(trend_scores)
        
        if alignment_std <= 0.5:
            score = 1.0
            quality = 'perfect'
        elif alignment_std <= 1.0:
            score = 0.75
            quality = 'good'
        elif alignment_std <= 1.5:
            score = 0.5
            quality = 'partial'
        else:
            score = 0.25
            quality = 'conflicting'
        
        return {
            'score': score,
            'quality': quality,
            'htf_trend': htf_trend,
            'ttf_trend': ttf_trend,
            'ltf_trend': ltf_trend,
            'std_deviation': alignment_std
        }

    def _check_swing_alignment(self, htf_df, ttf_df, ltf_df):
        """Check swing point alignment across timeframes"""
        alignment_data = []
        
        # Check recent swing highs
        htf_swing_highs = htf_df[htf_df['swing_high'] == 1].tail(3)
        ttf_swing_highs = ttf_df[ttf_df['swing_high'] == 1].tail(5)
        ltf_swing_highs = ltf_df[ltf_df['swing_high'] == 1].tail(8)
        
        # Check recent swing lows
        htf_swing_lows = htf_df[htf_df['swing_low'] == 1].tail(3)
        ttf_swing_lows = ttf_df[ttf_df['swing_low'] == 1].tail(5)
        ltf_swing_lows = ltf_df[ltf_df['swing_low'] == 1].tail(8)
        
        # Calculate alignment score based on swing point proximity
        swing_alignment_score = 0
        total_checks = 0
        
        # Check if HTF swing points are respected in lower timeframes
        for _, htf_swing in htf_swing_highs.iterrows():
            swing_time = htf_swing.name # Use index (timestamp)
            swing_price = htf_swing['high']
            
            # Find nearest TTF swing high
            ttf_nearest = self._find_nearest_swing(ttf_swing_highs, swing_time, swing_price)
            if ttf_nearest and abs(ttf_nearest['price_diff_pct']) < 0.01:  # Within 1%
                swing_alignment_score += 1
            
            # Find nearest LTF swing high
            ltf_nearest = self._find_nearest_swing(ltf_swing_highs, swing_time, swing_price)
            if ltf_nearest and abs(ltf_nearest['price_diff_pct']) < 0.005:  # Within 0.5%
                swing_alignment_score += 1
            
            total_checks += 2
        
        for _, htf_swing in htf_swing_lows.iterrows():
            swing_time = htf_swing.name # Use index (timestamp)
            swing_price = htf_swing['low']
            
            # Find nearest TTF swing low
            ttf_nearest = self._find_nearest_swing(ttf_swing_lows, swing_time, swing_price)
            if ttf_nearest and abs(ttf_nearest['price_diff_pct']) < 0.01:
                swing_alignment_score += 1
            
            # Find nearest LTF swing low
            ltf_nearest = self._find_nearest_swing(ltf_swing_lows, swing_time, swing_price)
            if ltf_nearest and abs(ltf_nearest['price_diff_pct']) < 0.005:
                swing_alignment_score += 1
            
            total_checks += 2
        
        score = swing_alignment_score / total_checks if total_checks > 0 else 0
        
        return {
            'score': score,
            'quality': 'excellent' if score >= 0.8 else 'good' if score >= 0.6 else 'fair' if score >= 0.4 else 'poor',
            'htf_swings': len(htf_swing_highs) + len(htf_swing_lows),
            'ttf_swings': len(ttf_swing_highs) + len(ttf_swing_lows),
            'ltf_swings': len(ltf_swing_highs) + len(ltf_swing_lows)
        }

    def _find_nearest_swing(self, swing_df, target_time, target_price):
        """Find nearest swing point to target time and price"""
        if swing_df.empty:
            return None
        
        # Find temporal proximity
        time_diffs = abs(swing_df.index - target_time)
        
        # --- FIX for TimedeltaIndex error ---
        # .argmin() returns the integer position of the minimum value
        nearest_pos = time_diffs.argmin() 
        nearest_swing = swing_df.iloc[nearest_pos]
        time_diff_val = time_diffs[nearest_pos]
        # --- End Fix ---

        # Calculate price difference percentage
        swing_price = nearest_swing['high'] if 'high' in swing_df.columns else nearest_swing['low']
        price_diff_pct = (swing_price - target_price) / target_price if target_price != 0 else 0
        
        return {
            'timestamp': nearest_swing.name,
            'price_diff_pct': price_diff_pct,
            'time_diff_hours': time_diff_val.total_seconds() / 3600
        }

    def _analyze_trend_structure(self, df):
        """Analyze the quality and strength of trend structure"""
        structure = pd.Series('neutral', index=df.index)
        
        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
             return structure # Not calculated yet
             
        # Get swing points
        swing_high_indices = df.index[df['swing_high'] == 1]
        swing_low_indices = df.index[df['swing_low'] == 1]
        
        if len(swing_high_indices) < 3 or len(swing_low_indices) < 3:
            return structure
        
        # Analyze higher highs/lower lows for uptrend
        recent_highs = swing_high_indices[-3:]
        recent_lows = swing_low_indices[-3:]
        
        # Check for consistent higher highs
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            highs_increasing = all(df.loc[recent_highs[i], 'high'] > df.loc[recent_highs[i-1], 'high'] 
                                for i in range(1, len(recent_highs)))
            lows_increasing = all(df.loc[recent_lows[i], 'low'] > df.loc[recent_lows[i-1], 'low'] 
                                for i in range(1, len(recent_lows)))
            
            if highs_increasing and lows_increasing:
                structure.iloc[-1] = 'strong_uptrend'
            elif highs_increasing:
                structure.iloc[-1] = 'weak_uptrend'
        
        # Check for consistent lower highs/lows for downtrend
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            highs_decreasing = all(df.loc[recent_highs[i], 'high'] < df.loc[recent_highs[i-1], 'high'] 
                                for i in range(1, len(recent_highs)))
            lows_decreasing = all(df.loc[recent_lows[i], 'low'] < df.loc[recent_lows[i-1], 'low'] 
                                for i in range(1, len(recent_lows)))
            
            if highs_decreasing and lows_decreasing:
                structure.iloc[-1] = 'strong_downtrend'
            elif highs_decreasing:
                structure.iloc[-1] = 'weak_downtrend'
        
        return structure

    def _analyze_market_structure(self, df):
        """Analyze overall market structure (trending, ranging, transitioning)"""
        structure = pd.Series('ranging', index=df.index)
        
        # Use ATR to measure volatility
        if 'atr' in df.columns:
            atr_values = df['atr'].values
            if len(atr_values) < 20: return structure
            atr_median = np.median(atr_values[-20:])
            current_atr = atr_values[-1]
            
            if atr_median > 0:
                # High volatility suggests trending
                if current_atr > atr_median * 1.5:
                    structure.iloc[-1] = 'trending_high_vol'
                elif current_atr < atr_median * 0.7:
                    structure.iloc[-1] = 'ranging_low_vol'
        
        # Use ADX for trend strength
        if 'adx' in df.columns:
            adx_values = df['adx'].values
            if len(adx_values) == 0: return structure
            current_adx = adx_values[-1]
            
            if current_adx > 25:
                if structure.iloc[-1] == 'trending_high_vol':
                    structure.iloc[-1] = 'strong_trend'
                else:
                    structure.iloc[-1] = 'trending'
            elif current_adx < 15:
                structure.iloc[-1] = 'ranging'
        
        return structure

    def _detect_hh_ll_pattern(self, df):
        """Detect Higher Highs + Higher Lows or Lower Highs + Lower Lows"""
        pattern = pd.Series(0, index=df.index)
        
        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
             return pattern
             
        swing_high_indices = df.index[df['swing_high'] == 1]
        swing_low_indices = df.index[df['swing_low'] == 1]
        
        if len(swing_high_indices) < 3 or len(swing_low_indices) < 3:
            return pattern
        
        # Check last 3 swings
        recent_highs = swing_high_indices[-3:]
        recent_lows = swing_low_indices[-3:]
        
        # Higher Highs + Higher Lows (Uptrend)
        hh = all(df.loc[recent_highs[i], 'high'] > df.loc[recent_highs[i-1], 'high'] 
                for i in range(1, len(recent_highs)))
        hl = all(df.loc[recent_lows[i], 'low'] > df.loc[recent_lows[i-1], 'low'] 
                for i in range(1, len(recent_lows)))
        
        # Lower Highs + Lower Lows (Downtrend)
        lh = all(df.loc[recent_highs[i], 'high'] < df.loc[recent_highs[i-1], 'high'] 
                for i in range(1, len(recent_highs)))
        ll = all(df.loc[recent_lows[i], 'low'] < df.loc[recent_lows[i-1], 'low'] 
                for i in range(1, len(recent_lows)))
        
        if hh and hl:
            pattern.iloc[-1] = 1  # HH + HL
        elif lh and ll:
            pattern.iloc[-1] = -1  # LH + LL
        
        return pattern

    def _detect_equal_highs_lows(self, df):
        """Detect equal highs/lows for potential breakout patterns"""
        pattern = pd.Series(0, index=df.index)
        
        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
             return pattern
             
        swing_high_indices = df.index[df['swing_high'] == 1]
        swing_low_indices = df.index[df['swing_low'] == 1]
        
        if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
            return pattern
        
        # Check for equal highs (resistance)
        last_two_highs = swing_high_indices[-2:]
        high1 = df.loc[last_two_highs[0], 'high']
        high2 = df.loc[last_two_highs[1], 'high']
        
        if high1 > 0 and abs(high1 - high2) / high1 < 0.002:  # Within 0.2%
            pattern.iloc[-1] = 1  # Equal highs
        
        # Check for equal lows (support)  
        last_two_lows = swing_low_indices[-2:]
        low1 = df.loc[last_two_lows[0], 'low']
        low2 = df.loc[last_two_lows[1], 'low']
        
        if low1 > 0 and abs(low1 - low2) / low1 < 0.002:  # Within 0.2%
            pattern.iloc[-1] = -1  # Equal lows
        
        return pattern

    def _detect_swing_failures(self, df):
        """Detect swing failure patterns (potential reversal signals)"""
        failures = pd.Series(0, index=df.index)
        
        if len(df) > 10 and 'swing_high' in df.columns and 'swing_low' in df.columns:
            # Bullish swing failure (bearish signal)
            # Price makes new high but closes below previous swing high
            current_high = df['high'].iloc[-1]
            prev_swing_high = df[df['swing_high'] == 1]['high'].tail(2).iloc[0] if len(df[df['swing_high'] == 1]) >= 2 else 0
            
            if current_high > prev_swing_high and df['close'].iloc[-1] < prev_swing_high:
                failures.iloc[-1] = -1
            
            # Bearish swing failure (bullish signal)
            # Price makes new low but closes above previous swing low
            current_low = df['low'].iloc[-1]
            prev_swing_low = df[df['swing_low'] == 1]['low'].tail(2).iloc[0] if len(df[df['swing_low'] == 1]) >= 2 else float('inf')
            
            if current_low < prev_swing_low and df['close'].iloc[-1] > prev_swing_low:
                failures.iloc[-1] = 1
        
        return failures

    def _detect_structure_break_bullish(self, df):
        """Detect bullish structure breaks (break of resistance)"""
        breaks = pd.Series(0, index=df.index)
        
        if len(df) < 20 or 'swing_high' not in df.columns:
            return breaks
        
        # Find recent resistance (swing highs)
        swing_highs = df[df['swing_high'] == 1]
        if len(swing_highs) < 2:
            return breaks
        
        recent_resistance = swing_highs['high'].tail(3).max()
        current_close = df['close'].iloc[-1]
        
        # Break above resistance with confirmation
        if current_close > recent_resistance and df['close'].iloc[-2] <= recent_resistance:
            # Volume confirmation
            if 'volume' in df.columns:
                avg_volume = df['volume'].tail(20).mean()
                if avg_volume > 0 and df['volume'].iloc[-1] > avg_volume * 1.2:
                    breaks.iloc[-1] = 1
        
        return breaks

    def _detect_structure_break_bearish(self, df):
        """Detect bearish structure breaks (break of support)"""
        breaks = pd.Series(0, index=df.index)
        
        if len(df) < 20 or 'swing_low' not in df.columns:
            return breaks
        
        # Find recent support (swing lows)
        swing_lows = df[df['swing_low'] == 1]
        if len(swing_lows) < 2:
            return breaks
        
        recent_support = swing_lows['low'].tail(3).min()
        current_close = df['close'].iloc[-1]
        
        # Break below support with confirmation
        if current_close < recent_support and df['close'].iloc[-2] >= recent_support:
            # Volume confirmation
            if 'volume' in df.columns:
                avg_volume = df['volume'].tail(20).mean()
                if avg_volume > 0 and df['volume'].iloc[-1] > avg_volume * 1.2:
                    breaks.iloc[-1] = 1
        
        return breaks

    def _detect_false_breakout_bullish(self, df):
        """Detect false breakout bullish (spring) patterns"""
        patterns = pd.Series(0, index=df.index)
        
        if len(df) < 10 or 'swing_low' not in df.columns:
            return patterns
        
        # Look for break below support followed by quick recovery
        swing_lows = df[df['swing_low'] == 1]
        if len(swing_lows) < 2:
            return patterns
        
        recent_support = swing_lows['low'].tail(2).iloc[0]
        
        # Check if price broke below support but closed back above
        for i in range(2, min(10, len(df))):
            if (df['low'].iloc[-i] < recent_support and 
                df['close'].iloc[-i] > recent_support and
                df['close'].iloc[-1] > recent_support):
                patterns.iloc[-1] = 1
                break
        
        return patterns

    def _detect_false_breakout_bearish(self, df):
        """Detect false breakout bearish (upthrust) patterns"""
        patterns = pd.Series(0, index=df.index)
        
        if len(df) < 10 or 'swing_high' not in df.columns:
            return patterns
        
        # Look for break above resistance followed by quick rejection
        swing_highs = df[df['swing_high'] == 1]
        if len(swing_highs) < 2:
            return patterns
        
        recent_resistance = swing_highs['high'].tail(2).iloc[0]
        
        # Check if price broke above resistance but closed back below
        for i in range(2, min(10, len(df))):
            if (df['high'].iloc[-i] > recent_resistance and 
                df['close'].iloc[-i] < recent_resistance and
                df['close'].iloc[-1] < recent_resistance):
                patterns.iloc[-1] = 1
                break
        
        return patterns

    def _detect_momentum_divergence_bullish(self, df):
        """Detect bullish momentum divergence (price makes lower low, momentum makes higher low)"""
        divergence = pd.Series(0, index=df.index)
        
        if len(df) < 20 or 'rsi' not in df.columns:
            return divergence
        
        # Look for price making lower lows but RSI making higher lows
        price_lows = df['low'].tail(10).values
        rsi_lows = df['rsi'].tail(10).values
        
        # Find the two most recent significant lows
        low_idx1, low_idx2 = -1, -2
        for i in range(1, len(price_lows)-1):
            if price_lows[i] < price_lows[i-1] and price_lows[i] < price_lows[i+1]:
                low_idx2 = low_idx1
                low_idx1 = i
        
        if low_idx1 > 0 and low_idx2 > 0:
            price_lower_low = price_lows[low_idx1] < price_lows[low_idx2]
            rsi_higher_low = rsi_lows[low_idx1] > rsi_lows[low_idx2]
            
            if price_lower_low and rsi_higher_low:
                divergence.iloc[-1] = 1
        
        return divergence

    def _detect_momentum_divergence_bearish(self, df):
        """Detect bearish momentum divergence (price makes higher high, momentum makes lower high)"""
        divergence = pd.Series(0, index=df.index)
        
        if len(df) < 20 or 'rsi' not in df.columns:
            return divergence
        
        # Look for price making higher highs but RSI making lower highs
        price_highs = df['high'].tail(10).values
        rsi_highs = df['rsi'].tail(10).values
        
        # Find the two most recent significant highs
        high_idx1, high_idx2 = -1, -2
        for i in range(1, len(price_highs)-1):
            if price_highs[i] > price_highs[i-1] and price_highs[i] > price_highs[i+1]:
                high_idx2 = high_idx1
                high_idx1 = i
        
        if high_idx1 > 0 and high_idx2 > 0:
            price_higher_high = price_highs[high_idx1] > price_highs[high_idx2]
            rsi_lower_high = rsi_highs[high_idx1] < rsi_highs[high_idx2]
            
            if price_higher_high and rsi_lower_high:
                divergence.iloc[-1] = 1
        
        return divergence

    def _detect_momentum_continuation(self, df):
        """Detect momentum continuation patterns"""
        continuation = pd.Series(0, index=df.index)
        
        if len(df) < 10:
            return continuation
        
        # Simple momentum check using price and moving averages
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            # Both MAs trending in same direction
            ema_20_up = df['ema_20'].iloc[-1] > df['ema_20'].iloc[-5]
            ema_50_up = df['ema_50'].iloc[-1] > df['ema_50'].iloc[-5]
            
            if ema_20_up and ema_50_up:
                continuation.iloc[-1] = 1  # Bullish continuation
            elif not ema_20_up and not ema_50_up:
                continuation.iloc[-1] = -1  # Bearish continuation
        
        return continuation

    def _detect_volume_breakout_confirmation(self, df):
        """Detect volume confirmation for breakouts"""
        confirmation = pd.Series(0, index=df.index)
        
        if len(df) < 20 or 'volume' not in df.columns:
            return confirmation
        
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        if avg_volume > 0:
            # High volume on upward move
            if (df['close'].iloc[-1] > df['open'].iloc[-1] and 
                current_volume > avg_volume * 1.5):
                confirmation.iloc[-1] = 1
            
            # High volume on downward move  
            elif (df['close'].iloc[-1] < df['open'].iloc[-1] and 
                current_volume > avg_volume * 1.5):
                confirmation.iloc[-1] = -1
        
        return confirmation

    def _detect_volume_divergence(self, df):
        """Detect volume divergence patterns"""
        divergence = pd.Series(0, index=df.index)
        
        if len(df) < 10 or 'volume' not in df.columns:
            return divergence
        
        # Decreasing volume on pullbacks in uptrend is bullish
        if 'trend_medium' in df.columns and df['trend_medium'].iloc[-1] == 1:
            recent_volume = df['volume'].tail(5).mean()
            prev_volume = df['volume'].tail(10).head(5).mean()
            
            if prev_volume > 0 and recent_volume < prev_volume * 0.8:  # Volume drying up
                divergence.iloc[-1] = 1
        
        return divergence
    
    def detect_advanced_price_patterns(self, df):
        """
        Updated to use optimized pattern detection
        """
        return self.optimized_detect_advanced_price_patterns(df)

    def _calculate_basic_swing_points(self, df, lookback=5):
        """
        Updated to use vectorized swing analysis
        """
        df = df.copy()
        swing_analysis = self.vectorized_swing_analysis(df, lookback)
        
        df['swing_high'] = swing_analysis['swing_high']
        df['swing_low'] = swing_analysis['swing_low']
        
        return df

    def get_regime_adaptation(self, strategy_config, regime_context):
        """Safely get regime adaptations with fallback for unknown regimes, Added while unit testing."""
        adaptations = strategy_config.get('regime_adaptations', {})
        
        # Return adaptation for known regime, or fallback to 'normal' regime
        if regime_context in adaptations:
            return adaptations[regime_context]
        elif 'normal' in adaptations:
            return adaptations['normal']
        elif adaptations:  # Return first available adaptation
            return next(iter(adaptations.values()))
        else:
            # Return default adaptation
            return {
                'parameter_modifier': 1.0,
                'additional_signals': []
            }
        
    def _create_signal_mask(self, htf_df, ttf_df, ltf_df, signals_config, direction):
        """Create signal mask for given configuration"""
        signal_mask = pd.Series(True, index=ltf_df.index)
        
        for tf, signals in signals_config.items():
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            
            for signal in signals:
                if signal in df.columns:
                    states = self.get_or_compute_states(df, signal)
                    if states is not None:
                        if direction == 'bullish':
                            signal_mask &= (states == 'bullish')
                        else:
                            signal_mask &= (states == 'bearish')
        
        return signal_mask

    def identify_sideways_conditions(self, df, atr_period=14, atr_threshold=0.5):
        """Vectorized sideways market detection"""
        if not isinstance(df, pd.DataFrame):
             print(f"⚠️ identify_sideways_conditions expected DataFrame, got {type(df)}")
             return pd.Series([False] * len(df), index=df.index)
             
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=atr_period).mean().values
        
        # Avoid division by zero if close is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_pct = (atr / close) * 100
            atr_pct[~np.isfinite(atr_pct)] = 0 # Handle NaNs/Infs
            
        low_vol = atr_pct < atr_threshold
        
        rolling_high = pd.Series(high).rolling(window=20).max().values
        rolling_low = pd.Series(low).rolling(window=20).min().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            range_pct = ((rolling_high - rolling_low) / rolling_low) * 100
            range_pct[~np.isfinite(range_pct)] = 0 # Handle NaNs/Infs

        tight_range = range_pct < (self.price_threshold * 2)
        
        sideways = low_vol | tight_range
        return pd.Series(sideways, index=df.index)

    def calculate_mtf_sl_tp(self, strategy):
        """Calculate SL/TP for MTF strategy using hierarchical approach"""
        
        try:
            # Get dataframes for this strategy's timeframes
            pair = "btc_usdt"  # Adjust based on your needs
            htf_df = self.all_dataframes.get(f"{pair}_{strategy['htf_timeframe']}")
            ttf_df = self.all_dataframes.get(f"{pair}_{strategy['ttf_timeframe']}")
            ltf_df = self.all_dataframes.get(f"{pair}_{strategy['ltf_timeframe']}")
            
            if htf_df is None or ttf_df is None or ltf_df is None:
                return strategy
            
            # Use latest price as entry (in live trading, this would be current market price)
            entry_price = ltf_df['close'].iloc[-1]
            
            planner = SLTPPlanner(
                entry_price=entry_price,
                symbol=strategy.get('pair_tf', 'BTC/USDT'),
                data_by_timeframe={
                    'htf': htf_df,
                    'ttf': ttf_df,
                    'ltf': ltf_df
                }
            )
            
            # Apply MTF hierarchical levels
            planner.set_mtf_levels(
                htf_df=htf_df,
                ttf_df=ttf_df,
                ltf_df=ltf_df,
                direction=strategy['direction']
            )
            
            # Apply regime-based adjustments if available
            if 'regime_context' in strategy:
                planner.set_by_regime(
                    regime_type=strategy['regime_context'],
                    volatility_regime=strategy.get('volatility_regime', 'normal')
                )
            
            # Apply ATR-based levels as backup
            planner.set_by_atr(atr_period=14, multiplier_sl=1.5, multiplier_tp=2.5)
            
            # Validate risk-reward
            planner.validate_risk_reward(min_rr=2.0)
            
            # Get the plan
            sl_tp_plan = planner.get_plan()
            
            # Add to strategy
            strategy['sl_tp_plan'] = sl_tp_plan
            
            # Extract the best valid plan (prioritize MTF_Hierarchical)
            best_plan = None
            if 'MTF_Hierarchical' in sl_tp_plan and sl_tp_plan['MTF_Hierarchical'].get('valid', False):
                best_plan = sl_tp_plan['MTF_Hierarchical']
            else:
                # Fallback to highest RRR valid plan
                valid_plans = {k: v for k, v in sl_tp_plan.items() if v.get('valid', False) and 'RRR' in v}
                if valid_plans:
                    best_plan = max(valid_plans.values(), key=lambda x: x.get('RRR', 0))
            
            if best_plan:
                strategy['stop_loss'] = best_plan['sl']
                strategy['take_profit'] = best_plan['tp']
                strategy['risk_reward_ratio'] = best_plan.get('RRR', 0)
                
                # Add trailing stop config
                trailing_config = planner.get_trailing_config()
                if trailing_config:
                    strategy['trailing_stop_config'] = {
                        'mode': trailing_config.mode,
                        'distance': trailing_config.pct if trailing_config.mode == 'percent' else trailing_config.atr_k,
                        'be_trigger_atr': trailing_config.be_trigger_atr
                    }
            
            return strategy
            
        except Exception as e:
            print(f"⚠️ Error calculating SL/TP for {strategy.get('id')}: {e}")
            return strategy
        
    def _analyze_volatility_characteristics(self, df):
        """
        Analyze volatility characteristics for a dataframe
        """
        if len(df) < 20:
            return {'volatility_regime': 'unknown', 'volatility_score': 0}
        
        volatility_analysis = {
            'volatility_regime': 'normal_volatility',
            'volatility_score': 0,
            'atr_percentage': 0,
            'bb_width': 0,
            'volatility_trend': 'stable'
        }
        
        # ATR-based volatility analysis
        if 'atr' in df.columns:
            current_atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            atr_percentage = (current_atr / current_price) * 100
            volatility_analysis['atr_percentage'] = atr_percentage
            
            # Historical ATR context
            historical_atr = df['atr'].tail(50).median()
            historical_atr_pct = (historical_atr / current_price) * 100
            
            if atr_percentage > historical_atr_pct * 1.3:
                volatility_analysis['volatility_regime'] = 'high_volatility'
                volatility_analysis['volatility_score'] += 2
            elif atr_percentage < historical_atr_pct * 0.7:
                volatility_analysis['volatility_regime'] = 'low_volatility'
                volatility_analysis['volatility_score'] += 0
            else:
                volatility_analysis['volatility_regime'] = 'normal_volatility'
                volatility_analysis['volatility_score'] += 1
        
        # Bollinger Band width analysis
        if 'bb_width' in df.columns:
            current_bb_width = df['bb_width'].iloc[-1]
            historical_bb_width = df['bb_width'].tail(50).median()
            volatility_analysis['bb_width'] = current_bb_width
            
            if current_bb_width > historical_bb_width * 1.4:
                volatility_analysis['volatility_score'] += 2
            elif current_bb_width < historical_bb_width * 0.6:
                volatility_analysis['volatility_score'] += 0
            else:
                volatility_analysis['volatility_score'] += 1
        
        # Volatility trend
        if len(df) >= 30:
            recent_volatility = df['atr'].tail(10).mean() if 'atr' in df.columns else 0
            previous_volatility = df['atr'].tail(30).head(20).mean() if 'atr' in df.columns else 0
            
            if recent_volatility > previous_volatility * 1.2:
                volatility_analysis['volatility_trend'] = 'increasing'
            elif recent_volatility < previous_volatility * 0.8:
                volatility_analysis['volatility_trend'] = 'decreasing'
        
        return volatility_analysis

    def _analyze_momentum_cascade(self, htf_df, ttf_df, ltf_df):
        """
        Analyze momentum cascade across timeframes
        """
        cascade_analysis = {
            'primary_flow_direction': 'mixed',
            'momentum_strength': 'weak',
            'cascade_score': 0,
            'timeframe_momentum': {},
            'alignment_score': 0
        }
        
        # Analyze momentum for each timeframe
        htf_momentum = self._analyze_timeframe_momentum(htf_df)
        ttf_momentum = self._analyze_timeframe_momentum(ttf_df)
        ltf_momentum = self._analyze_timeframe_momentum(ltf_df)
        
        cascade_analysis['timeframe_momentum'] = {
            'htf': htf_momentum['primary_direction'],
            'ttf': ttf_momentum['primary_direction'],
            'ltf': ltf_momentum['primary_direction']
        }
        
        # Determine flow direction
        momentum_strengths = {
            'htf': htf_momentum['strength_score'],
            'ttf': ttf_momentum['strength_score'],
            'ltf': ltf_momentum['strength_score']
        }
        
        # Check for cascading momentum (HTF → TTF → LTF)
        if (htf_momentum['primary_direction'] == ttf_momentum['primary_direction'] == 
            ltf_momentum['primary_direction']):
            cascade_analysis['primary_flow_direction'] = 'htf_to_ltf'
            cascade_analysis['cascade_score'] = min(momentum_strengths.values())
            cascade_analysis['alignment_score'] = 100
        # Check for reversal cascading (LTF → TTF → HTF)
        elif (ltf_momentum['primary_direction'] == ttf_momentum['primary_direction'] and
            ltf_momentum['strength_score'] > htf_momentum['strength_score']):
            cascade_analysis['primary_flow_direction'] = 'ltf_to_htf'
            cascade_analysis['cascade_score'] = ltf_momentum['strength_score']
            cascade_analysis['alignment_score'] = 80
        else:
            cascade_analysis['primary_flow_direction'] = 'mixed'
            cascade_analysis['cascade_score'] = 0
            cascade_analysis['alignment_score'] = 50
        
        # Determine overall momentum strength
        avg_strength = sum(momentum_strengths.values()) / 3
        if avg_strength >= 80:
            cascade_analysis['momentum_strength'] = 'strong'
        elif avg_strength >= 60:
            cascade_analysis['momentum_strength'] = 'moderate'
        elif avg_strength >= 40:
            cascade_analysis['momentum_strength'] = 'weak'
        else:
            cascade_analysis['momentum_strength'] = 'reversing'
        
        return cascade_analysis
    
    def _analyze_timeframe_momentum(self, df):
        """Helper for Mode N: Analyzes momentum for a single timeframe."""
        if 'rsi' not in df.columns or 'macd' not in df.columns:
            return 0 # Neutral
        
        # Get last bar data
        rsi_val = df['rsi'].iloc[-1]
        macd_val = df['macd'].iloc[-1]
        macd_signal_val = df['macd_signal'].iloc[-1]

        score = 0
        if rsi_val > 55 and macd_val > macd_signal_val:
            score = 1 # Bullish
        elif rsi_val < 45 and macd_val < macd_signal_val:
            score = -1 # Bearish
            
        return score

    def _analyze_momentum_cascade(self, htf_df, ttf_df, ltf_df):
        """Helper for Mode N: Analyzes momentum flow across timeframes."""
        htf_mom = self._analyze_timeframe_momentum(htf_df)
        ttf_mom = self._analyze_timeframe_momentum(ttf_df)
        ltf_mom = self._analyze_timeframe_momentum(ltf_df)
        
        momentum_flows = {
            'htf_momentum': 'bullish' if htf_mom > 0 else 'bearish' if htf_mom < 0 else 'neutral',
            'ttf_momentum': 'bullish' if ttf_mom > 0 else 'bearish' if ttf_mom < 0 else 'neutral',
            'ltf_momentum': 'bullish' if ltf_mom > 0 else 'bearish' if ltf_mom < 0 else 'neutral',
        }
        
        if htf_mom == ttf_mom == ltf_mom:
            momentum_flows['primary_flow_direction'] = 'htf_to_ltf'
            momentum_flows['momentum_strength'] = 'strong' if htf_mom != 0 else 'neutral'
            momentum_flows['cascade_score'] = 100
        elif ltf_mom != 0 and ttf_mom == ltf_mom:
            momentum_flows['primary_flow_direction'] = 'ltf_to_htf' # Reversal
            momentum_flows['momentum_strength'] = 'reversing'
            momentum_flows['cascade_score'] = 50
        else:
            momentum_flows['primary_flow_direction'] = 'mixed'
            momentum_flows['momentum_strength'] = 'conflicting'
            momentum_flows['cascade_score'] = 0
            
        return momentum_flows
    
    def _identify_regime_contexts(self, htf_df, ttf_df, ltf_df):
        """
        Helper for Mode K.
        FIXED to read the 'historical_regime' column from the DataFrames,
        not a dict.
        """
        try:
            # Get the regime from the *last bar* of the LTF dataframe
            ltf_regime_val = ltf_df['historical_regime'].iloc[-1]
        except (KeyError, IndexError):
            return ['unknown'] # Fallback

        # Simple logic: use the LTF regime as the primary context
        if 'trending' in str(ltf_regime_val):
            return ['trending']
        if 'ranging' in str(ltf_regime_val):
            return ['ranging']
        if 'transition' in str(ltf_regime_val):
            return ['transition']
        
        return ['unknown']

    def get_regime_adaptation(self, strategy_config, regime_context):
        """
        Helper for Mode K.
        Safely gets the adaptation for the current regime.
        """
        if regime_context in strategy_config['regime_adaptations']:
            return strategy_config['regime_adaptations'][regime_context]
        
        # Fallback to the first available adaptation (e.g., 'trending')
        first_key = list(strategy_config['regime_adaptations'].keys())[0]
        return strategy_config['regime_adaptations'][first_key]

    def _build_adaptive_signal_set(self, strategy_config, regime_context, htf_df, ttf_df, ltf_df):
        """Helper for Mode K: Builds the signal set."""
        adaptation = self.get_regime_adaptation(strategy_config, regime_context)
        
        return {
            'htf': strategy_config['core_signals']['htf'] + adaptation.get('additional_signals', []),
            'ttf': strategy_config['core_signals']['ttf'],
            'ltf': strategy_config['core_signals']['ltf'],
        }

    def _create_adaptive_signal_mask(self, adaptive_signals, htf_df, ttf_df, ltf_df, direction):
        """Helper for Mode K: Creates the signal mask."""
        active_mask = pd.Series(True, index=ltf_df.index)
        direction_str = 'bullish' if direction == 'bullish' else 'bearish'

        for tf, signals in adaptive_signals.items():
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            for sig in signals:
                if sig in df.columns:
                    states = self.get_or_compute_states(df, sig)
                    if states is not None:
                        active_mask &= (states == direction_str)
        return active_mask
