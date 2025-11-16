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


class TrendAnalysisSystem:
    """
    Advanced trend detection and classification system
    Uses multiple methodologies to determine trend strength and direction
    """
    
    def __init__(self):
        self.trend_cache = {}
    
    def classify_mtf_trend_strength(self, htf_df, ttf_df, ltf_df):
        """
        Comprehensive multi-timeframe trend classification
        Returns detailed trend analysis across all timeframes
        """
        trend_analysis = {
            'htf_analysis': self._analyze_trend_comprehensive(htf_df),
            'ttf_analysis': self._analyze_trend_comprehensive(ttf_df),
            'ltf_analysis': self._analyze_trend_comprehensive(ltf_df),
            'overall_trend': self._synthesize_mtf_trend(htf_df, ttf_df, ltf_df),
            'trend_alignment_score': 0,
            'structure_quality_score': 0,
            'momentum_alignment': 'neutral'
        }
        
        # Calculate alignment scores
        trend_analysis['trend_alignment_score'] = self._calculate_trend_alignment(
            trend_analysis['htf_analysis'],
            trend_analysis['ttf_analysis'], 
            trend_analysis['ltf_analysis']
        )
        
        trend_analysis['structure_quality_score'] = self._calculate_structure_quality(
            htf_df, ttf_df, ltf_df
        )
        
        trend_analysis['momentum_alignment'] = self._analyze_momentum_alignment(
            htf_df, ttf_df, ltf_df
        )
        
        return trend_analysis
    
    def _analyze_trend_comprehensive(self, df):
        """
        Comprehensive trend analysis using multiple methodologies
        """
        analysis = {
            'primary_trend': 'neutral',
            'trend_strength': 0,  # 0-100
            'trend_duration_bars': 0,
            'structure_quality': 'unknown',
            'swing_structure': 'unknown',
            'momentum_bias': 'neutral',
            'key_levels': {},
            'breakout_potential': 0
        }
        
        if len(df) < 50:
            return analysis
        
        # 1. PRICE STRUCTURE ANALYSIS (Most Important)
        structure_analysis = self._analyze_price_structure(df)
        analysis.update(structure_analysis)
        
        # 2. MOVING AVERAGE ANALYSIS
        ma_analysis = self._analyze_moving_averages(df)
        analysis.update(ma_analysis)
        
        # 3. MOMENTUM ANALYSIS
        momentum_analysis = self._analyze_momentum_indicators(df)
        analysis.update(momentum_analysis)
        
        # 4. VOLUME ANALYSIS
        volume_analysis = self._analyze_volume_profile(df)
        analysis.update(volume_analysis)
        
        # 5. SYNTHESIZE PRIMARY TREND
        analysis['primary_trend'] = self._synthesize_primary_trend(analysis)
        analysis['trend_strength'] = self._calculate_trend_strength_score(analysis)
        
        return analysis
    
    def _analyze_price_structure(self, df):
        """Analyze trend using pure price action and swing points"""
        analysis = {
            'structure_trend': 'neutral',
            'swing_structure': 'unclear',
            'higher_highs_count': 0,
            'higher_lows_count': 0, 
            'lower_highs_count': 0,
            'lower_lows_count': 0,
            'structure_quality': 'poor'
        }
        
        # Get swing points
        swing_highs = df[df['swing_high'] == 1]
        swing_lows = df[df['swing_low'] == 1]
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return analysis
        
        # Analyze last 5 swings for structure
        recent_highs = swing_highs.tail(5)
        recent_lows = swing_lows.tail(5)
        
        # Count higher highs/lower lows
        for i in range(1, len(recent_highs)):
            if recent_highs.iloc[i]['high'] > recent_highs.iloc[i-1]['high']:
                analysis['higher_highs_count'] += 1
            else:
                analysis['lower_highs_count'] += 1
        
        for i in range(1, len(recent_lows)):
            if recent_lows.iloc[i]['low'] > recent_lows.iloc[i-1]['low']:
                analysis['higher_lows_count'] += 1
            else:
                analysis['lower_lows_count'] += 1
        
        # Determine structure
        if analysis['higher_highs_count'] >= 3 and analysis['higher_lows_count'] >= 3:
            analysis['structure_trend'] = 'strong_uptrend'
            analysis['swing_structure'] = 'hh_hl'
        elif analysis['higher_highs_count'] >= 2 and analysis['higher_lows_count'] >= 2:
            analysis['structure_trend'] = 'uptrend'
            analysis['swing_structure'] = 'hh_hl'
        elif analysis['lower_highs_count'] >= 3 and analysis['lower_lows_count'] >= 3:
            analysis['structure_trend'] = 'strong_downtrend' 
            analysis['swing_structure'] = 'lh_ll'
        elif analysis['lower_highs_count'] >= 2 and analysis['lower_lows_count'] >= 2:
            analysis['structure_trend'] = 'downtrend'
            analysis['swing_structure'] = 'lh_ll'
        else:
            analysis['structure_trend'] = 'ranging'
            analysis['swing_structure'] = 'mixed'
        
        # Assess structure quality
        total_swings = (analysis['higher_highs_count'] + analysis['lower_highs_count'] + 
                    analysis['higher_lows_count'] + analysis['lower_lows_count'])
        
        if total_swings >= 8:
            consistency = max(
                analysis['higher_highs_count'] + analysis['higher_lows_count'],
                analysis['lower_highs_count'] + analysis['lower_lows_count']
            ) / total_swings
            
            if consistency >= 0.8:
                analysis['structure_quality'] = 'excellent'
            elif consistency >= 0.6:
                analysis['structure_quality'] = 'good'
            else:
                analysis['structure_quality'] = 'poor'
        
        return analysis
    
    def _analyze_moving_averages(self, df):
        """Analyze trend using multiple moving average systems"""
        analysis = {
            'ma_trend': 'neutral',
            'ma_alignment': 'mixed',
            'ma_slope': 'flat',
            'ma_distance_score': 0
        }
        
        # Check if we have necessary MA columns
        required_mas = ['ema_20', 'ema_50', 'ema_200', 'sma_50', 'sma_200']
        if not all(ma in df.columns for ma in required_mas):
            return analysis
        
        # MA Alignment check
        current_price = df['close'].iloc[-1]
        ma_bullish = 0
        ma_total = 0
        
        for ma in required_mas:
            ma_value = df[ma].iloc[-1]
            if current_price > ma_value:
                ma_bullish += 1
            ma_total += 1
        
        ma_alignment_ratio = ma_bullish / ma_total
        
        if ma_alignment_ratio >= 0.8:
            analysis['ma_trend'] = 'strong_uptrend'
            analysis['ma_alignment'] = 'bullish'
        elif ma_alignment_ratio >= 0.6:
            analysis['ma_trend'] = 'uptrend'
            analysis['ma_alignment'] = 'bullish'
        elif ma_alignment_ratio <= 0.2:
            analysis['ma_trend'] = 'strong_downtrend'
            analysis['ma_alignment'] = 'bearish'
        elif ma_alignment_ratio <= 0.4:
            analysis['ma_trend'] = 'downtrend'
            analysis['ma_alignment'] = 'bearish'
        else:
            analysis['ma_trend'] = 'neutral'
            analysis['ma_alignment'] = 'mixed'
        
        # MA Slope analysis
        ema_20_slope = self._calculate_ma_slope(df, 'ema_20', 5)
        ema_50_slope = self._calculate_ma_slope(df, 'ema_50', 10)
        
        if ema_20_slope > 0.001 and ema_50_slope > 0.001:
            analysis['ma_slope'] = 'rising'
        elif ema_20_slope < -0.001 and ema_50_slope < -0.001:
            analysis['ma_slope'] = 'falling'
        else:
            analysis['ma_slope'] = 'flat'
        
        # Distance from key MAs
        distance_50 = abs(current_price - df['ema_50'].iloc[-1]) / df['ema_50'].iloc[-1]
        distance_200 = abs(current_price - df['ema_200'].iloc[-1]) / df['ema_200'].iloc[-1]
        analysis['ma_distance_score'] = (distance_50 + distance_200) / 2
        
        return analysis
    
    def _calculate_ma_slope(self, df, ma_column, period):
        """Calculate the slope of a moving average"""
        if len(df) < period + 1:
            return 0
        
        current_ma = df[ma_column].iloc[-1]
        previous_ma = df[ma_column].iloc[-period]
        
        return (current_ma - previous_ma) / previous_ma
    
    def _analyze_momentum_indicators(self, df):
        """Analyze trend using momentum oscillators"""
        analysis = {
            'momentum_trend': 'neutral',
            'momentum_strength': 'neutral',
            'oscillator_alignment': 'mixed',
            'divergence_present': False
        }
        
        momentum_indicators = {
            'rsi': (30, 70),
            'macd': (0, 0),  # Bullish above 0
            'stoch_k': (20, 80),
            'cci': (-100, 100),
            'adx': (25, 25)  # Trend strength
        }
        
        bullish_signals = 0
        total_indicators = 0
        
        for indicator, (oversold, overbought) in momentum_indicators.items():
            if indicator not in df.columns:
                continue
                
            current_value = df[indicator].iloc[-1]
            total_indicators += 1
            
            if indicator == 'macd':
                if current_value > 0:
                    bullish_signals += 1
            elif indicator == 'adx':
                if current_value > overbought:  # Strong trend
                    bullish_signals += 0.5  # Partial credit for trend strength
            else:
                if current_value > oversold and current_value < (oversold + overbought) / 2:
                    bullish_signals += 1  # Bullish but not overbought
        
        if total_indicators > 0:
            momentum_score = bullish_signals / total_indicators
            
            if momentum_score >= 0.7:
                analysis['momentum_trend'] = 'bullish'
                analysis['momentum_strength'] = 'strong'
            elif momentum_score >= 0.5:
                analysis['momentum_trend'] = 'bullish'
                analysis['momentum_strength'] = 'moderate'
            elif momentum_score <= 0.3:
                analysis['momentum_trend'] = 'bearish'
                analysis['momentum_strength'] = 'strong'
            elif momentum_score <= 0.5:
                analysis['momentum_trend'] = 'bearish'
                analysis['momentum_strength'] = 'moderate'
        
        # Check for divergence
        analysis['divergence_present'] = self._check_momentum_divergence(df)
        
        return analysis
    
    def _check_momentum_divergence(self, df):
        """Check for momentum divergence patterns"""
        if len(df) < 20 or 'rsi' not in df.columns:
            return False
        
        # Simple RSI divergence check
        price_highs = df['high'].tail(10)
        rsi_highs = df['rsi'].tail(10)
        
        # Find peaks in price and RSI
        price_peak_idx = price_highs.idxmax()
        rsi_peak_idx = rsi_highs.idxmax()
        
        # If price made new high but RSI didn't (bearish divergence)
        if (price_peak_idx == price_highs.index[-1] and 
            rsi_peak_idx != rsi_highs.index[-1] and
            rsi_highs.iloc[-1] < rsi_highs.max() * 0.95):
            return True
        
        return False
    
    def _analyze_volume_profile(self, df):
        """Analyze volume characteristics for trend confirmation"""
        analysis = {
            'volume_trend': 'neutral',
            'volume_characteristics': 'normal',
            'accumulation_distribution': 'neutral'
        }
        
        if 'volume' not in df.columns or 'obv' not in df.columns:
            return analysis
        
        # Volume trend analysis
        volume_sma_20 = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        volume_ratio = current_volume / volume_sma_20
        
        if volume_ratio > 1.5:
            analysis['volume_characteristics'] = 'high_volume'
        elif volume_ratio < 0.7:
            analysis['volume_characteristics'] = 'low_volume'
        
        # OBV trend analysis
        obv_trend = self._calculate_obv_trend(df)
        analysis['accumulation_distribution'] = obv_trend
        
        # Volume-Price relationship
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-5]) / df['volume'].iloc[-5]
        
        if price_change > 0.02 and volume_change > 0.3:
            analysis['volume_trend'] = 'bullish_confirmation'
        elif price_change < -0.02 and volume_change > 0.3:
            analysis['volume_trend'] = 'bearish_confirmation'
        elif price_change > 0.02 and volume_change < -0.2:
            analysis['volume_trend'] = 'divergence_warning'
        
        return analysis
    
    def _calculate_obv_trend(self, df, period=20):
        """Calculate OBV trend direction"""
        if len(df) < period:
            return 'neutral'
        
        obv_values = df['obv'].tail(period)
        obv_slope = (obv_values.iloc[-1] - obv_values.iloc[0]) / obv_values.iloc[0]
        
        if obv_slope > 0.05:
            return 'accumulation'
        elif obv_slope < -0.05:
            return 'distribution'
        else:
            return 'neutral'
    
    def _synthesize_primary_trend(self, analysis):
        """Synthesize all trend analysis into primary trend direction"""
        trend_scores = {
            'strong_uptrend': 0,
            'uptrend': 0,
            'neutral': 0,
            'downtrend': 0,
            'strong_downtrend': 0
        }
        
        # Structure analysis weighting (40%)
        structure_map = {
            'strong_uptrend': 'strong_uptrend',
            'uptrend': 'uptrend',
            'neutral': 'neutral',
            'downtrend': 'downtrend',
            'strong_downtrend': 'strong_downtrend'
        }
        
        if analysis['structure_trend'] in structure_map:
            trend_scores[structure_map[analysis['structure_trend']]] += 4
        
        # MA analysis weighting (30%)
        ma_map = {
            'strong_uptrend': 'strong_uptrend',
            'uptrend': 'uptrend',
            'neutral': 'neutral',
            'downtrend': 'downtrend',
            'strong_downtrend': 'strong_downtrend'
        }
        
        if analysis['ma_trend'] in ma_map:
            trend_scores[ma_map[analysis['ma_trend']]] += 3
        
        # Momentum analysis weighting (20%)
        momentum_map = {
            'bullish': 'uptrend',
            'bearish': 'downtrend',
            'neutral': 'neutral'
        }
        
        if analysis['momentum_trend'] in momentum_map:
            mapped_trend = momentum_map[analysis['momentum_trend']]
            if analysis['momentum_strength'] == 'strong':
                mapped_trend = 'strong_' + mapped_trend
            trend_scores[mapped_trend] += 2
        
        # Volume analysis weighting (10%)
        volume_map = {
            'bullish_confirmation': 'uptrend',
            'bearish_confirmation': 'downtrend',
            'neutral': 'neutral'
        }
        
        if analysis['volume_trend'] in volume_map:
            trend_scores[volume_map[analysis['volume_trend']]] += 1
        
        # Determine primary trend
        primary_trend = max(trend_scores, key=trend_scores.get)
        
        return primary_trend
    
    def _calculate_trend_strength_score(self, analysis):
        """Calculate overall trend strength score (0-100)"""
        strength_components = []
        
        # Structure quality (30%)
        structure_score_map = {
            'excellent': 100,
            'good': 75,
            'poor': 25,
            'unknown': 0
        }
        strength_components.append(structure_score_map.get(analysis['structure_quality'], 0) * 0.3)
        
        # MA alignment (25%)
        ma_score_map = {
            'bullish': 100,
            'bearish': 100,  # Strong alignment either way
            'mixed': 50,
            'neutral': 25
        }
        strength_components.append(ma_score_map.get(analysis['ma_alignment'], 0) * 0.25)
        
        # Momentum strength (25%)
        momentum_score_map = {
            'strong': 100,
            'moderate': 75,
            'neutral': 50,
            'weak': 25
        }
        strength_components.append(momentum_score_map.get(analysis['momentum_strength'], 0) * 0.25)
        
        # Volume confirmation (20%)
        volume_score_map = {
            'bullish_confirmation': 100,
            'bearish_confirmation': 100,
            'neutral': 50,
            'divergence_warning': 25
        }
        strength_components.append(volume_score_map.get(analysis['volume_trend'], 0) * 0.2)
        
        return sum(strength_components)
    
    def _synthesize_mtf_trend(self, htf_df, ttf_df, ltf_df):
        """Synthesize multi-timeframe trend into single analysis"""
        htf_trend = self._analyze_trend_comprehensive(htf_df)
        ttf_trend = self._analyze_trend_comprehensive(ttf_df)
        ltf_trend = self._analyze_trend_comprehensive(ltf_df)
        
        # Weighted synthesis (HTF:50%, TTF:30%, LTF:20%)
        trend_weights = {
            'strong_uptrend': 0,
            'uptrend': 0,
            'neutral': 0,
            'downtrend': 0,
            'strong_downtrend': 0
        }
        
        # HTF weighting (50%)
        trend_weights[htf_trend['primary_trend']] += 5
        
        # TTF weighting (30%)
        trend_weights[ttf_trend['primary_trend']] += 3
        
        # LTF weighting (20%)
        trend_weights[ltf_trend['primary_trend']] += 2
        
        primary_trend = max(trend_weights, key=trend_weights.get)
        
        # Calculate overall strength (average)
        overall_strength = (
            htf_trend['trend_strength'] * 0.5 +
            ttf_trend['trend_strength'] * 0.3 +
            ltf_trend['trend_strength'] * 0.2
        )
        
        return {
            'primary_trend': primary_trend,
            'trend_strength': overall_strength,
            'htf_trend': htf_trend['primary_trend'],
            'ttf_trend': ttf_trend['primary_trend'],
            'ltf_trend': ltf_trend['primary_trend'],
            'alignment': self._assess_trend_alignment(htf_trend, ttf_trend, ltf_trend)
        }
    
    def _assess_trend_alignment(self, htf_trend, ttf_trend, ltf_trend):
        """Assess how well timeframes are aligned"""
        trends = [htf_trend['primary_trend'], ttf_trend['primary_trend'], ltf_trend['primary_trend']]
        
        # Convert to numerical scores for comparison
        trend_scores = []
        for trend in trends:
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
            return 'perfect_alignment'
        elif alignment_std <= 1.0:
            return 'good_alignment'
        elif alignment_std <= 1.5:
            return 'partial_alignment'
        else:
            return 'conflicting'
    
    def _calculate_trend_alignment(self, htf_analysis, ttf_analysis, ltf_analysis):
        """Calculate numerical trend alignment score (0-100)"""
        trends = [htf_analysis['primary_trend'], ttf_analysis['primary_trend'], ltf_analysis['primary_trend']]
        
        # Count matching trends
        trend_counts = {}
        for trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        max_count = max(trend_counts.values())
        alignment_ratio = max_count / 3
        
        return alignment_ratio * 100
    
    def _calculate_structure_quality(self, htf_df, ttf_df, ltf_df):
        """Calculate overall structure quality score"""
        htf_structure = self._analyze_price_structure(htf_df)
        ttf_structure = self._analyze_price_structure(ttf_df)
        ltf_structure = self._analyze_price_structure(ltf_df)
        
        structure_scores = {
            'excellent': 100,
            'good': 75,
            'poor': 25,
            'unknown': 0
        }
        
        avg_score = (
            structure_scores.get(htf_structure['structure_quality'], 0) * 0.5 +
            structure_scores.get(ttf_structure['structure_quality'], 0) * 0.3 +
            structure_scores.get(ltf_structure['structure_quality'], 0) * 0.2
        )
        
        return avg_score
    
    def _analyze_momentum_alignment(self, htf_df, ttf_df, ltf_df):
        """Analyze momentum alignment across timeframes"""
        htf_momentum = self._analyze_momentum_indicators(htf_df)
        ttf_momentum = self._analyze_momentum_indicators(ttf_df)
        ltf_momentum = self._analyze_momentum_indicators(ltf_df)
        
        momentums = [htf_momentum['momentum_trend'], ttf_momentum['momentum_trend'], ltf_momentum['momentum_trend']]
        
        bullish_count = sum(1 for m in momentums if 'bullish' in m)
        bearish_count = sum(1 for m in momentums if 'bearish' in m)
        
        if bullish_count >= 2:
            return 'bullish_alignment'
        elif bearish_count >= 2:
            return 'bearish_alignment'
        else:
            return 'mixed_momentum'

