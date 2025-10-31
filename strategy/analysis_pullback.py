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

try:
    from discovery_mapping import (
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
    print("Warning: discovery_mapping.py not found. System cannot function without it.")
    MAPPER_AVAILABLE = False
    exit(1)

class PullbackAnalysisSystem:
        """
        Advanced pullback quality analysis and scoring system
        Evaluates pullbacks across multiple dimensions for quality assessment
        """
        
        def __init__(self):
            self.pullback_cache = {}
        
        def analyze_pullback_quality(self, df, trend_direction='bullish'):
            """
            Comprehensive pullback quality analysis
            Returns detailed quality assessment and scoring
            """
            quality_analysis = {
                'overall_score': 0,
                'quality_grade': 'F',
                'depth_score': 0,
                'structure_score': 0,
                'volume_score': 0,
                'momentum_score': 0,
                'fibonacci_score': 0,
                'time_score': 0,
                'risk_reward_score': 0,
                'recommendation': 'avoid',
                'confidence': 'low'
            }
            
            if len(df) < 30:
                return quality_analysis
            
            # 1. PULLBACK DEPTH ANALYSIS
            depth_analysis = self._analyze_pullback_depth(df, trend_direction)
            quality_analysis.update(depth_analysis)
            
            # 2. STRUCTURE ANALYSIS
            structure_analysis = self._analyze_pullback_structure(df, trend_direction)
            quality_analysis.update(structure_analysis)
            
            # 3. VOLUME ANALYSIS
            volume_analysis = self._analyze_pullback_volume(df, trend_direction)
            quality_analysis.update(volume_analysis)
            
            # 4. MOMENTUM ANALYSIS
            momentum_analysis = self._analyze_pullback_momentum(df, trend_direction)
            quality_analysis.update(momentum_analysis)
            
            # 5. FIBONACCI ANALYSIS
            fibonacci_analysis = self._analyze_fibonacci_levels(df, trend_direction)
            quality_analysis.update(fibonacci_analysis)
            
            # 6. TIME ANALYSIS
            time_analysis = self._analyze_pullback_time(df, trend_direction)
            quality_analysis.update(time_analysis)
            
            # 7. RISK-REWARD ANALYSIS
            risk_reward_analysis = self._analyze_risk_reward(df, trend_direction)
            quality_analysis.update(risk_reward_analysis)
            
            # CALCULATE OVERALL SCORE
            overall_score = self._calculate_overall_score(quality_analysis)
            quality_analysis['overall_score'] = overall_score
            quality_analysis['quality_grade'] = self._convert_score_to_grade(overall_score)
            quality_analysis['recommendation'] = self._generate_recommendation(overall_score)
            quality_analysis['confidence'] = self._assess_confidence(quality_analysis)
            
            return quality_analysis
        
        def _analyze_pullback_depth(self, df, trend_direction):
            """Analyze pullback depth and retracement percentage"""
            analysis = {
                'depth_score': 0,
                'retracement_percentage': 0,
                'depth_quality': 'poor',
                'ideal_retracement': False
            }
            
            # Calculate retracement from last swing
            if 'last_swing_high' in df.columns and 'last_swing_low' in df.columns:
                if trend_direction == 'bullish':
                    swing_high = df['last_swing_high'].iloc[-1]
                    swing_low = df['last_swing_low'].iloc[-1]
                    current_price = df['close'].iloc[-1]
                    
                    if swing_high > swing_low:  # Valid swing data
                        total_move = swing_high - swing_low
                        retracement = (swing_high - current_price) / total_move
                        analysis['retracement_percentage'] = retracement * 100
                        
                        # Score based on ideal retracement levels
                        if 38.2 <= retracement * 100 <= 61.8:
                            analysis['depth_score'] = 100
                            analysis['depth_quality'] = 'excellent'
                            analysis['ideal_retracement'] = True
                        elif 23.6 <= retracement * 100 <= 78.6:
                            analysis['depth_score'] = 75
                            analysis['depth_quality'] = 'good'
                        elif retracement * 100 < 23.6:
                            analysis['depth_score'] = 50
                            analysis['depth_quality'] = 'shallow'
                        else:  # > 78.6%
                            analysis['depth_score'] = 25
                            analysis['depth_quality'] = 'deep'
            
            return analysis
        
        def _analyze_pullback_structure(self, df, trend_direction):
            """Analyze the structural quality of the pullback"""
            analysis = {
                'structure_score': 0,
                'structure_quality': 'poor',
                'orderly_decline': False,
                'support_levels': 0,
                'break_of_structure': False
            }
            
            # Check for orderly decline (smooth pullback vs chaotic)
            if len(df) >= 10:
                recent_lows = df['low'].tail(10)
                recent_highs = df['high'].tail(10)
                
                # Calculate volatility during pullback
                price_range = (recent_highs.max() - recent_lows.min()) / recent_lows.min()
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                current_price = df['close'].iloc[-1]
                
                if atr > 0:
                    normalized_volatility = price_range / (atr / current_price)
                    
                    if normalized_volatility < 2.0:
                        analysis['orderly_decline'] = True
                        analysis['structure_score'] += 25
            
            # Count support levels (Fibonacci, moving averages, previous swings)
            support_count = 0
            
            # Fibonacci support
            fib_columns = [col for col in df.columns if 'fib_' in col and col != 'fib_levels']
            for fib_col in fib_columns:
                if fib_col in df.columns:
                    fib_value = df[fib_col].iloc[-1]
                    price_diff = abs(current_price - fib_value) / current_price
                    if price_diff < 0.005:  # Within 0.5%
                        support_count += 1
            
            # Moving average support
            ma_columns = ['ema_20', 'ema_50', 'sma_50', 'sma_200']
            for ma_col in ma_columns:
                if ma_col in df.columns:
                    ma_value = df[ma_col].iloc[-1]
                    price_diff = abs(current_price - ma_value) / current_price
                    if price_diff < 0.01:  # Within 1%
                        support_count += 1
            
            analysis['support_levels'] = support_count
            analysis['structure_score'] += support_count * 15  # Up to 45 points
            
            # Check for break of structure
            if 'swing_low' in df.columns:
                recent_swing_lows = df[df['swing_low'] == 1].tail(3)
                if len(recent_swing_lows) >= 2:
                    last_swing_low = recent_swing_lows['low'].iloc[-1]
                    if current_price < last_swing_low:
                        analysis['break_of_structure'] = True
                        analysis['structure_score'] -= 20
            
            # Cap structure score
            analysis['structure_score'] = max(0, min(100, analysis['structure_score']))
            
            # Assign quality rating
            if analysis['structure_score'] >= 80:
                analysis['structure_quality'] = 'excellent'
            elif analysis['structure_score'] >= 60:
                analysis['structure_quality'] = 'good'
            elif analysis['structure_score'] >= 40:
                analysis['structure_quality'] = 'fair'
            else:
                analysis['structure_quality'] = 'poor'
            
            return analysis
        
        def _analyze_pullback_volume(self, df, trend_direction):
            """Analyze volume characteristics during pullback"""
            analysis = {
                'volume_score': 0,
                'volume_profile': 'neutral',
                'volume_divergence': False,
                'volume_contraction': False
            }
            
            if 'volume' not in df.columns:
                return analysis
            
            # Analyze volume over last 10 bars
            recent_volume = df['volume'].tail(10)
            previous_volume = df['volume'].tail(20).head(10)
            
            volume_contraction = recent_volume.mean() < previous_volume.mean() * 0.8
            analysis['volume_contraction'] = volume_contraction
            
            if volume_contraction:
                analysis['volume_score'] += 40
                analysis['volume_profile'] = 'contracting'
            
            # Check for volume divergence (price down, volume down = good for pullback)
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            volume_change = (recent_volume.mean() - previous_volume.mean()) / previous_volume.mean()
            
            if price_change < 0 and volume_change < 0:
                analysis['volume_divergence'] = True
                analysis['volume_score'] += 30
            
            # Volume on potential reversal bars
            if len(df) >= 3:
                recent_closes = df['close'].tail(3)
                recent_volumes = df['volume'].tail(3)
                
                # Look for reversal bar with high volume
                if (recent_closes.iloc[-1] > recent_closes.iloc[-2] and 
                    recent_volumes.iloc[-1] > recent_volumes.iloc[-2] * 1.2):
                    analysis['volume_score'] += 30
            
            analysis['volume_score'] = min(100, analysis['volume_score'])
            
            return analysis
        
        def _analyze_pullback_momentum(self, df, trend_direction):
            """Analyze momentum characteristics during pullback"""
            analysis = {
                'momentum_score': 0,
                'momentum_state': 'neutral',
                'oversold_condition': False,
                'momentum_divergence': False,
                'momentum_reversal': False
            }
            
            # RSI Analysis
            if 'rsi' in df.columns:
                current_rsi = df['rsi'].iloc[-1]
                previous_rsi = df['rsi'].iloc[-5] if len(df) >= 5 else current_rsi
                
                # Oversold condition
                if current_rsi < 30:
                    analysis['oversold_condition'] = True
                    analysis['momentum_score'] += 25
                
                # Bullish momentum divergence
                if (df['low'].iloc[-1] < df['low'].iloc[-5] and 
                    current_rsi > previous_rsi):
                    analysis['momentum_divergence'] = True
                    analysis['momentum_score'] += 35
            
            # MACD Analysis
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                current_macd = df['macd'].iloc[-1]
                current_signal = df['macd_signal'].iloc[-1]
                previous_macd = df['macd'].iloc[-3] if len(df) >= 3 else current_macd
                
                # MACD crossing above signal
                if current_macd > current_signal and previous_macd <= current_signal:
                    analysis['momentum_reversal'] = True
                    analysis['momentum_score'] += 40
            
            # Stochastic Analysis
            if 'stoch_k' in df.columns:
                current_stoch = df['stoch_k'].iloc[-1]
                if current_stoch < 20:
                    analysis['momentum_score'] += 20
                elif current_stoch > 20 and current_stoch < 50:
                    analysis['momentum_score'] += 10
            
            analysis['momentum_score'] = min(100, analysis['momentum_score'])
            
            # Momentum state classification
            if analysis['momentum_score'] >= 70:
                analysis['momentum_state'] = 'bullish_reversal'
            elif analysis['momentum_score'] >= 50:
                analysis['momentum_state'] = 'stabilizing'
            elif analysis['momentum_score'] >= 30:
                analysis['momentum_state'] = 'oversold'
            else:
                analysis['momentum_state'] = 'bearish_momentum'
            
            return analysis
        
        def _analyze_fibonacci_levels(self, df, trend_direction):
            """Analyze Fibonacci retracement and extension levels"""
            analysis = {
                'fibonacci_score': 0,
                'key_fib_levels': [],
                'fib_cluster': False,
                'extension_targets': []
            }
            
            # Check proximity to key Fibonacci levels
            fib_levels = {
                'fib_23.6': 0.236,
                'fib_38.2': 0.382, 
                'fib_50.0': 0.500,
                'fib_61.8': 0.618,
                'fib_78.6': 0.786
            }
            
            current_price = df['close'].iloc[-1]
            nearby_levels = []
            
            for fib_col, fib_value in fib_levels.items():
                if fib_col in df.columns:
                    fib_price = df[fib_col].iloc[-1]
                    price_diff = abs(current_price - fib_price) / current_price
                    
                    if price_diff < 0.005:  # Within 0.5%
                        nearby_levels.append(fib_col)
                        
                        # Score based on importance of level
                        if fib_col in ['fib_38.2', 'fib_61.8']:
                            analysis['fibonacci_score'] += 35
                        elif fib_col == 'fib_50.0':
                            analysis['fibonacci_score'] += 30
                        else:
                            analysis['fibonacci_score'] += 20
            
            analysis['key_fib_levels'] = nearby_levels
            
            # Check for Fibonacci cluster (multiple nearby levels)
            if len(nearby_levels) >= 2:
                analysis['fib_cluster'] = True
                analysis['fibonacci_score'] += 15
            
            analysis['fibonacci_score'] = min(100, analysis['fibonacci_score'])
            
            return analysis
        
        def _analyze_pullback_time(self, df, trend_direction):
            """Analyze temporal characteristics of the pullback"""
            analysis = {
                'time_score': 0,
                'pullback_duration_bars': 0,
                'time_symmetry': 'unknown',
                'maturity': 'early'
            }
            
            # Calculate pullback duration
            if 'bars_since_swing_high' in df.columns:
                pullback_bars = df['bars_since_swing_high'].iloc[-1]
                analysis['pullback_duration_bars'] = pullback_bars
                
                # Score based on ideal duration (neither too short nor too long)
                if 3 <= pullback_bars <= 8:
                    analysis['time_score'] = 100
                    analysis['maturity'] = 'ideal'
                elif 2 <= pullback_bars <= 12:
                    analysis['time_score'] = 75
                    analysis['maturity'] = 'good'
                elif pullback_bars == 1:
                    analysis['time_score'] = 50
                    analysis['maturity'] = 'early'
                else:  # > 12 bars
                    analysis['time_score'] = 25
                    analysis['maturity'] = 'extended'
            
            return analysis
        
        def _analyze_risk_reward(self, df, trend_direction):
            """Analyze risk-reward characteristics"""
            analysis = {
                'risk_reward_score': 0,
                'estimated_rr_ratio': 1.0,
                'stop_levels': [],
                'target_levels': []
            }
            
            # Simple risk-reward estimation
            if 'last_swing_high' in df.columns and 'last_swing_low' in df.columns:
                swing_high = df['last_swing_high'].iloc[-1]
                swing_low = df['last_swing_low'].iloc[-1]
                current_price = df['close'].iloc[-1]
                
                if trend_direction == 'bullish' and swing_high > swing_low:
                    # Risk (distance to stop)
                    risk_distance = current_price - swing_low
                    
                    # Reward (distance to target - previous high + extension)
                    reward_distance = swing_high - current_price
                    
                    if risk_distance > 0:
                        rr_ratio = reward_distance / risk_distance
                        analysis['estimated_rr_ratio'] = rr_ratio
                        
                        if rr_ratio >= 3.0:
                            analysis['risk_reward_score'] = 100
                        elif rr_ratio >= 2.0:
                            analysis['risk_reward_score'] = 80
                        elif rr_ratio >= 1.5:
                            analysis['risk_reward_score'] = 60
                        elif rr_ratio >= 1.0:
                            analysis['risk_reward_score'] = 40
                        else:
                            analysis['risk_reward_score'] = 20
            
            return analysis
        
        def _calculate_overall_score(self, analysis):
            """Calculate weighted overall pullback quality score"""
            weights = {
                'depth_score': 0.20,      # 20% - Retracement depth
                'structure_score': 0.25,  # 25% - Structural quality
                'volume_score': 0.15,     # 15% - Volume characteristics
                'momentum_score': 0.15,   # 15% - Momentum conditions
                'fibonacci_score': 0.10,  # 10% - Fibonacci alignment
                'time_score': 0.10,       # 10% - Temporal factors
                'risk_reward_score': 0.05 # 5%  - Risk-reward
            }
            
            overall_score = 0
            for component, weight in weights.items():
                overall_score += analysis[component] * weight
            
            return round(overall_score, 1)
        
        def _convert_score_to_grade(self, score):
            """Convert numerical score to letter grade"""
            if score >= 90:
                return 'A+'
            elif score >= 85:
                return 'A'
            elif score >= 80:
                return 'A-'
            elif score >= 75:
                return 'B+'
            elif score >= 70:
                return 'B'
            elif score >= 65:
                return 'B-'
            elif score >= 60:
                return 'C+'
            elif score >= 55:
                return 'C'
            elif score >= 50:
                return 'C-'
            elif score >= 40:
                return 'D'
            else:
                return 'F'
        
        def _generate_recommendation(self, score):
            """Generate trading recommendation based on score"""
            if score >= 80:
                return 'high_quality'
            elif score >= 70:
                return 'good_quality'
            elif score >= 60:
                return 'moderate_quality'
            elif score >= 50:
                return 'low_quality'
            else:
                return 'avoid'
        
        def _assess_confidence(self, analysis):
            """Assess confidence level in the analysis"""
            high_confidence_components = 0
            total_components = 7
            
            # Components that contribute to confidence
            if analysis['depth_score'] >= 70:
                high_confidence_components += 1
            if analysis['structure_score'] >= 70:
                high_confidence_components += 1
            if analysis['volume_score'] >= 60:
                high_confidence_components += 1
            if analysis['momentum_score'] >= 70:
                high_confidence_components += 1
            
            confidence_ratio = high_confidence_components / total_components
            
            if confidence_ratio >= 0.7:
                return 'high'
            elif confidence_ratio >= 0.5:
                return 'medium'
            else:
                return 'low'
