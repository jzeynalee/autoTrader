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


class ConfluenceScoringSystem:
    """
    Advanced confluence scoring system for MTF strategy evaluation
    Quantifies signal alignment strength across multiple dimensions
    """
    
    def __init__(self):
        self.confluence_cache = {}
        self.scoring_weights = {
            'timeframe_alignment': 0.25,
            'signal_strength': 0.20,
            'volume_confirmation': 0.15,
            'momentum_alignment': 0.15,
            'structure_quality': 0.15,
            'regime_alignment': 0.10
        }
    
    def calculate_mtf_confluence_score(self, htf_df, ttf_df, ltf_df, signals_config, direction):
        """
        Calculate comprehensive confluence score for MTF signal configuration
        
        Args:
            htf_df, ttf_df, ltf_df: Dataframes for each timeframe
            signals_config: Dictionary of signals per timeframe
            direction: 'bullish' or 'bearish'
            
        Returns:
            Dictionary with comprehensive confluence scoring
        """
        htf_key = f"{len(htf_df)}_{htf_df['timestamp'].iloc[-1]}"
        ttf_key = f"{len(ttf_df)}_{ttf_df['timestamp'].iloc[-1]}"
        ltf_key = f"{len(ltf_df)}_{ltf_df['timestamp'].iloc[-1]}"
        cache_key = f"confluence_{htf_key}_{ttf_key}_{ltf_key}_{direction}"
        if cache_key in self.confluence_cache:
            return self.confluence_cache[cache_key]
        
        confluence_analysis = {
            'overall_score': 0,
            'grade': 'F',
            'component_scores': {},
            'signal_breakdown': {},
            'strength_indicators': {},
            'recommendation': 'avoid',
            'confidence_level': 'low'
        }
        
        try:
            # 1. TIMEFRAME ALIGNMENT SCORE
            timeframe_score = self._calculate_timeframe_alignment_score(
                htf_df, ttf_df, ltf_df, signals_config, direction
            )
            confluence_analysis['component_scores']['timeframe_alignment'] = timeframe_score
            
            # 2. SIGNAL STRENGTH SCORE
            signal_strength_score = self._calculate_signal_strength_score(
                htf_df, ttf_df, ltf_df, signals_config, direction
            )
            confluence_analysis['component_scores']['signal_strength'] = signal_strength_score
            
            # 3. VOLUME CONFIRMATION SCORE
            volume_score = self._calculate_volume_confirmation_score(htf_df, ttf_df, ltf_df, direction)
            confluence_analysis['component_scores']['volume_confirmation'] = volume_score
            
            # 4. MOMENTUM ALIGNMENT SCORE
            momentum_score = self._calculate_momentum_alignment_score(htf_df, ttf_df, ltf_df, direction)
            confluence_analysis['component_scores']['momentum_alignment'] = momentum_score
            
            # 5. STRUCTURE QUALITY SCORE
            structure_score = self._calculate_structure_quality_score(htf_df, ttf_df, ltf_df)
            confluence_analysis['component_scores']['structure_quality'] = structure_score
            
            # 6. REGIME ALIGNMENT SCORE
            regime_score = self._calculate_regime_alignment_score(htf_df, ttf_df, ltf_df, direction)
            confluence_analysis['component_scores']['regime_alignment'] = regime_score
            
            # CALCULATE OVERALL WEIGHTED SCORE
            overall_score = self._calculate_weighted_score(confluence_analysis['component_scores'])
            confluence_analysis['overall_score'] = overall_score
            
            # ADD QUALITATIVE ASSESSMENTS
            confluence_analysis.update(self._generate_qualitative_assessments(overall_score, confluence_analysis['component_scores']))
            
            # SIGNAL BREAKDOWN ANALYSIS
            confluence_analysis['signal_breakdown'] = self._analyze_signal_breakdown(
                htf_df, ttf_df, ltf_df, signals_config, direction
            )
            
            # STRENGTH INDICATORS
            confluence_analysis['strength_indicators'] = self._calculate_strength_indicators(
                htf_df, ttf_df, ltf_df, direction
            )
            
        except Exception as e:
            print(f"Warning: Confluence scoring error: {str(e)}")
            # Return default low scores on error
            for component in self.scoring_weights.keys():
                confluence_analysis['component_scores'][component] = 0
            confluence_analysis['overall_score'] = 0
        
        # Cache the result
        self.confluence_cache[cache_key] = confluence_analysis
        return confluence_analysis
    
    def _calculate_timeframe_alignment_score(self, htf_df, ttf_df, ltf_df, signals_config, direction):
        """
        Calculate how well signals align across timeframes
        """
        alignment_scores = []
        total_signals = 0
        
        # Check signal alignment for each timeframe
        for tf, signals in signals_config.items():
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            tf_signals = 0
            tf_matching = 0
            
            for signal in signals:
                if signal in df.columns:
                    states = self._get_signal_states(df, signal)
                    if states is not None:
                        tf_signals += 1
                        if direction == 'bullish':
                            if states.iloc[-1] == 'bullish':
                                tf_matching += 1
                        else:
                            if states.iloc[-1] == 'bearish':
                                tf_matching += 1
            
            if tf_signals > 0:
                tf_score = tf_matching / tf_signals
                alignment_scores.append(tf_score)
                total_signals += tf_signals
        
        if not alignment_scores:
            return 0
        
        # Calculate alignment consistency
        avg_alignment = sum(alignment_scores) / len(alignment_scores)
        
        # Bonus for perfect alignment across all timeframes
        perfect_alignment_bonus = 1.0 if all(score >= 0.8 for score in alignment_scores) else 0.0
        
        # Penalty for low signal count
        signal_count_penalty = max(0, 1 - (10 / total_signals)) if total_signals < 10 else 0
        
        final_score = (avg_alignment * 0.8 + perfect_alignment_bonus * 0.2) * (1 - signal_count_penalty)
        return min(100, final_score * 100)
    
    def _calculate_signal_strength_score(self, htf_df, ttf_df, ltf_df, signals_config, direction):
        """
        Calculate the strength and conviction of signals
        """
        strength_metrics = {
            'signal_intensity': 0,
            'consistency': 0,
            'conviction': 0
        }
        
        all_signal_strengths = []
        signal_consistencies = []
        
        for tf, signals in signals_config.items():
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            tf_strengths = []
            
            for signal in signals:
                if signal in df.columns:
                    # Calculate signal strength based on various factors
                    strength = self._calculate_individual_signal_strength(df, signal, direction)
                    tf_strengths.append(strength)
                    
                    # Calculate consistency over recent periods
                    consistency = self._calculate_signal_consistency(df, signal, direction)
                    signal_consistencies.append(consistency)
            
            if tf_strengths:
                all_signal_strengths.extend(tf_strengths)
        
        if all_signal_strengths:
            strength_metrics['signal_intensity'] = sum(all_signal_strengths) / len(all_signal_strengths)
        
        if signal_consistencies:
            strength_metrics['consistency'] = sum(signal_consistencies) / len(signal_consistencies)
        
        # Calculate conviction (combination of intensity and consistency)
        strength_metrics['conviction'] = (
            strength_metrics['signal_intensity'] * 0.6 + 
            strength_metrics['consistency'] * 0.4
        )
        
        return min(100, strength_metrics['conviction'] * 100)
    
    def _calculate_volume_confirmation_score(self, htf_df, ttf_df, ltf_df, direction):
        """
        Calculate volume confirmation across timeframes
        """
        volume_scores = []
        
        for df in [htf_df, ttf_df, ltf_df]:
            if 'volume' not in df.columns:
                continue
                
            volume_score = 0
            current_volume = df['volume'].iloc[-1]
            volume_sma = df['volume'].rolling(20).mean().iloc[-1]
            
            # Volume above average is positive
            if current_volume > volume_sma * 1.2:
                volume_score += 0.4
            elif current_volume > volume_sma:
                volume_score += 0.2
            
            # Volume trend alignment
            if len(df) >= 10:
                recent_volume_avg = df['volume'].tail(5).mean()
                previous_volume_avg = df['volume'].tail(10).head(5).mean()
                
                if direction == 'bullish' and recent_volume_avg > previous_volume_avg:
                    volume_score += 0.3
                elif direction == 'bearish' and recent_volume_avg < previous_volume_avg:
                    volume_score += 0.3
            
            # Volume-based indicators
            if 'volume_breakout_confirmation' in df.columns:
                if df['volume_breakout_confirmation'].iloc[-1] > 0:
                    volume_score += 0.3
            
            volume_scores.append(min(1.0, volume_score))
        
        if not volume_scores:
            return 0
        
        return min(100, (sum(volume_scores) / len(volume_scores)) * 100)
    
    def _calculate_momentum_alignment_score(self, htf_df, ttf_df, ltf_df, direction):
        """
        Calculate momentum indicator alignment
        """
        momentum_indicators = ['rsi', 'macd', 'stoch_k', 'momentum', 'adx']
        alignment_scores = []
        
        for indicator in momentum_indicators:
            tf_scores = []
            
            for df in [htf_df, ttf_df, ltf_df]:
                if indicator in df.columns:
                    # Simple momentum direction assessment
                    value = df[indicator].iloc[-1]
                    
                    if indicator == 'rsi':
                        if direction == 'bullish' and 30 < value < 70:
                            tf_scores.append(0.7)
                        elif direction == 'bullish' and value > 50:
                            tf_scores.append(0.9)
                        elif direction == 'bearish' and 30 < value < 70:
                            tf_scores.append(0.7)
                        elif direction == 'bearish' and value < 50:
                            tf_scores.append(0.9)
                        else:
                            tf_scores.append(0.3)
                    
                    elif indicator == 'macd':
                        if 'macd_signal' in df.columns:
                            macd_signal = df['macd_signal'].iloc[-1]
                            if direction == 'bullish' and value > macd_signal:
                                tf_scores.append(0.8)
                            elif direction == 'bearish' and value < macd_signal:
                                tf_scores.append(0.8)
                            else:
                                tf_scores.append(0.3)
            
            if tf_scores:
                # Calculate alignment across timeframes for this indicator
                indicator_alignment = sum(tf_scores) / len(tf_scores)
                alignment_scores.append(indicator_alignment)
        
        if not alignment_scores:
            return 50  # Neutral score if no momentum indicators
        
        return min(100, (sum(alignment_scores) / len(alignment_scores)) * 100)
    
    def _calculate_structure_quality_score(self, htf_df, ttf_df, ltf_df):
        """
        Calculate price structure quality score
        """
        structure_scores = []
        
        for df in [htf_df, ttf_df, ltf_df]:
            score = 0
            
            # Swing point analysis
            if 'swing_high' in df.columns and 'swing_low' in df.columns:
                recent_swings = df[['swing_high', 'swing_low']].tail(20).sum().sum()
                if 3 <= recent_swings <= 8:  # Ideal swing density
                    score += 0.4
            
            # Trend structure
            if 'higher_highs_lower_lows' in df.columns:
                structure_value = df['higher_highs_lower_lows'].iloc[-1]
                if abs(structure_value) == 1:  # Clear trend structure
                    score += 0.3
            
            # Support/Resistance clarity
            if 'equal_highs_lows' in df.columns:
                recent_patterns = df['equal_highs_lows'].tail(10).abs().sum()
                if recent_patterns >= 2:  # Clear levels
                    score += 0.3
            
            structure_scores.append(min(1.0, score))
        
        if not structure_scores:
            return 0
        
        return min(100, (sum(structure_scores) / len(structure_scores)) * 100)
    
    def _calculate_regime_alignment_score(self, htf_df, ttf_df, ltf_df, direction):
        """
        Calculate alignment with current market regime
        """
        try:
            # Simple regime detection
            regime_scores = []
            
            for df in [htf_df, ttf_df, ltf_df]:
                score = 0
                
                # Volatility regime alignment
                if 'atr' in df.columns:
                    current_atr = df['atr'].iloc[-1]
                    atr_sma = df['atr'].rolling(20).mean().iloc[-1]
                    
                    if current_atr > atr_sma * 1.3:
                        # High volatility - certain strategies work better
                        score += 0.3
                    elif current_atr < atr_sma * 0.7:
                        # Low volatility - different strategies
                        score += 0.3
                    else:
                        # Normal volatility
                        score += 0.5
                
                # Trend regime
                if 'adx' in df.columns:
                    current_adx = df['adx'].iloc[-1]
                    if current_adx > 25:  # Strong trend
                        score += 0.4
                    elif current_adx > 20:  # Moderate trend
                        score += 0.6
                    else:  # Ranging
                        score += 0.3
                
                regime_scores.append(min(1.0, score))
            
            if not regime_scores:
                return 50
            
            return min(100, (sum(regime_scores) / len(regime_scores)) * 100)
            
        except Exception:
            return 50  # Neutral on error
    
    def _calculate_individual_signal_strength(self, df, signal, direction):
        """
        Calculate strength of an individual signal
        """
        try:
            strength = 0.5  # Base strength
            
            if signal in df.columns:
                current_value = df[signal].iloc[-1]
                
                # Signal-specific strength calculations
                if 'rsi' in signal:
                    if direction == 'bullish' and current_value < 40:
                        strength = 0.8 - (current_value / 100)
                    elif direction == 'bearish' and current_value > 60:
                        strength = (current_value / 100) - 0.4
                
                elif 'macd' in signal:
                    if 'macd_signal' in df.columns:
                        signal_line = df['macd_signal'].iloc[-1]
                        divergence = abs(current_value - signal_line)
                        strength = min(0.9, divergence * 10)  # Scale divergence
                
                elif 'bb' in signal:
                    # Bollinger Band position strength
                    if abs(current_value) > 0.8:  # Near bands
                        strength = 0.8
                    elif abs(current_value) > 0.5:
                        strength = 0.6
                
                elif any(pattern in signal for pattern in ['swing', 'breakout', 'divergence']):
                    # Pattern-based signals
                    if current_value != 0:
                        strength = 0.7 + (abs(current_value) * 0.3)
            
            return strength
            
        except Exception:
            return 0.3  # Weak strength on error
    
    def _calculate_signal_consistency(self, df, signal, direction, lookback=10):
        """
        Calculate consistency of signal over recent periods
        """
        try:
            if signal not in df.columns or len(df) < lookback:
                return 0.3
            
            recent_data = df[signal].tail(lookback)
            consistent_signals = 0
            
            for value in recent_data:
                if direction == 'bullish' and value > 0:
                    consistent_signals += 1
                elif direction == 'bearish' and value < 0:
                    consistent_signals += 1
            
            return consistent_signals / lookback
            
        except Exception:
            return 0.3
    
    def _calculate_weighted_score(self, component_scores):
        """
        Calculate weighted overall score from components
        """
        total_score = 0
        total_weight = 0
        
        for component, weight in self.scoring_weights.items():
            if component in component_scores:
                total_score += component_scores[component] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return min(100, total_score / total_weight)
    
    def _generate_qualitative_assessments(self, overall_score, component_scores):
        """
        Generate qualitative assessments based on scores
        """
        # Grade assignment
        if overall_score >= 90:
            grade = 'A+'
            recommendation = 'high_confidence'
            confidence = 'very_high'
        elif overall_score >= 85:
            grade = 'A'
            recommendation = 'high_confidence'
            confidence = 'high'
        elif overall_score >= 80:
            grade = 'A-'
            recommendation = 'good_confidence'
            confidence = 'high'
        elif overall_score >= 75:
            grade = 'B+'
            recommendation = 'good_confidence'
            confidence = 'medium_high'
        elif overall_score >= 70:
            grade = 'B'
            recommendation = 'moderate_confidence'
            confidence = 'medium'
        elif overall_score >= 65:
            grade = 'B-'
            recommendation = 'moderate_confidence'
            confidence = 'medium'
        elif overall_score >= 60:
            grade = 'C+'
            recommendation = 'low_confidence'
            confidence = 'medium_low'
        elif overall_score >= 55:
            grade = 'C'
            recommendation = 'low_confidence'
            confidence = 'low'
        elif overall_score >= 50:
            grade = 'C-'
            recommendation = 'very_low_confidence'
            confidence = 'low'
        elif overall_score >= 40:
            grade = 'D'
            recommendation = 'avoid'
            confidence = 'very_low'
        else:
            grade = 'F'
            recommendation = 'avoid'
            confidence = 'very_low'
        
        # Strength analysis
        strengths = []
        weaknesses = []
        
        for component, score in component_scores.items():
            if score >= 80:
                strengths.append(component)
            elif score <= 40:
                weaknesses.append(component)
        
        return {
            'grade': grade,
            'recommendation': recommendation,
            'confidence_level': confidence,
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def _analyze_signal_breakdown(self, htf_df, ttf_df, ltf_df, signals_config, direction):
        """
        Detailed analysis of signal performance breakdown
        """
        breakdown = {
            'timeframe_performance': {},
            'signal_categories': {},
            'alignment_metrics': {}
        }
        
        # Timeframe performance
        for tf, signals in signals_config.items():
            df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
            tf_signals = len([s for s in signals if s in df.columns])
            tf_active = sum(1 for s in signals if s in df.columns and self._is_signal_active(df, s, direction))
            
            breakdown['timeframe_performance'][tf] = {
                'total_signals': tf_signals,
                'active_signals': tf_active,
                'activation_rate': tf_active / tf_signals if tf_signals > 0 else 0
            }
        
        # Signal categories
        categories = {
            'momentum': ['rsi', 'macd', 'stoch', 'momentum'],
            'trend': ['ma', 'ema', 'adx', 'trend'],
            'volatility': ['bb', 'atr', 'volatility'],
            'volume': ['volume', 'obv', 'cmf'],
            'pattern': ['swing', 'breakout', 'divergence', 'pattern']
        }
        
        for category, indicators in categories.items():
            category_signals = []
            for signals in signals_config.values():
                category_signals.extend([s for s in signals if any(ind in s for ind in indicators)])
            
            breakdown['signal_categories'][category] = {
                'signal_count': len(set(category_signals)),
                'coverage': len(set(category_signals)) / len(indicators) if indicators else 0
            }
        
        return breakdown
    
    def _calculate_strength_indicators(self, htf_df, ttf_df, ltf_df, direction):
        """
        Calculate additional strength indicators
        """
        indicators = {
            'multi_timeframe_confirmation': 0,
            'signal_density': 0,
            'risk_adjusted_score': 0
        }
        
        # Multi-timeframe confirmation
        tf_directions = []
        for df in [htf_df, ttf_df, ltf_df]:
            tf_trend = self._assess_timeframe_direction(df, direction)
            tf_directions.append(tf_trend)
        
        indicators['multi_timeframe_confirmation'] = sum(tf_directions) / len(tf_directions)
        
        # Signal density (signals per timeframe)
        total_signals = sum(len(df.columns) for df in [htf_df, ttf_df, ltf_df])
        indicators['signal_density'] = min(1.0, total_signals / 300)  # Normalize
        
        # Risk-adjusted score (simplified)
        volatility = self._calculate_composite_volatility(htf_df, ttf_df, ltf_df)
        indicators['risk_adjusted_score'] = max(0, 1 - volatility)  # Lower volatility = better
        
        return indicators
    
    def _is_signal_active(self, df, signal, direction):
        """Check if a signal is currently active"""
        try:
            if signal not in df.columns:
                return False
            
            value = df[signal].iloc[-1]
            if direction == 'bullish':
                return value > 0
            else:
                return value < 0
        except:
            return False
    
    def _assess_timeframe_direction(self, df, desired_direction):
        """Assess if timeframe aligns with desired direction"""
        try:
            # Simple trend assessment
            if len(df) < 10:
                return 0.5
            
            price_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            
            if desired_direction == 'bullish' and price_trend > 0:
                return 0.8
            elif desired_direction == 'bearish' and price_trend < 0:
                return 0.8
            else:
                return 0.3
        except:
            return 0.5
    
    def _calculate_composite_volatility(self, htf_df, ttf_df, ltf_df):
        """Calculate composite volatility across timeframes"""
        volatilities = []
        
        for df in [htf_df, ttf_df, ltf_df]:
            if 'atr' in df.columns and len(df) > 0:
                current_atr = df['atr'].iloc[-1]
                current_price = df['close'].iloc[-1]
                normalized_atr = current_atr / current_price
                volatilities.append(normalized_atr)
        
        if not volatilities:
            return 0.02  # Default low volatility
        
        return sum(volatilities) / len(volatilities)
    
    def _get_signal_states(self, df, signal):
        """Get signal states with error handling"""
        try:
            if hasattr(self, 'get_or_compute_states'):
                return self.get_or_compute_states(df, signal)
            else:
                # Fallback basic implementation
                if signal in df.columns:
                    return df[signal]
                return None
        except Exception:
            return None
