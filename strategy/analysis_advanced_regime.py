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

class AdvancedRegimeDetectionSystem:
    """
    Advanced market regime detection incorporating price action, structure, and momentum
    Provides nuanced regime classification for adaptive strategy selection
    """
    
    def __init__(self):
        self.regime_cache = {}
        self.regime_transition_tracker = {}
    
    def detect_advanced_market_regimes(self, df):
        """
        Comprehensive market regime detection with price action integration
        Returns detailed regime classification and transition analysis
        """
        if len(df) < 50:
            return self._create_default_regime_analysis()
        
        # Create a copy to avoid modifying original
        df_analysis = df.copy()
        
        regime_analysis = {
            'primary_regime': 'unknown',
            'regime_confidence': 0,
            'regime_strength': 0,
            'sub_regime': 'unknown',
            'transition_phase': False,
            'regime_duration_bars': 0,
            'key_characteristics': {},
            'regime_score_breakdown': {},
            'adaptive_parameters': {}
        }
        
        # 1. VOLATILITY REGIME ANALYSIS
        volatility_analysis = self._analyze_volatility_regime(df_analysis)
        regime_analysis.update(volatility_analysis)
        
        # 2. TREND REGIME ANALYSIS  
        trend_analysis = self._analyze_trend_regime(df_analysis)
        regime_analysis.update(trend_analysis)
        
        # 3. MARKET STRUCTURE ANALYSIS
        structure_analysis = self._analyze_market_structure_regime(df_analysis)
        regime_analysis.update(structure_analysis)
        
        # 4. MOMENTUM REGIME ANALYSIS
        momentum_analysis = self._analyze_momentum_regime(df_analysis)
        regime_analysis.update(momentum_analysis)
        
        # 5. VOLUME REGIME ANALYSIS
        volume_analysis = self._analyze_volume_regime(df_analysis)
        regime_analysis.update(volume_analysis)
        
        # 6. SYNTHESIZE PRIMARY REGIME
        primary_regime = self._synthesize_primary_regime(regime_analysis)
        regime_analysis['primary_regime'] = primary_regime
        
        # 7. CALCULATE CONFIDENCE AND STRENGTH
        confidence_metrics = self._calculate_regime_confidence(regime_analysis)
        regime_analysis.update(confidence_metrics)
        
        # 8. DETECT TRANSITION PHASES
        transition_analysis = self._detect_regime_transitions(df_analysis, regime_analysis)
        regime_analysis.update(transition_analysis)
        
        # 9. GENERATE ADAPTIVE PARAMETERS
        adaptive_params = self._generate_adaptive_parameters(regime_analysis)
        regime_analysis['adaptive_parameters'] = adaptive_params
        
        return regime_analysis
    
    def _analyze_volatility_regime(self, df):
        """Analyze volatility characteristics to determine regime"""
        volatility_analysis = {
            'volatility_regime': 'normal',
            'volatility_score': 0,
            'atr_regime': 'normal',
            'bb_regime': 'normal',
            'volatility_clustering': False
        }
        
        if len(df) < 20:
            return volatility_analysis
        
        # ATR-based volatility analysis
        if 'atr' in df.columns:
            atr_values = df['atr'].tail(50)
            current_atr = atr_values.iloc[-1]
            atr_median = atr_values.median()
            atr_std = atr_values.std()
            
            # Normalize ATR by price
            current_price = df['close'].iloc[-1]
            normalized_atr = current_atr / current_price
            
            if normalized_atr > atr_median / current_price + atr_std / current_price:
                volatility_analysis['atr_regime'] = 'high_volatility'
                volatility_analysis['volatility_score'] += 2
            elif normalized_atr < atr_median / current_price - atr_std / current_price:
                volatility_analysis['atr_regime'] = 'low_volatility'
                volatility_analysis['volatility_score'] += 0
            else:
                volatility_analysis['atr_regime'] = 'normal_volatility'
                volatility_analysis['volatility_score'] += 1
        
        # Bollinger Band Width analysis
        if 'bb_width' in df.columns:
            bb_width_values = df['bb_width'].tail(50)
            current_bb_width = bb_width_values.iloc[-1]
            bb_median = bb_width_values.median()
            bb_std = bb_width_values.std()
            
            if current_bb_width > bb_median + bb_std:
                volatility_analysis['bb_regime'] = 'high_volatility'
                volatility_analysis['volatility_score'] += 2
            elif current_bb_width < bb_median - bb_std:
                volatility_analysis['bb_regime'] = 'low_volatility'
                volatility_analysis['volatility_score'] += 0
            else:
                volatility_analysis['bb_regime'] = 'normal_volatility'
                volatility_analysis['volatility_score'] += 1
        
        # Check for volatility clustering (GARCH-like behavior)
        returns = df['close'].pct_change().dropna()
        if len(returns) > 20:
            # Simple volatility clustering detection
            recent_vol = returns.tail(10).std()
            previous_vol = returns.tail(20).head(10).std()
            
            if previous_vol > 0.0 and abs(recent_vol - previous_vol) / previous_vol > 0.5:
                volatility_analysis['volatility_clustering'] = True
                volatility_analysis['volatility_score'] += 1
        
        # Determine overall volatility regime
        if volatility_analysis['volatility_score'] >= 4:
            volatility_analysis['volatility_regime'] = 'high_volatility'
        elif volatility_analysis['volatility_score'] <= 1:
            volatility_analysis['volatility_regime'] = 'low_volatility'
        else:
            volatility_analysis['volatility_regime'] = 'normal_volatility'
        
        return volatility_analysis
    
    def _analyze_trend_regime(self, df):
        """Analyze trend characteristics and strength"""
        trend_analysis = {
            'trend_regime': 'ranging',
            'trend_strength': 0,
            'trend_direction': 'neutral',
            'adx_regime': 'no_trend',
            'ma_regime': 'mixed',
            'structure_trend': 'unclear'
        }
        
        if len(df) < 30:
            return trend_analysis
        
        # ADX-based trend analysis
        if 'adx' in df.columns:
            current_adx = df['adx'].iloc[-1]
            adx_values = df['adx'].tail(20)
            adx_trend = adx_values.mean()
            
            if current_adx > 25:
                trend_analysis['adx_regime'] = 'strong_trend'
                trend_analysis['trend_strength'] += 2
            elif current_adx > 20:
                trend_analysis['adx_regime'] = 'trending'
                trend_analysis['trend_strength'] += 1
            else:
                trend_analysis['adx_regime'] = 'no_trend'
                trend_analysis['trend_strength'] += 0
        
        # Moving Average trend analysis
        ma_columns = ['ema_20', 'ema_50', 'sma_50', 'sma_200']
        ma_bullish = 0
        ma_total = 0
        
        current_price = df['close'].iloc[-1]
        for ma_col in ma_columns:
            if ma_col in df.columns:
                ma_value = df[ma_col].iloc[-1]
                if current_price > ma_value:
                    ma_bullish += 1
                ma_total += 1
        
        if ma_total > 0:
            ma_alignment = ma_bullish / ma_total
            if ma_alignment >= 0.75:
                trend_analysis['ma_regime'] = 'bullish_alignment'
                trend_analysis['trend_strength'] += 2
                trend_analysis['trend_direction'] = 'bullish'
            elif ma_alignment <= 0.25:
                trend_analysis['ma_regime'] = 'bearish_alignment'
                trend_analysis['trend_strength'] += 2
                trend_analysis['trend_direction'] = 'bearish'
            else:
                trend_analysis['ma_regime'] = 'mixed'
                trend_analysis['trend_strength'] += 0
        
        # Price structure trend analysis
        if 'higher_highs_lower_lows' in df.columns:
            structure_value = df['higher_highs_lower_lows'].iloc[-1]
            if structure_value == 1:
                trend_analysis['structure_trend'] = 'uptrend'
                trend_analysis['trend_strength'] += 1
                trend_analysis['trend_direction'] = 'bullish'
            elif structure_value == -1:
                trend_analysis['structure_trend'] = 'downtrend'
                trend_analysis['trend_strength'] += 1
                trend_analysis['trend_direction'] = 'bearish'
        
        # Determine overall trend regime
        if trend_analysis['trend_strength'] >= 4:
            trend_analysis['trend_regime'] = 'strong_trend'
        elif trend_analysis['trend_strength'] >= 2:
            trend_analysis['trend_regime'] = 'trending'
        else:
            trend_analysis['trend_regime'] = 'ranging'
        
        return trend_analysis
    
    def _analyze_market_structure_regime(self, df):
        """Analyze market structure for regime classification"""
        structure_analysis = {
            'structure_regime': 'balanced',
            'swing_density': 'normal',
            'range_characteristics': 'normal',
            'breakout_potential': 'low',
            'support_resistance_clarity': 'medium'
        }
        
        if len(df) < 20:
            return structure_analysis
        
        # Swing point density analysis
        if 'swing_high' in df.columns and 'swing_low' in df.columns:
            recent_swing_highs = df['swing_high'].tail(20).sum()
            recent_swing_lows = df['swing_low'].tail(20).sum()
            total_swings = recent_swing_highs + recent_swing_lows
            
            if total_swings > 8:
                structure_analysis['swing_density'] = 'high'
                structure_analysis['structure_regime'] = 'choppy'
            elif total_swings < 3:
                structure_analysis['swing_density'] = 'low'
                structure_analysis['structure_regime'] = 'directional'
            else:
                structure_analysis['swing_density'] = 'normal'
        
        # Range analysis
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        range_pct = (recent_high - recent_low) / recent_low
        
        if range_pct < 0.02:  # Less than 2% range
            structure_analysis['range_characteristics'] = 'tight_range'
            structure_analysis['breakout_potential'] = 'high'
        elif range_pct > 0.08:  # More than 8% range
            structure_analysis['range_characteristics'] = 'wide_range'
            structure_analysis['breakout_potential'] = 'low'
        else:
            structure_analysis['range_characteristics'] = 'normal_range'
            structure_analysis['breakout_potential'] = 'medium'
        
        # Support/Resistance clarity
        if 'equal_highs_lows' in df.columns:
            equal_patterns = df['equal_highs_lows'].tail(10).abs().sum()
            if equal_patterns >= 2:
                structure_analysis['support_resistance_clarity'] = 'high'
            elif equal_patterns == 1:
                structure_analysis['support_resistance_clarity'] = 'medium'
            else:
                structure_analysis['support_resistance_clarity'] = 'low'
        
        return structure_analysis
    
    def _analyze_momentum_regime(self, df):
        """Analyze momentum characteristics for regime classification"""
        momentum_analysis = {
            'momentum_regime': 'neutral',
            'momentum_strength': 'moderate',
            'oscillator_regime': 'neutral',
            'divergence_presence': 'none',
            'momentum_consistency': 'mixed'
        }
        
        if len(df) < 20:
            return momentum_analysis
        
        # RSI regime analysis
        if 'rsi' in df.columns:
            current_rsi = df['rsi'].iloc[-1]
            rsi_values = df['rsi'].tail(14)
            rsi_std = rsi_values.std()
            
            if current_rsi > 70:
                momentum_analysis['oscillator_regime'] = 'overbought'
                momentum_analysis['momentum_strength'] = 'strong'
            elif current_rsi < 30:
                momentum_analysis['oscillator_regime'] = 'oversold'
                momentum_analysis['momentum_strength'] = 'strong'
            elif rsi_std < 10:
                momentum_analysis['oscillator_regime'] = 'compressed'
                momentum_analysis['momentum_strength'] = 'weak'
            else:
                momentum_analysis['oscillator_regime'] = 'neutral'
                momentum_analysis['momentum_strength'] = 'moderate'
        
        # MACD regime analysis
        if 'macd' in df.columns and 'macd_hist' in df.columns:
            macd_values = df['macd'].tail(10)
            macd_hist_values = df['macd_hist'].tail(10)
            
            # Check for consistent MACD direction
            macd_positive = (macd_values > 0).sum()
            macd_negative = (macd_values < 0).sum()
            
            if macd_positive >= 8:
                momentum_analysis['momentum_regime'] = 'bullish_momentum'
            elif macd_negative >= 8:
                momentum_analysis['momentum_regime'] = 'bearish_momentum'
            
            # Check for histogram divergence
            hist_positive = (macd_hist_values > 0).sum()
            hist_negative = (macd_hist_values < 0).sum()
            
            if hist_positive >= 6 and macd_negative >= 6:
                momentum_analysis['divergence_presence'] = 'bullish_divergence'
            elif hist_negative >= 6 and macd_positive >= 6:
                momentum_analysis['divergence_presence'] = 'bearish_divergence'
        
        # Momentum consistency analysis
        if 'momentum_continuation' in df.columns:
            continuation_values = df['momentum_continuation'].tail(10)
            consistent_bullish = (continuation_values == 1).sum()
            consistent_bearish = (continuation_values == -1).sum()
            
            if consistent_bullish >= 6:
                momentum_analysis['momentum_consistency'] = 'bullish_consistent'
            elif consistent_bearish >= 6:
                momentum_analysis['momentum_consistency'] = 'bearish_consistent'
            elif consistent_bullish + consistent_bearish >= 8:
                momentum_analysis['momentum_consistency'] = 'consistent'
            else:
                momentum_analysis['momentum_consistency'] = 'mixed'
        
        return momentum_analysis
    
    def _analyze_volume_regime(self, df):
        """Analyze volume characteristics for regime classification"""
        volume_analysis = {
            'volume_regime': 'normal',
            'volume_trend': 'stable',
            'volume_volatility': 'normal',
            'accumulation_distribution': 'neutral'
        }
        
        if 'volume' not in df.columns:
            return volume_analysis
        
        # Volume trend analysis
        volume_values = df['volume'].tail(20)
        volume_sma_10 = volume_values.rolling(10).mean().iloc[-1]
        volume_sma_20 = volume_values.rolling(20).mean().iloc[-1]
        
        if volume_sma_10 > volume_sma_20 * 1.1:
            volume_analysis['volume_trend'] = 'increasing'
            volume_analysis['volume_regime'] = 'high_volume'
        elif volume_sma_10 < volume_sma_20 * 0.9:
            volume_analysis['volume_trend'] = 'decreasing'
            volume_analysis['volume_regime'] = 'low_volume'
        else:
            volume_analysis['volume_trend'] = 'stable'
            volume_analysis['volume_regime'] = 'normal'
        
        # Volume volatility analysis
        volume_std = volume_values.std()
        volume_mean = volume_values.mean()
        volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
        
        if volume_cv > 0.5:
            volume_analysis['volume_volatility'] = 'high'
        elif volume_cv < 0.2:
            volume_analysis['volume_volatility'] = 'low'
        else:
            volume_analysis['volume_volatility'] = 'normal'
        
        # Accumulation/Distribution analysis
        if 'cmf' in df.columns:
            cmf_values = df['cmf'].tail(10)
            positive_cmf = (cmf_values > 0).sum()
            negative_cmf = (cmf_values < 0).sum()
            
            if positive_cmf >= 8:
                volume_analysis['accumulation_distribution'] = 'accumulation'
            elif negative_cmf >= 8:
                volume_analysis['accumulation_distribution'] = 'distribution'
        
        return volume_analysis
    
    def _synthesize_primary_regime(self, regime_analysis):
        """Synthesize all regime analyses into primary regime classification"""
        regime_scores = {
            'strong_trend_high_vol': 0,      # Trending with high volatility
            'strong_trend_normal_vol': 0,    # Trending with normal volatility
            'weak_trend': 0,                 # Weak trending
            'ranging_high_vol': 0,           # Choppy with high volatility  
            'ranging_normal_vol': 0,         # Normal ranging
            'ranging_low_vol': 0,            # Low volatility compression
            'transition_high_vol': 0,        # High volatility transition
            'transition_normal_vol': 0       # Normal volatility transition
        }
        
        # Trend strength scoring
        if regime_analysis['trend_regime'] == 'strong_trend':
            if regime_analysis['volatility_regime'] == 'high_volatility':
                regime_scores['strong_trend_high_vol'] += 3
            else:
                regime_scores['strong_trend_normal_vol'] += 3
        elif regime_analysis['trend_regime'] == 'trending':
            regime_scores['weak_trend'] += 2
        
        # Ranging market scoring
        if regime_analysis['trend_regime'] == 'ranging':
            if regime_analysis['volatility_regime'] == 'high_volatility':
                regime_scores['ranging_high_vol'] += 3
            elif regime_analysis['volatility_regime'] == 'low_volatility':
                regime_scores['ranging_low_vol'] += 3
            else:
                regime_scores['ranging_normal_vol'] += 2
        
        # Structure-based scoring
        if regime_analysis['structure_regime'] == 'choppy':
            regime_scores['ranging_high_vol'] += 1
        elif regime_analysis['structure_regime'] == 'directional':
            regime_scores['strong_trend_normal_vol'] += 1
        
        # Momentum-based scoring
        if regime_analysis['momentum_regime'] in ['bullish_momentum', 'bearish_momentum']:
            regime_scores['strong_trend_normal_vol'] += 1
        elif regime_analysis['oscillator_regime'] in ['overbought', 'oversold']:
            regime_scores['transition_high_vol'] += 1
        
        # Volume-based scoring
        if regime_analysis['volume_regime'] == 'high_volume':
            if regime_analysis['trend_regime'] == 'ranging':
                regime_scores['transition_high_vol'] += 2
            else:
                regime_scores['strong_trend_high_vol'] += 1
        
        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        
        return primary_regime
    
    def _calculate_regime_confidence(self, regime_analysis):
        """Calculate confidence metrics for regime classification"""
        confidence_metrics = {
            'regime_confidence': 0,
            'regime_strength': 0,
            'consistency_score': 0
        }
        
        # Base confidence on agreement between different analyses
        agreement_components = 0
        total_components = 5  # volatility, trend, structure, momentum, volume
        
        # Check volatility agreement
        if (regime_analysis['volatility_regime'] == 'high_volatility' and 
            regime_analysis['volume_regime'] == 'high_volume'):
            agreement_components += 1
        
        # Check trend agreement
        if (regime_analysis['trend_regime'] in ['strong_trend', 'trending'] and
            regime_analysis['momentum_regime'] in ['bullish_momentum', 'bearish_momentum']):
            agreement_components += 1
        
        # Check structure agreement
        if (regime_analysis['structure_regime'] == 'choppy' and
            regime_analysis['trend_regime'] == 'ranging'):
            agreement_components += 1
        
        # Calculate confidence scores
        confidence_metrics['regime_confidence'] = (agreement_components / total_components) * 100
        confidence_metrics['regime_strength'] = regime_analysis['trend_strength'] * 20  # Convert to 0-100 scale
        confidence_metrics['consistency_score'] = self._calculate_consistency_score(regime_analysis)
        
        return confidence_metrics
    
    def _calculate_consistency_score(self, regime_analysis):
        """Calculate consistency score across different timeframes"""
        # This would typically compare multiple timeframe analyses
        # For now, we'll use a simplified approach
        consistency_factors = []
        
        # Trend consistency
        if regime_analysis['trend_regime'] == regime_analysis['adx_regime']:
            consistency_factors.append(1)
        
        # Volatility consistency
        if regime_analysis['volatility_regime'] == regime_analysis['bb_regime']:
            consistency_factors.append(1)
        
        # Structure consistency
        if regime_analysis['structure_regime'] != 'balanced':
            consistency_factors.append(1)
        
        return (sum(consistency_factors) / 3) * 100 if consistency_factors else 50
    
    def _detect_regime_transitions(self, df, regime_analysis):
        """Detect potential regime transitions and phase changes"""
        transition_analysis = {
            'transition_phase': False,
            'transition_direction': 'none',
            'transition_confidence': 0,
            'potential_new_regime': 'unknown',
            'transition_trigger': 'none'
        }
        
        if len(df) < 30:
            return transition_analysis
        
        # Check for volatility expansion (potential transition trigger)
        if 'atr' in df.columns:
            current_atr = df['atr'].iloc[-1]
            previous_atr = df['atr'].iloc[-5]
            atr_change = (current_atr - previous_atr) / previous_atr
            
            if abs(atr_change) > 0.3:  # 30% change in ATR
                transition_analysis['transition_phase'] = True
                transition_analysis['transition_confidence'] += 50
                transition_analysis['transition_trigger'] = 'volatility_expansion'
        
        # Check for trend breakdown
        if 'higher_highs_lower_lows' in df.columns:
            structure_values = df['higher_highs_lower_lows'].tail(5)
            if len(structure_values.unique()) > 1:  # Changing structure
                transition_analysis['transition_phase'] = True
                transition_analysis['transition_confidence'] += 30
        
        # Check for momentum reversal
        if 'rsi' in df.columns:
            rsi_values = df['rsi'].tail(10)
            if (rsi_values.iloc[-1] > 70 and rsi_values.iloc[-5] < 30) or \
            (rsi_values.iloc[-1] < 30 and rsi_values.iloc[-5] > 70):
                transition_analysis['transition_phase'] = True
                transition_analysis['transition_confidence'] += 40
        
        # Determine transition direction
        if transition_analysis['transition_phase']:
            if regime_analysis['primary_regime'].startswith('ranging'):
                transition_analysis['transition_direction'] = 'ranging_to_trending'
                transition_analysis['potential_new_regime'] = 'strong_trend_normal_vol'
            else:
                transition_analysis['transition_direction'] = 'trending_to_ranging'
                transition_analysis['potential_new_regime'] = 'ranging_normal_vol'
        
        return transition_analysis
    
    def _generate_adaptive_parameters(self, regime_analysis):
        """Generate adaptive trading parameters based on regime"""
        adaptive_params = {
            'position_sizing': 'normal',
            'risk_multiplier': 1.0,
            'stop_loss_type': 'standard',
            'take_profit_ratio': 2.0,
            'timeframe_preference': 'medium',
            'signal_strength_threshold': 0.6,
            'confirmation_requirements': 'standard'
        }
        
        primary_regime = regime_analysis['primary_regime']
        
        if primary_regime == 'strong_trend_high_vol':
            adaptive_params.update({
                'position_sizing': 'reduced',
                'risk_multiplier': 0.7,
                'stop_loss_type': 'wide',
                'take_profit_ratio': 3.0,
                'signal_strength_threshold': 0.7
            })
        elif primary_regime == 'strong_trend_normal_vol':
            adaptive_params.update({
                'position_sizing': 'normal',
                'risk_multiplier': 1.0,
                'stop_loss_type': 'standard',
                'take_profit_ratio': 2.5,
                'signal_strength_threshold': 0.6
            })
        elif primary_regime == 'ranging_high_vol':
            adaptive_params.update({
                'position_sizing': 'reduced',
                'risk_multiplier': 0.5,
                'stop_loss_type': 'tight',
                'take_profit_ratio': 1.5,
                'confirmation_requirements': 'strict'
            })
        elif primary_regime == 'ranging_low_vol':
            adaptive_params.update({
                'position_sizing': 'normal',
                'risk_multiplier': 1.0,
                'stop_loss_type': 'standard',
                'take_profit_ratio': 1.0,  # Breakout trading
                'signal_strength_threshold': 0.8
            })
        elif primary_regime == 'transition_high_vol':
            adaptive_params.update({
                'position_sizing': 'minimal',
                'risk_multiplier': 0.3,
                'stop_loss_type': 'very_wide',
                'take_profit_ratio': 4.0,
                'signal_strength_threshold': 0.9
            })
        
        return adaptive_params
    
    def _create_default_regime_analysis(self):
        """Create default regime analysis when insufficient data"""
        return {
            'primary_regime': 'unknown',
            'regime_confidence': 0,
            'regime_strength': 0,
            'sub_regime': 'unknown',
            'transition_phase': False,
            'regime_duration_bars': 0,
            'key_characteristics': {},
            'regime_score_breakdown': {},
            'adaptive_parameters': {
                'position_sizing': 'normal',
                'risk_multiplier': 1.0,
                'stop_loss_type': 'standard',
                'take_profit_ratio': 2.0,
                'signal_strength_threshold': 0.7
            }
        }
    


