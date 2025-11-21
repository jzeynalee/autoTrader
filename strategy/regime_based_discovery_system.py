"""
regime_based_discovery_system.py
"Strategy Playbook" Generator

This module builds on the AdvancedRegimeDetectionSystem.
Its purpose is to analyze all swing points *within* a detected regime
and find all technical indicators and price action patterns that
*confirm* that regime's status.

This creates a "Repository of Strategies" (or a "playbook") that maps
each regime (e.g., "Low-Vol Bull Trend") to the set of indicators
and patterns that are most frequently associated with it in the data.

This repository can then be used by a signal_finder.py to match
real-time conditions against the playbook for high-probability trades.

Architecture:
    AdvancedRegimeDetectionSystem (pre-run)
        -> HybridSwingRegistry (with HMM labels)
        -> HMM RegimeState definitions
    
    RegimeStrategyDiscovery (this file)
        -> Iterates over all labeled swings
        -> StrategyIndicatorRules (helper class)
            -> Checks swing features (RSI, MACD, Patterns, Structure)
            -> Finds all features that *confirm* the swing's regime
        -> Builds Strategy Repository (dict)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import json

try:
    from .analysis_advanced_regime import AdvancedRegimeDetectionSystem, RegimeState
    from .regime_statistical_analysis_sqlite import RegimeStatisticalAnalyzer
except ImportError:
    print("Error: Could not import 'AdvancedRegimeDetectionSystem'.")
    print("Please ensure 'analysis_advanced_regime.py' is in the same Python path.")
    class AdvancedRegimeDetectionSystem: pass
    class RegimeState: pass

# ============================================================================
# SECTION 1: INDICATOR & PATTERN LOGIC
# ============================================================================


class StrategyIndicatorRules:
    """
    COMPREHENSIVE indicator confirmation system.
    Systematically checks all 100+ technical indicators for regime confirmation.
    """
    
    def __init__(self, rsi_bull: float = 55, rsi_bear: float = 45, adx_trend: float = 20):
        """
        Initialize with configurable thresholds.
        
        Args:
            rsi_bull: RSI threshold for bullish (default 55)
            rsi_bear: RSI threshold for bearish (default 45)
            adx_trend: ADX threshold for trending market (default 20)
        """
        self.rsi_bull = rsi_bull
        self.rsi_bear = rsi_bear
        self.adx_trend = adx_trend
        
        # =====================================================================
        # INDICATOR CATEGORIES (for systematic checking)
        # =====================================================================
        
        # Momentum Indicators
        self.momentum_indicators = [
            'rsi', 'rsi_30', 'stoch_k', 'stoch_d', 'stoch_rsi', 'stoch_rsi_k', 'stoch_rsi_d',
            'williams_r', 'roc', 'roc_20', 'momentum', 'momentum_pct', 'connors_rsi',
            'ultimate_osc', 'tsi', 'smi', 'smi_signal', 'smi_ergodic', 'smi_ergodic_signal',
            'cci', 'woodies_cci', 'woodies_cci_signal'
        ]
        
        # Trend Indicators
        self.trend_indicators = [
            'macd', 'macd_hist', 'macd_signal', 'ppo', 'ppo_hist', 'ppo_signal',
            'adx', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down', 'aroon_osc',
            'trix', 'trix_signal', 'dpo', 'kst', 'kst_signal', 'mass_index',
            'ao', 'ichimoku_conversion', 'ichimoku_base', 'ichimoku_a', 'ichimoku_b',
            'ichimoku_lagging', 'psar', 'vortex_pos', 'vortex_neg', 'vortex_diff'
        ]
        
        # Volume Indicators
        self.volume_indicators = [
            'obv', 'obv_change', 'cmf', 'force_index', 'force_index_ema', 'mfi',
            'ad', 'eom', 'eom_sma', 'vpt', 'nvi', 'chaikin_osc',
            'volume_zscore', 'volume_roc', 'volume_ma5', 'volume_decreasing',
            'volume_spike', 'bop'
        ]
        
        # Volatility Indicators
        self.volatility_indicators = [
            'atr', 'atr_percent', 'atr_ratio', 'bb_width', 'bb_pct', 'bb_bandwidth',
            'kc_width', 'kc_pct', 'dc_width', 'dc_pct', 'ulcer_index',
            'normalized_variance', 'choppiness', 'bull_power', 'bear_power'
        ]
        
        # Moving Averages
        self.ma_indicators = [
            'sma_7', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_9', 'ema_12', 'ema_26', 'ema_50', 'ema_200',
            'wma_20', 'dema', 'tema', 'hma', 'kama', 'vwma', 'smma', 'lsma',
            'mcginley', 'vwap', 'bb_middle', 'bb_upper', 'bb_lower',
            'kc_middle', 'kc_upper', 'kc_lower', 'dc_middle', 'dc_upper', 'dc_lower'
        ]
        
        # Price Action Indicators
        self.price_action_indicators = [
            'local_slope', 'return_gradient', 'directional_persistence',
            'trend_short', 'trend_medium', 'trend_long',
            'structure_type', 'structure_break', 'pullback_depth',
            'pullback_from_high_pct', 'pullback_from_low_pct',
            'swing_high', 'swing_low', 'last_swing_high', 'last_swing_low',
            'bars_since_swing_high', 'bars_since_swing_low'
        ]
        
        # Fibonacci & Support/Resistance
        self.fibonacci_indicators = [
            'fib_23_6', 'fib_38_2', 'fib_50_0', 'fib_61_8', 'fib_78_6',
            'near_fib_236', 'near_fib_382', 'near_fib_500', 'near_fib_618', 'near_fib_786',
            'sr_confluence_score', 'pullback_quality_score'
        ]
        
        # Pullback Indicators
        self.pullback_indicators = [
            'pullback_to_ema9', 'pullback_to_ema20', 'pullback_to_sma50',
            'healthy_bull_pullback', 'healthy_bear_pullback',
            'pullback_complete_bull', 'pullback_complete_bear',
            'failed_pullback_bull', 'failed_pullback_bear',
            'higher_lows_pattern', 'lower_highs_pattern',
            'abc_pullback_bull', 'abc_pullback_bear',
            'measured_move_bull_target', 'measured_move_bear_target',
            'distance_to_bull_target_pct', 'distance_to_bear_target_pct',
            'pullback_stage'
        ]
        
        # Candlestick Patterns
        self.candlestick_patterns = {
            'bullish': [
                'pattern_hammer', 'pattern_inverted_hammer', 'pattern_bullish_engulfing',
                'pattern_morning_star', 'pattern_piercing_line', 'pattern_three_white_soldiers',
                'pattern_bullish_harami', 'pattern_morning_doji_star', 'pattern_three_inside_up',
                'pattern_three_outside_up', 'pattern_belt_hold_bull', 'pattern_tweezer_bottom',
                'pattern_abandoned_baby_bull', 'pattern_counterattack_bull', 'pattern_breakaway_bull',
                'pattern_kicking_bull', 'pattern_kicking_by_length_bull', 'pattern_marubozu_bull'
            ],
            'bearish': [
                'pattern_shooting_star', 'pattern_hanging_man', 'pattern_bearish_engulfing',
                'pattern_evening_star', 'pattern_dark_cloud_cover', 'pattern_three_black_crows',
                'pattern_bearish_harami', 'pattern_evening_doji_star', 'pattern_three_inside_down',
                'pattern_three_outside_down', 'pattern_belt_hold_bear', 'pattern_tweezer_top',
                'pattern_abandoned_baby_bear', 'pattern_counterattack_bear', 'pattern_breakaway_bear',
                'pattern_kicking_bear', 'pattern_kicking_by_length_bear', 'pattern_marubozu_bear'
            ],
            'neutral': [
                'pattern_doji', 'pattern_long_legged_doji', 'pattern_dragonfly_doji',
                'pattern_gravestone_doji', 'pattern_rickshaw_man'
            ]
        }
        
        # Chart Patterns
        self.chart_patterns = {
            'bullish': [
                'double_bottom', 'triple_bottom', 'ascending_triangle',
                'cup_and_handle', 'inverse_head_and_shoulders', 'bullish_flag',
                'bullish_pennant', 'rounding_bottom', 'falling_wedge'
            ],
            'bearish': [
                'double_top', 'triple_top', 'descending_triangle',
                'head_and_shoulders', 'bearish_flag', 'bearish_pennant',
                'rounding_top', 'rising_wedge'
            ],
            'neutral': [
                'symmetrical_triangle', 'rectangle', 'sideways_channel'
            ]
        }
        
        # Anomaly Indicators
        self.anomaly_indicators = [
            'gap_intensity', 'extended_bar_ratio', 'volume_spike',
            'recent_bullish_patterns', 'recent_bearish_patterns', 'recent_doji'
        ]

    # =========================================================================
    # MAIN CONFIRMATION FUNCTIONS
    # =========================================================================
    
    def get_bullish_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Comprehensive bullish confirmation check across ALL indicators.
        
        Args:
            row: Single row from swing DataFrame with all indicators
            
        Returns:
            Tuple of (indicators_list, patterns_list)
        """
        indicators: List[str] = []
        patterns: List[str] = []
        
        # === MOMENTUM CONFIRMATIONS ===
        indicators.extend(self._check_momentum_bullish(row))
        
        # === TREND CONFIRMATIONS ===
        indicators.extend(self._check_trend_bullish(row))
        
        # === VOLUME CONFIRMATIONS ===
        indicators.extend(self._check_volume_bullish(row))
        
        # === VOLATILITY STATE ===
        indicators.extend(self._check_volatility_bullish(row))
        
        # === MOVING AVERAGE ALIGNMENT ===
        indicators.extend(self._check_ma_alignment_bullish(row))
        
        # === FIBONACCI CONFIRMATIONS ===
        indicators.extend(self._check_fibonacci_bullish(row))
        
        # === PRICE ACTION PATTERNS ===
        patterns.extend(self._check_price_action_bullish(row))
        
        # === PULLBACK PATTERNS ===
        patterns.extend(self._check_pullback_bullish(row))
        
        # === CANDLESTICK PATTERNS ===
        patterns.extend(self._check_candlestick_patterns(row, 'bullish'))
        
        # === CHART PATTERNS ===
        patterns.extend(self._check_chart_patterns(row, 'bullish'))
        
        # === ANOMALY DETECTION ===
        indicators.extend(self._check_anomalies_bullish(row))
        
        return indicators, patterns
    
    def get_bearish_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """Comprehensive bearish confirmation check."""
        indicators: List[str] = []
        patterns: List[str] = []
        
        # === MOMENTUM CONFIRMATIONS ===
        indicators.extend(self._check_momentum_bearish(row))
        
        # === TREND CONFIRMATIONS ===
        indicators.extend(self._check_trend_bearish(row))
        
        # === VOLUME CONFIRMATIONS ===
        indicators.extend(self._check_volume_bearish(row))
        
        # === VOLATILITY STATE ===
        indicators.extend(self._check_volatility_bearish(row))
        
        # === MOVING AVERAGE ALIGNMENT ===
        indicators.extend(self._check_ma_alignment_bearish(row))
        
        # === FIBONACCI CONFIRMATIONS ===
        indicators.extend(self._check_fibonacci_bearish(row))
        
        # === PRICE ACTION PATTERNS ===
        patterns.extend(self._check_price_action_bearish(row))
        
        # === PULLBACK PATTERNS ===
        patterns.extend(self._check_pullback_bearish(row))
        
        # === CANDLESTICK PATTERNS ===
        patterns.extend(self._check_candlestick_patterns(row, 'bearish'))
        
        # === CHART PATTERNS ===
        patterns.extend(self._check_chart_patterns(row, 'bearish'))
        
        # === ANOMALY DETECTION ===
        indicators.extend(self._check_anomalies_bearish(row))
        
        return indicators, patterns
    
    def get_ranging_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """Comprehensive ranging/neutral confirmation check."""
        indicators: List[str] = []
        patterns: List[str] = []
        
        # RSI in neutral zone
        if 'rsi' in row and self.rsi_bear <= row['rsi'] <= self.rsi_bull:
            indicators.append(f"RSI Neutral ({self.rsi_bear}-{self.rsi_bull}): {row['rsi']:.1f}")
        
        # ADX showing no trend
        if 'adx' in row and row['adx'] < self.adx_trend:
            indicators.append(f"ADX < {self.adx_trend} (Ranging): {row['adx']:.1f}")
        
        # Bollinger Bands narrow (squeeze)
        if 'bb_width' in row and row['bb_width'] < 0.02:
            indicators.append(f"BB Squeeze (Width={row['bb_width']:.4f})")
        
        # Choppiness Index high
        if 'choppiness' in row and row['choppiness'] > 60:
            indicators.append(f"Choppy Market (Chop={row['choppiness']:.1f})")
        
        # Volume low
        if 'volume_zscore' in row and row['volume_zscore'] < -0.5:
            indicators.append(f"Low Volume (Z={row['volume_zscore']:.2f})")
        
        # Neutral candlestick patterns
        patterns.extend(self._check_candlestick_patterns(row, 'neutral'))
        
        # Range-bound chart patterns
        patterns.extend(self._check_chart_patterns(row, 'neutral'))
        
        # Price near middle of channels
        if 'bb_pct' in row and 0.4 <= row['bb_pct'] <= 0.6:
            patterns.append("Price Mid-Bollinger Band")
        
        if 'kc_pct' in row and 0.4 <= row['kc_pct'] <= 0.6:
            patterns.append("Price Mid-Keltner Channel")
        
        return indicators, patterns

    # =========================================================================
    # MOMENTUM INDICATOR CHECKS
    # =========================================================================
    
    def _check_momentum_bullish(self, row: pd.Series) -> List[str]:
        """Check all momentum indicators for bullish signals."""
        signals = []
        
        # === RSI Family ===
        if 'rsi' in row and row['rsi'] > self.rsi_bull:
            signals.append(f"RSI > {self.rsi_bull} ({row['rsi']:.1f})")
        
        if 'rsi_30' in row and row['rsi_30'] > 55:
            signals.append(f"RSI(30) Bullish ({row['rsi_30']:.1f})")
        
        if 'connors_rsi' in row and row['connors_rsi'] > 60:
            signals.append(f"Connors RSI Strong ({row['connors_rsi']:.1f})")
        
        # === Stochastic Oscillators ===
        if 'stoch_k' in row and 'stoch_d' in row:
            if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 80:
                signals.append(f"Stochastic Bullish (K={row['stoch_k']:.1f}, D={row['stoch_d']:.1f})")
            elif row['stoch_k'] > 80:
                signals.append(f"Stochastic Overbought (K={row['stoch_k']:.1f})")
        
        if 'stoch_rsi_k' in row and row['stoch_rsi_k'] > 50:
            signals.append(f"Stoch RSI Bullish ({row['stoch_rsi_k']:.1f})")
        
        # === Williams %R ===
        if 'williams_r' in row and row['williams_r'] > -50:
            signals.append(f"Williams %R Bullish ({row['williams_r']:.1f})")
        
        # === Rate of Change ===
        if 'roc' in row and row['roc'] > 2:
            signals.append(f"ROC Positive ({row['roc']:.2f}%)")
        
        if 'roc_20' in row and row['roc_20'] > 5:
            signals.append(f"ROC(20) Strong ({row['roc_20']:.2f}%)")
        
        # === Momentum ===
        if 'momentum' in row and row['momentum'] > 0:
            signals.append(f"Momentum Positive ({row['momentum']:.4f})")
        
        if 'momentum_pct' in row and row['momentum_pct'] > 2:
            signals.append(f"Momentum % Strong ({row['momentum_pct']:.2f}%)")
        
        # === Ultimate Oscillator ===
        if 'ultimate_osc' in row and row['ultimate_osc'] > 50:
            signals.append(f"Ultimate Osc Bullish ({row['ultimate_osc']:.1f})")
        
        # === TSI (True Strength Index) ===
        if 'tsi' in row and row['tsi'] > 0:
            signals.append(f"TSI Positive ({row['tsi']:.2f})")
        
        # === SMI (Stochastic Momentum Index) ===
        if 'smi' in row and 'smi_signal' in row:
            if row['smi'] > row['smi_signal'] and row['smi'] > 0:
                signals.append(f"SMI Bullish Cross ({row['smi']:.2f})")
        
        if 'smi_ergodic' in row and row['smi_ergodic'] > 0:
            signals.append(f"SMI Ergodic Positive ({row['smi_ergodic']:.2f})")
        
        # === CCI (Commodity Channel Index) ===
        if 'cci' in row and row['cci'] > 100:
            signals.append(f"CCI Overbought ({row['cci']:.1f})")
        elif 'cci' in row and 0 < row['cci'] < 100:
            signals.append(f"CCI Bullish ({row['cci']:.1f})")
        
        if 'woodies_cci' in row and 'woodies_cci_signal' in row:
            if row['woodies_cci'] > row['woodies_cci_signal']:
                signals.append("Woodies CCI Bullish")
        
        return signals
    
    def _check_momentum_bearish(self, row: pd.Series) -> List[str]:
        """Check all momentum indicators for bearish signals."""
        signals = []
        
        # === RSI Family ===
        if 'rsi' in row and row['rsi'] < self.rsi_bear:
            signals.append(f"RSI < {self.rsi_bear} ({row['rsi']:.1f})")
        
        if 'rsi_30' in row and row['rsi_30'] < 45:
            signals.append(f"RSI(30) Bearish ({row['rsi_30']:.1f})")
        
        if 'connors_rsi' in row and row['connors_rsi'] < 40:
            signals.append(f"Connors RSI Weak ({row['connors_rsi']:.1f})")
        
        # === Stochastic Oscillators ===
        if 'stoch_k' in row and 'stoch_d' in row:
            if row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 20:
                signals.append(f"Stochastic Bearish (K={row['stoch_k']:.1f}, D={row['stoch_d']:.1f})")
            elif row['stoch_k'] < 20:
                signals.append(f"Stochastic Oversold (K={row['stoch_k']:.1f})")
        
        if 'stoch_rsi_k' in row and row['stoch_rsi_k'] < 50:
            signals.append(f"Stoch RSI Bearish ({row['stoch_rsi_k']:.1f})")
        
        # === Williams %R ===
        if 'williams_r' in row and row['williams_r'] < -50:
            signals.append(f"Williams %R Bearish ({row['williams_r']:.1f})")
        
        # === Rate of Change ===
        if 'roc' in row and row['roc'] < -2:
            signals.append(f"ROC Negative ({row['roc']:.2f}%)")
        
        if 'roc_20' in row and row['roc_20'] < -5:
            signals.append(f"ROC(20) Weak ({row['roc_20']:.2f}%)")
        
        # === Momentum ===
        if 'momentum' in row and row['momentum'] < 0:
            signals.append(f"Momentum Negative ({row['momentum']:.4f})")
        
        if 'momentum_pct' in row and row['momentum_pct'] < -2:
            signals.append(f"Momentum % Weak ({row['momentum_pct']:.2f}%)")
        
        # === Ultimate Oscillator ===
        if 'ultimate_osc' in row and row['ultimate_osc'] < 50:
            signals.append(f"Ultimate Osc Bearish ({row['ultimate_osc']:.1f})")
        
        # === TSI ===
        if 'tsi' in row and row['tsi'] < 0:
            signals.append(f"TSI Negative ({row['tsi']:.2f})")
        
        # === SMI ===
        if 'smi' in row and 'smi_signal' in row:
            if row['smi'] < row['smi_signal'] and row['smi'] < 0:
                signals.append(f"SMI Bearish Cross ({row['smi']:.2f})")
        
        if 'smi_ergodic' in row and row['smi_ergodic'] < 0:
            signals.append(f"SMI Ergodic Negative ({row['smi_ergodic']:.2f})")
        
        # === CCI ===
        if 'cci' in row and row['cci'] < -100:
            signals.append(f"CCI Oversold ({row['cci']:.1f})")
        elif 'cci' in row and -100 < row['cci'] < 0:
            signals.append(f"CCI Bearish ({row['cci']:.1f})")
        
        return signals

    # =========================================================================
    # TREND INDICATOR CHECKS
    # =========================================================================
    
    def _check_trend_bullish(self, row: pd.Series) -> List[str]:
        """Check all trend indicators for bullish signals."""
        signals = []
        
        # === MACD Family ===
        if 'macd_hist' in row and row['macd_hist'] > 0:
            signals.append(f"MACD Histogram Positive ({row['macd_hist']:.4f})")
        
        if 'macd' in row and 'macd_signal' in row:
            if row['macd'] > row['macd_signal']:
                signals.append("MACD Above Signal")
                if row['macd'] > 0:
                    signals.append("MACD in Positive Territory")
        
        # === PPO (Percentage Price Oscillator) ===
        if 'ppo' in row and row['ppo'] > 0:
            signals.append(f"PPO Positive ({row['ppo']:.2f})")
        
        if 'ppo_hist' in row and row['ppo_hist'] > 0:
            signals.append(f"PPO Histogram Positive ({row['ppo_hist']:.2f})")
        
        # === ADX & Directional Movement ===
        if 'adx' in row and row['adx'] > self.adx_trend:
            if 'di_plus' in row and 'di_minus' in row:
                if row['di_plus'] > row['di_minus']:
                    strength = "Strong" if row['adx'] > 30 else "Moderate"
                    signals.append(f"ADX {strength} Bullish Trend ({row['adx']:.1f})")
                    signals.append(f"+DI > -DI ({row['di_plus']:.1f} vs {row['di_minus']:.1f})")
        
        # === Aroon Indicator ===
        if 'aroon_up' in row and 'aroon_down' in row:
            if row['aroon_up'] > 70:
                signals.append(f"Aroon Up Strong ({row['aroon_up']:.1f})")
            if row['aroon_up'] > row['aroon_down']:
                signals.append(f"Aroon Bullish ({row['aroon_up']:.1f} vs {row['aroon_down']:.1f})")
        
        if 'aroon_osc' in row and row['aroon_osc'] > 50:
            signals.append(f"Aroon Oscillator Bullish ({row['aroon_osc']:.1f})")
        
        # === TRIX ===
        if 'trix' in row and 'trix_signal' in row:
            if row['trix'] > row['trix_signal']:
                signals.append("TRIX Above Signal")
            if row['trix'] > 0:
                signals.append(f"TRIX Positive ({row['trix']:.4f})")
        
        # === DPO (Detrended Price Oscillator) ===
        if 'dpo' in row and row['dpo'] > 0:
            signals.append(f"DPO Positive ({row['dpo']:.4f})")
        
        # === KST (Know Sure Thing) ===
        if 'kst' in row and 'kst_signal' in row:
            if row['kst'] > row['kst_signal']:
                signals.append("KST Above Signal")
        
        # === Mass Index ===
        if 'mass_index' in row and row['mass_index'] > 27:
            signals.append(f"Mass Index Reversal Signal ({row['mass_index']:.2f})")
        
        # === Awesome Oscillator ===
        if 'ao' in row and row['ao'] > 0:
            signals.append(f"Awesome Oscillator Positive ({row['ao']:.4f})")
        
        # === Ichimoku Cloud ===
        if 'close' in row:
            close = row['close']
            
            # Price above cloud
            if 'ichimoku_a' in row and 'ichimoku_b' in row:
                cloud_top = max(row['ichimoku_a'], row['ichimoku_b'])
                if close > cloud_top:
                    signals.append("Price Above Ichimoku Cloud")
            
            # Tenkan-Kijun cross
            if 'ichimoku_conversion' in row and 'ichimoku_base' in row:
                if row['ichimoku_conversion'] > row['ichimoku_base']:
                    signals.append("Ichimoku TK Cross Bullish")
        
        # === Parabolic SAR ===
        if 'psar' in row and 'close' in row:
            if row['close'] > row['psar']:
                signals.append("Price Above PSAR")
        
        # === Vortex Indicator ===
        if 'vortex_pos' in row and 'vortex_neg' in row:
            if row['vortex_pos'] > row['vortex_neg']:
                signals.append(f"Vortex Bullish (+VI={row['vortex_pos']:.2f})")
        
        if 'vortex_diff' in row and row['vortex_diff'] > 0:
            signals.append(f"Vortex Diff Positive ({row['vortex_diff']:.2f})")
        
        return signals
    
    def _check_trend_bearish(self, row: pd.Series) -> List[str]:
        """Check all trend indicators for bearish signals."""
        signals = []
        
        # === MACD Family ===
        if 'macd_hist' in row and row['macd_hist'] < 0:
            signals.append(f"MACD Histogram Negative ({row['macd_hist']:.4f})")
        
        if 'macd' in row and 'macd_signal' in row:
            if row['macd'] < row['macd_signal']:
                signals.append("MACD Below Signal")
                if row['macd'] < 0:
                    signals.append("MACD in Negative Territory")
        
        # === PPO ===
        if 'ppo' in row and row['ppo'] < 0:
            signals.append(f"PPO Negative ({row['ppo']:.2f})")
        
        if 'ppo_hist' in row and row['ppo_hist'] < 0:
            signals.append(f"PPO Histogram Negative ({row['ppo_hist']:.2f})")
        
        # === ADX & Directional Movement ===
        if 'adx' in row and row['adx'] > self.adx_trend:
            if 'di_plus' in row and 'di_minus' in row:
                if row['di_minus'] > row['di_plus']:
                    strength = "Strong" if row['adx'] > 30 else "Moderate"
                    signals.append(f"ADX {strength} Bearish Trend ({row['adx']:.1f})")
                    signals.append(f"-DI > +DI ({row['di_minus']:.1f} vs {row['di_plus']:.1f})")
        
        # === Aroon Indicator ===
        if 'aroon_up' in row and 'aroon_down' in row:
            if row['aroon_down'] > 70:
                signals.append(f"Aroon Down Strong ({row['aroon_down']:.1f})")
            if row['aroon_down'] > row['aroon_up']:
                signals.append(f"Aroon Bearish ({row['aroon_down']:.1f} vs {row['aroon_up']:.1f})")

        if 'aroon_osc' in row and row['aroon_osc'] < -50:
            signals.append(f"Aroon Oscillator Bearish ({row['aroon_osc']:.1f})")
        
        # === TRIX ===
        if 'trix' in row and 'trix_signal' in row:
            if row['trix'] < row['trix_signal']:
                signals.append("TRIX Below Signal")
            if row['trix'] < 0:
                signals.append(f"TRIX Negative ({row['trix']:.4f})")
        
        # === DPO ===
        if 'dpo' in row and row['dpo'] < 0:
            signals.append(f"DPO Negative ({row['dpo']:.4f})")
        
        # === KST ===
        if 'kst' in row and 'kst_signal' in row:
            if row['kst'] < row['kst_signal']:
                signals.append("KST Below Signal")
        
        # === Awesome Oscillator ===
        if 'ao' in row and row['ao'] < 0:
            signals.append(f"Awesome Oscillator Negative ({row['ao']:.4f})")
        
        # === Ichimoku Cloud ===
        if 'close' in row:
            close = row['close']
            
            # Price below cloud
            if 'ichimoku_a' in row and 'ichimoku_b' in row:
                cloud_bottom = min(row['ichimoku_a'], row['ichimoku_b'])
                if close < cloud_bottom:
                    signals.append("Price Below Ichimoku Cloud")
            
            # Tenkan-Kijun cross
            if 'ichimoku_conversion' in row and 'ichimoku_base' in row:
                if row['ichimoku_conversion'] < row['ichimoku_base']:
                    signals.append("Ichimoku TK Cross Bearish")
        
        # === Parabolic SAR ===
        if 'psar' in row and 'close' in row:
            if row['close'] < row['psar']:
                signals.append("Price Below PSAR")
        
        # === Vortex Indicator ===
        if 'vortex_pos' in row and 'vortex_neg' in row:
            if row['vortex_neg'] > row['vortex_pos']:
                signals.append(f"Vortex Bearish (-VI={row['vortex_neg']:.2f})")
        
        if 'vortex_diff' in row and row['vortex_diff'] < 0:
            signals.append(f"Vortex Diff Negative ({row['vortex_diff']:.2f})")
        
        return signals

    # =========================================================================
    # VOLUME INDICATOR CHECKS
    # =========================================================================
    
    def _check_volume_bullish(self, row: pd.Series) -> List[str]:
        """Check all volume indicators for bullish signals."""
        signals = []
        
        # === On-Balance Volume (OBV) ===
        if 'obv_change' in row and row['obv_change'] > 0:
            signals.append("OBV Rising")
        
        # === Chaikin Money Flow (CMF) ===
        if 'cmf' in row:
            if row['cmf'] > 0.1:
                signals.append(f"CMF Strong Positive ({row['cmf']:.3f})")
            elif row['cmf'] > 0:
                signals.append(f"CMF Positive ({row['cmf']:.3f})")
        
        # === Money Flow Index (MFI) ===
        if 'mfi' in row:
            if row['mfi'] > 50:
                signals.append(f"MFI Bullish ({row['mfi']:.1f})")
            if row['mfi'] > 80:
                signals.append(f"MFI Overbought ({row['mfi']:.1f})")
        
        # === Force Index ===
        if 'force_index' in row and row['force_index'] > 0:
            signals.append("Force Index Positive")
        
        if 'force_index_ema' in row and row['force_index_ema'] > 0:
            signals.append("Force Index EMA Bullish")
        
        # === Accumulation/Distribution (A/D) ===
        if 'ad' in row and row['ad'] > 0:
            signals.append("A/D Line Positive")
        
        # === Chaikin Oscillator ===
        if 'chaikin_osc' in row and row['chaikin_osc'] > 0:
            signals.append(f"Chaikin Osc Positive ({row['chaikin_osc']:.2e})")
        
        # === Ease of Movement (EOM) ===
        if 'eom' in row and row['eom'] > 0:
            signals.append("Ease of Movement Positive")
        
        if 'eom_sma' in row and row['eom_sma'] > 0:
            signals.append("EOM SMA Positive")
        
        # === Volume Price Trend (VPT) ===
        if 'vpt' in row and row['vpt'] > 0:
            signals.append("VPT Rising")
        
        # === Negative Volume Index (NVI) ===
        if 'nvi' in row:
            # NVI rising indicates smart money accumulation
            # Compare to a baseline or previous value if available
            signals.append(f"NVI Value: {row['nvi']:.2f}")
        
        # === Volume Z-Score ===
        if 'volume_zscore' in row:
            if row['volume_zscore'] > 2.0:
                signals.append(f"Extreme Volume Surge (Z={row['volume_zscore']:.2f})")
            elif row['volume_zscore'] > 1.0:
                signals.append(f"High Volume (Z={row['volume_zscore']:.2f})")
        
        # === Volume ROC ===
        if 'volume_roc' in row and row['volume_roc'] > 0.5:
            signals.append(f"Volume Increasing ({row['volume_roc']:.2%})")
        
        # === Balance of Power (BOP) ===
        if 'bop' in row and row['bop'] > 0.5:
            signals.append(f"Balance of Power Bullish ({row['bop']:.2f})")
        
        # === Volume Spike ===
        if 'volume_spike' in row and row['volume_spike'] == 1:
            signals.append("Volume Spike Detected")
        
        return signals
    
    def _check_volume_bearish(self, row: pd.Series) -> List[str]:
        """Check all volume indicators for bearish signals."""
        signals = []
        
        # === OBV ===
        if 'obv_change' in row and row['obv_change'] < 0:
            signals.append("OBV Falling")
        
        # === CMF ===
        if 'cmf' in row:
            if row['cmf'] < -0.1:
                signals.append(f"CMF Strong Negative ({row['cmf']:.3f})")
            elif row['cmf'] < 0:
                signals.append(f"CMF Negative ({row['cmf']:.3f})")
        
        # === MFI ===
        if 'mfi' in row:
            if row['mfi'] < 50:
                signals.append(f"MFI Bearish ({row['mfi']:.1f})")
            if row['mfi'] < 20:
                signals.append(f"MFI Oversold ({row['mfi']:.1f})")
        
        # === Force Index ===
        if 'force_index' in row and row['force_index'] < 0:
            signals.append("Force Index Negative")
        
        if 'force_index_ema' in row and row['force_index_ema'] < 0:
            signals.append("Force Index EMA Bearish")
        
        # === A/D ===
        if 'ad' in row and row['ad'] < 0:
            signals.append("A/D Line Negative")
        
        # === Chaikin Oscillator ===
        if 'chaikin_osc' in row and row['chaikin_osc'] < 0:
            signals.append(f"Chaikin Osc Negative ({row['chaikin_osc']:.2e})")
        
        # === EOM ===
        if 'eom' in row and row['eom'] < 0:
            signals.append("Ease of Movement Negative")
        
        if 'eom_sma' in row and row['eom_sma'] < 0:
            signals.append("EOM SMA Negative")
        
        # === VPT ===
        if 'vpt' in row and row['vpt'] < 0:
            signals.append("VPT Falling")
        
        # === Volume ROC ===
        if 'volume_roc' in row and row['volume_roc'] < -0.3:
            signals.append(f"Volume Decreasing ({row['volume_roc']:.2%})")
        
        # === BOP ===
        if 'bop' in row and row['bop'] < -0.5:
            signals.append(f"Balance of Power Bearish ({row['bop']:.2f})")
        
        # === Low Volume Warning ===
        if 'volume_decreasing' in row and row['volume_decreasing'] == 1:
            signals.append("Volume Decreasing (Potential Reversal)")
        
        return signals

    # =========================================================================
    # VOLATILITY INDICATOR CHECKS
    # =========================================================================
    
    def _check_volatility_bullish(self, row: pd.Series) -> List[str]:
        """Check volatility indicators for bullish context."""
        signals = []
        
        # === Bollinger Bands ===
        if 'bb_pct' in row:
            pct = row['bb_pct']
            if pct < 0.2:
                signals.append(f"Price Near Lower BB ({pct:.2%}) - Potential Bounce")
            elif pct > 0.8:
                signals.append(f"Price Near Upper BB ({pct:.2%}) - Strong Momentum")
        
        if 'bb_width' in row:
            if row['bb_width'] < 0.02:
                signals.append(f"BB Squeeze ({row['bb_width']:.4f}) - Breakout Potential")
            elif row['bb_width'] > 0.05:
                signals.append(f"BB Expansion ({row['bb_width']:.4f}) - High Volatility")
        
        # === Keltner Channels ===
        if 'kc_pct' in row:
            pct = row['kc_pct']
            if 0.3 < pct < 0.7:
                signals.append("Price Mid-Keltner (Neutral Zone)")
            elif pct > 0.8:
                signals.append("Price Near Upper KC (Strong)")
        
        # === ATR ===
        if 'atr_percent' in row:
            atr_pct = row['atr_percent']
            if atr_pct < 1.5:
                signals.append(f"Low Volatility (ATR={atr_pct:.2f}%) - Consolidation")
            elif atr_pct > 3.0:
                signals.append(f"High Volatility (ATR={atr_pct:.2f}%) - Trending")
        
        if 'atr_ratio' in row:
            signals.append(f"ATR Ratio: {row['atr_ratio']:.4f}")
        
        # === Donchian Channels ===
        if 'dc_pct' in row and row['dc_pct'] > 0.8:
            signals.append("Price Near Donchian Upper (Breakout)")
        
        # === Ulcer Index ===
        if 'ulcer_index' in row and row['ulcer_index'] < 2:
            signals.append(f"Low Drawdown Risk (Ulcer={row['ulcer_index']:.2f})")
        
        # === Bull/Bear Power ===
        if 'bull_power' in row and row['bull_power'] > 0:
            signals.append(f"Bull Power Positive ({row['bull_power']:.4f})")
        
        if 'bear_power' in row and row['bear_power'] > -0.5:
            signals.append("Bear Power Weakening")
        
        # === Choppiness Index ===
        if 'choppiness' in row:
            if row['choppiness'] < 40:
                signals.append(f"Low Chop ({row['choppiness']:.1f}) - Trending")
        
        # === Normalized Variance ===
        if 'normalized_variance' in row and row['normalized_variance'] < 0.02:
            signals.append(f"Low Variance ({row['normalized_variance']:.4f})")
        
        return signals
    
    def _check_volatility_bearish(self, row: pd.Series) -> List[str]:
        """Check volatility indicators for bearish context."""
        signals = []
        
        # === Bollinger Bands ===
        if 'bb_pct' in row:
            pct = row['bb_pct']
            if pct > 0.8:
                signals.append(f"Price Near Upper BB ({pct:.2%}) - Potential Reversal")
            elif pct < 0.2:
                signals.append(f"Price Near Lower BB ({pct:.2%}) - Weak Momentum")
        
        # === Keltner Channels ===
        if 'kc_pct' in row and row['kc_pct'] < 0.2:
            signals.append("Price Near Lower KC (Weak)")
        
        # === Ulcer Index ===
        if 'ulcer_index' in row and row['ulcer_index'] > 5:
            signals.append(f"High Drawdown Risk (Ulcer={row['ulcer_index']:.2f})")
        
        # === Bull/Bear Power ===
        if 'bull_power' in row and row['bull_power'] < -0.5:
            signals.append("Bull Power Weakening")
        
        if 'bear_power' in row and row['bear_power'] < 0:
            signals.append(f"Bear Power Negative ({row['bear_power']:.4f})")
        
        # === Choppiness Index ===
        if 'choppiness' in row and row['choppiness'] > 60:
            signals.append(f"High Chop ({row['choppiness']:.1f}) - Sideways")
        
        return signals

    # =========================================================================
    # MOVING AVERAGE CHECKS
    # =========================================================================
    
    def _check_ma_alignment_bullish(self, row: pd.Series) -> List[str]:
        """Check moving average alignment for bullish signals."""
        signals = []
        
        if 'close' not in row:
            return signals
        
        close = row['close']
        
        # === Price Above MAs ===
        ma_checks = {
            'ema_9': 'EMA(9)',
            'ema_26': 'EMA(26)',
            'sma_50': 'SMA(50)',
            'sma_200': 'SMA(200)',
            'ema_200': 'EMA(200)',
            'vwma': 'VWMA',
            'vwap': 'VWAP'
        }
        
        for ma_col, ma_name in ma_checks.items():
            if ma_col in row and pd.notna(row[ma_col]):
                if close > row[ma_col]:
                    pct_above = ((close - row[ma_col]) / row[ma_col]) * 100
                    signals.append(f"Price > {ma_name} (+{pct_above:.2f}%)")
        
        # === MA Crosses (Golden Cross, etc.) ===
        if 'ema_9' in row and 'ema_26' in row:
            if row['ema_9'] > row['ema_26']:
                signals.append("Fast EMA > Slow EMA")
        
        if 'sma_50' in row and 'sma_200' in row:
            if row['sma_50'] > row['sma_200']:
                signals.append("Golden Cross (SMA50 > SMA200)")
        
        if 'ema_50' in row and 'ema_200' in row:
            if row['ema_50'] > row['ema_200']:
                signals.append("EMA Golden Cross (EMA50 > EMA200)")
        
        # === MA Sequence (Perfect Alignment) ===
        ma_sequence = []
        for ma in ['ema_9', 'ema_26', 'sma_50', 'sma_200']:
            if ma in row:
                ma_sequence.append((ma, row[ma]))
        
        if len(ma_sequence) >= 3:
            sorted_mas = sorted(ma_sequence, key=lambda x: x[1], reverse=True)
            if sorted_mas == ma_sequence:
                signals.append("Perfect Bullish MA Alignment")
        
        # === Distance from Key MAs ===
        if 'sma_200' in row and pd.notna(row['sma_200']):
            dist = ((close - row['sma_200']) / row['sma_200']) * 100
            if 0 < dist < 5:
                signals.append(f"Near SMA(200) Support ({dist:.2f}%)")
        
        return signals
    
    def _check_ma_alignment_bearish(self, row: pd.Series) -> List[str]:
        """Check moving average alignment for bearish signals."""
        signals = []
        
        if 'close' not in row:
            return signals
        
        close = row['close']
        
        # === Price Below MAs ===
        ma_checks = {
            'ema_9': 'EMA(9)',
            'ema_26': 'EMA(26)',
            'sma_50': 'SMA(50)',
            'sma_200': 'SMA(200)',
            'ema_200': 'EMA(200)',
            'vwma': 'VWMA',
            'vwap': 'VWAP'
        }
        
        for ma_col, ma_name in ma_checks.items():
            if ma_col in row and pd.notna(row[ma_col]):
                if close < row[ma_col]:
                    pct_below = ((row[ma_col] - close) / row[ma_col]) * 100
                    signals.append(f"Price < {ma_name} (-{pct_below:.2f}%)")
        
        # === MA Crosses (Death Cross, etc.) ===
        if 'ema_9' in row and 'ema_26' in row:
            if row['ema_9'] < row['ema_26']:
                signals.append("Fast EMA < Slow EMA")
        
        if 'sma_50' in row and 'sma_200' in row:
            if row['sma_50'] < row['sma_200']:
                signals.append("Death Cross (SMA50 < SMA200)")
        
        if 'ema_50' in row and 'ema_200' in row:
            if row['ema_50'] < row['ema_200']:
                signals.append("EMA Death Cross (EMA50 < EMA200)")
        
        # === MA Sequence (Perfect Bearish Alignment) ===
        ma_sequence = []
        for ma in ['ema_9', 'ema_26', 'sma_50', 'sma_200']:
            if ma in row:
                ma_sequence.append((ma, row[ma]))
        
        if len(ma_sequence) >= 3:
            sorted_mas = sorted(ma_sequence, key=lambda x: x[1])
            if sorted_mas == ma_sequence:
                signals.append("Perfect Bearish MA Alignment")
        
        # === Distance from Key MAs ===
        if 'sma_200' in row and pd.notna(row['sma_200']):
            dist = ((row['sma_200'] - close) / row['sma_200']) * 100
            if 0 < dist < 5:
                signals.append(f"Near SMA(200) Resistance ({dist:.2f}%)")
        
        return signals

    # =========================================================================
    # FIBONACCI & SUPPORT/RESISTANCE CHECKS
    # =========================================================================
    
    def _check_fibonacci_bullish(self, row: pd.Series) -> List[str]:
        """Check Fibonacci levels for bullish setups."""
        signals = []
        
        # === Near Fibonacci Levels ===
        fib_levels = {
            'near_fib_236': '23.6% Retracement',
            'near_fib_382': '38.2% Retracement',
            'near_fib_500': '50% Retracement',
            'near_fib_618': '61.8% Golden Ratio',
            'near_fib_786': '78.6% Retracement'
        }
        
        for fib_col, fib_name in fib_levels.items():
            if fib_col in row and row[fib_col] == 1:
                signals.append(f"Near Fib {fib_name} (Bounce Zone)")
        
        # === Support/Resistance Confluence ===
        if 'sr_confluence_score' in row and row['sr_confluence_score'] >= 2:
            signals.append(f"S/R Confluence ({int(row['sr_confluence_score'])} levels)")
        
        # === Pullback Quality ===
        if 'pullback_quality_score' in row and row['pullback_quality_score'] >= 60:
            signals.append(f"High Quality Pullback (Score={int(row['pullback_quality_score'])})")
        
        return signals
    
    def _check_fibonacci_bearish(self, row: pd.Series) -> List[str]:
        """Check Fibonacci levels for bearish setups."""
        signals = []
        
        # Near resistance Fibonacci levels
        fib_levels = {
            'near_fib_236': '23.6% Extension',
            'near_fib_382': '38.2% Extension',
            'near_fib_618': '61.8% Extension'
        }
        
        for fib_col, fib_name in fib_levels.items():
            if fib_col in row and row[fib_col] == 1:
                signals.append(f"Near Fib {fib_name} (Resistance Zone)")
        
        # S/R confluence as resistance
        if 'sr_confluence_score' in row and row['sr_confluence_score'] >= 2:
            signals.append(f"Resistance Confluence ({int(row['sr_confluence_score'])} levels)")
        
        return signals

    # =========================================================================
    # PRICE ACTION PATTERN CHECKS
    # =========================================================================
    
    def _check_price_action_bullish(self, row: pd.Series) -> List[str]:
        """Check price action patterns for bullish signals."""
        patterns = []
        
        # === Structure Types ===
        if 'structure_type' in row:
            if row['structure_type'] == 'HH':
                patterns.append("Higher High (HH)")
            elif row['structure_type'] == 'HL':
                patterns.append("Higher Low (HL)")
        
        # === Structure Breaks ===
        if 'structure_break' in row and row['structure_break'] == 1:
            patterns.append("Bullish Structure Break")
        
        # === Trend Direction ===
        if 'trend_short' in row and row['trend_short'] == 1:
            patterns.append("Short-Term Uptrend")
        
        if 'trend_medium' in row and row['trend_medium'] == 1:
            patterns.append("Medium-Term Uptrend")
        
        if 'trend_long' in row and row['trend_long'] == 1:
            patterns.append("Long-Term Uptrend")
        
        # === Swing Points ===
        if 'swing_low' in row and row['swing_low'] == 1:
            patterns.append("Swing Low Formed")
        
        # === Directional Persistence ===
        if 'directional_persistence' in row and row['directional_persistence'] > 0.2:
            patterns.append(f"Strong Upward Persistence ({row['directional_persistence']:.2f})")
        
        # === Local Slope ===
        if 'local_slope' in row and row['local_slope'] > 0:
            patterns.append(f"Positive Price Slope ({row['local_slope']:.4f})")
        
        return patterns
    
    def _check_price_action_bearish(self, row: pd.Series) -> List[str]:
        """Check price action patterns for bearish signals."""
        patterns = []
        
        # === Structure Types ===
        if 'structure_type' in row:
            if row['structure_type'] == 'LH':
                patterns.append("Lower High (LH)")
            elif row['structure_type'] == 'LL':
                patterns.append("Lower Low (LL)")
        
        # === Structure Breaks ===
        if 'structure_break' in row and row['structure_break'] == 1:
            patterns.append("Bearish Structure Break")
        
        # === Trend Direction ===
        if 'trend_short' in row and row['trend_short'] == -1:
            patterns.append("Short-Term Downtrend")
        
        if 'trend_medium' in row and row['trend_medium'] == -1:
            patterns.append("Medium-Term Downtrend")
        
        if 'trend_long' in row and row['trend_long'] == -1:
            patterns.append("Long-Term Downtrend")
        
        # === Swing Points ===
        if 'swing_high' in row and row['swing_high'] == 1:
            patterns.append("Swing High Formed")
        
        # === Directional Persistence ===
        if 'directional_persistence' in row and row['directional_persistence'] < -0.2:
            patterns.append(f"Strong Downward Persistence ({row['directional_persistence']:.2f})")
        
        # === Local Slope ===
        if 'local_slope' in row and row['local_slope'] < 0:
            patterns.append(f"Negative Price Slope ({row['local_slope']:.4f})")
        
        return patterns

    # =========================================================================
    # PULLBACK PATTERN CHECKS
    # =========================================================================
    
    def _check_pullback_bullish(self, row: pd.Series) -> List[str]:
        """Check pullback patterns for bullish continuation."""
        patterns = []
        
        # === Pullback Completion ===
        if 'pullback_complete_bull' in row and row['pullback_complete_bull'] == 1:
            patterns.append("✅ Bullish Pullback Complete")
        
        # === Healthy Pullback ===
        if 'healthy_bull_pullback' in row and row['healthy_bull_pullback'] == 1:
            patterns.append("Healthy Bull Pullback")
        
        # === Higher Lows Pattern ===
        if 'higher_lows_pattern' in row and row['higher_lows_pattern'] == 1:
            patterns.append("Higher Lows Pattern (Accumulation)")
        
        # === ABC Pullback ===
        if 'abc_pullback_bull' in row and row['abc_pullback_bull'] == 1:
            patterns.append("ABC Bullish Continuation Pattern")
        
        # === Pullback to Moving Averages ===
        if 'pullback_to_ema9' in row and row['pullback_to_ema9'] == 1:
            patterns.append("Pullback to EMA(9)")
        
        if 'pullback_to_ema20' in row and row['pullback_to_ema20'] == 1:
            patterns.append("Pullback to EMA(20)")
        
        if 'pullback_to_sma50' in row and row['pullback_to_sma50'] == 1:
            patterns.append("Pullback to SMA(50)")
        
        # === Pullback Depth ===
        if 'pullback_from_high_pct' in row:
            depth = abs(row['pullback_from_high_pct'])
            if 3 <= depth <= 12:
                patterns.append(f"Ideal Pullback Depth ({depth:.1f}%)")
        
        # === Measured Move ===
        if 'distance_to_bull_target_pct' in row and row['distance_to_bull_target_pct'] > 0:
            patterns.append(f"Measured Move Target +{row['distance_to_bull_target_pct']:.1f}%")
        
        # === Pullback Stage ===
        if 'pullback_stage' in row:
            stage = row['pullback_stage']
            if stage == 'resumption_bull':
                patterns.append("🚀 Trend Resumption Phase")
            elif stage == 'pullback_late':
                patterns.append("Late Pullback (Entry Zone)")
        
        return patterns
    
    def _check_pullback_bearish(self, row: pd.Series) -> List[str]:
        """Check pullback patterns for bearish continuation."""
        patterns = []
        
        # === Pullback Completion ===
        if 'pullback_complete_bear' in row and row['pullback_complete_bear'] == 1:
            patterns.append("✅ Bearish Pullback Complete")
        
        # === Healthy Pullback ===
        if 'healthy_bear_pullback' in row and row['healthy_bear_pullback'] == 1:
            patterns.append("Healthy Bear Pullback")
        
        # === Lower Highs Pattern ===
        if 'lower_highs_pattern' in row and row['lower_highs_pattern'] == 1:
            patterns.append("Lower Highs Pattern (Distribution)")
        
        # === ABC Pullback ===
        if 'abc_pullback_bear' in row and row['abc_pullback_bear'] == 1:
            patterns.append("ABC Bearish Continuation Pattern")
        
        # === Failed Pullback ===
        if 'failed_pullback_bear' in row and row['failed_pullback_bear'] == 1:
            patterns.append("Failed Bear Pullback (Reversal Risk)")
        
        # === Measured Move ===
        if 'distance_to_bear_target_pct' in row and row['distance_to_bear_target_pct'] > 0:
            patterns.append(f"Measured Move Target -{row['distance_to_bear_target_pct']:.1f}%")
        
        # === Pullback Stage ===
        if 'pullback_stage' in row:
            stage = row['pullback_stage']
            if stage == 'resumption_bear':
                patterns.append("📉 Downtrend Resumption Phase")
            elif stage == 'pullback_late':
                patterns.append("Late Pullback (Short Entry Zone)")
        return patterns

    # =========================================================================
    # CANDLESTICK PATTERN CHECKS
    # =========================================================================
    
    def _check_candlestick_patterns(self, row: pd.Series, direction: str) -> List[str]:
        """
        Check candlestick patterns.
        
        Args:
            row: DataFrame row with pattern columns
            direction: 'bullish', 'bearish', or 'neutral'
        """
        patterns = []
        
        pattern_list = self.candlestick_patterns.get(direction, [])
        
        for pattern_col in pattern_list:
            if pattern_col in row and row[pattern_col] == 1:
                # Convert pattern name to human-readable
                name = pattern_col.replace('pattern_', '').replace('_', ' ').title()
                
                # Add emoji for visual clarity
                if direction == 'bullish':
                    patterns.append(f"🟢 Candlestick: {name}")
                elif direction == 'bearish':
                    patterns.append(f"🔴 Candlestick: {name}")
                else:
                    patterns.append(f"⚪ Candlestick: {name}")
        
        # === Recent Pattern Count ===
        if direction == 'bullish' and 'recent_bullish_patterns' in row:
            if row['recent_bullish_patterns'] > 1:
                patterns.append(f"Multiple Bullish Candles ({int(row['recent_bullish_patterns'])} recent)")
        
        if direction == 'bearish' and 'recent_bearish_patterns' in row:
            if row['recent_bearish_patterns'] > 1:
                patterns.append(f"Multiple Bearish Candles ({int(row['recent_bearish_patterns'])} recent)")
        
        if direction == 'neutral' and 'recent_doji' in row:
            if row['recent_doji'] > 0:
                patterns.append(f"Indecision Candles ({int(row['recent_doji'])} recent)")
        
        return patterns

    # =========================================================================
    # CHART PATTERN CHECKS
    # =========================================================================
    
    def _check_chart_patterns(self, row: pd.Series, direction: str) -> List[str]:
        """
        Check chart patterns.
        
        Args:
            row: DataFrame row with pattern columns
            direction: 'bullish', 'bearish', or 'neutral'
        """
        patterns = []
        
        pattern_list = self.chart_patterns.get(direction, [])
        
        for pattern_col in pattern_list:
            if pattern_col in row and row[pattern_col] == 1:
                # Convert pattern name to human-readable
                name = pattern_col.replace('_', ' ').title()
                
                # Add emoji and context
                if direction == 'bullish':
                    patterns.append(f"📈 Chart Pattern: {name}")
                elif direction == 'bearish':
                    patterns.append(f"📉 Chart Pattern: {name}")
                else:
                    patterns.append(f"↔️ Chart Pattern: {name}")
        
        return patterns

    # =========================================================================
    # ANOMALY DETECTION
    # =========================================================================
    
    def _check_anomalies_bullish(self, row: pd.Series) -> List[str]:
        """Check for bullish price action anomalies."""
        signals = []
        
        # === Gap Detection ===
        if 'gap_intensity' in row and row['gap_intensity'] > 0.01:
            signals.append(f"Gap Up ({row['gap_intensity']:.2%})")
        
        # === Extended Bars ===
        if 'extended_bar_ratio' in row and row['extended_bar_ratio'] > 2.0:
            signals.append(f"Extended Bullish Bar (Ratio={row['extended_bar_ratio']:.2f})")
        
        # === Volume Anomalies ===
        if 'volume_spike' in row and row['volume_spike'] == 1:
            signals.append("⚡ Volume Spike (Institutional Activity)")
        
        return signals
    
    def _check_anomalies_bearish(self, row: pd.Series) -> List[str]:
        """Check for bearish price action anomalies."""
        signals = []
        
        # === Gap Detection ===
        if 'gap_intensity' in row and row['gap_intensity'] < -0.01:
            signals.append(f"Gap Down ({row['gap_intensity']:.2%})")
        
        # === Extended Bars ===
        if 'extended_bar_ratio' in row and row['extended_bar_ratio'] > 2.0:
            # Check if bearish extended bar
            if 'close' in row and 'open' in row and row['close'] < row['open']:
                signals.append(f"Extended Bearish Bar (Ratio={row['extended_bar_ratio']:.2f})")
        
        # === Volume Anomalies ===
        if 'volume_spike' in row and row['volume_spike'] == 1:
            signals.append("⚡ Volume Spike (Distribution Alert)")
        
        return signals

# ============================================================================
# SECTION 2: DISCOVERY SYSTEM (FIXED)
# ============================================================================

class RegimeStrategyDiscovery:
    """
    Builds the Repository of Strategies by analyzing HMM regime-labeled swings.
    
    FIXED: Now uses regime_id as dictionary keys throughout.
    """
    
    def __init__(self, regime_system: AdvancedRegimeDetectionSystem):
        if not regime_system.hmm_classifier or not regime_system.hmm_classifier.regime_states:
            raise ValueError("The AdvancedRegimeDetectionSystem has not been run or failed to train.")
        
        self.regime_system = regime_system
        self.regime_states_map: Dict[int, RegimeState] = regime_system.hmm_classifier.regime_states
        self.indicator_rules = StrategyIndicatorRules()
        self.strategy_repository: Dict[int, Dict] = {}  # Changed key type to int
        self.stats_analyzer = None # Will be injected if DB available
    
    def discover_strategies(self) -> Dict[str, Dict]:
        """
        Analyzes the HybridSwingRegistry and builds the strategy repository per *regime instance*.
        Returns:
            Dict[str, Dict]: mapping regime_instance_id -> playbook metadata
        """
        
        swing_df = self.regime_system.registry.to_dataframe()
        
        if 'hmm_regime' not in swing_df.columns:
            print("  ⚠️  'hmm_regime' column not found. Run HMM first.")
            return {}
        
        # Prefer instance-level key if available
        if 'regime_instance_id' in swing_df.columns:
            keys = swing_df['regime_instance_id'].dropna().unique().tolist()
        else:
            keys = swing_df['hmm_regime'].dropna().unique().astype(int).tolist()
            keys = [f"R{k}" for k in keys]

        # Initialize repositories
        temp_repository: Dict[str, Dict[str, Set]] = {}
        instance_meta: Dict[str, Dict] = {}
        
        for inst in keys:
            temp_repository[inst] = {
                'confirming_indicators': set(),
                'strategy_patterns': set()
            }

        # ========================================================================
        # Group swings by instance FIRST
        # ========================================================================
        instance_groups = {}
        for _, row in swing_df.iterrows():
            inst_key = None
            if 'regime_instance_id' in swing_df.columns and pd.notna(row.get('regime_instance_id')):
                inst_key = row['regime_instance_id']
            elif pd.notna(row.get('hmm_regime')):
                inst_key = f"R{int(row['hmm_regime'])}"
            else:
                continue

            if inst_key not in instance_groups:
                instance_groups[inst_key] = []
            
            instance_groups[inst_key].append(row)
        
        # ========================================================================
        # Process each instance with aggregated data
        # ========================================================================
        for inst_key, swings in instance_groups.items():
            if len(swings) == 0:
                continue
                
            if inst_key not in temp_repository:
                temp_repository[inst_key] = {
                    'confirming_indicators': set(),
                    'strategy_patterns': set()
                }
            
            # Get regime metadata
            first_swing = swings[0]
            regime_id = None
            if 'hmm_regime' in first_swing and pd.notna(first_swing['hmm_regime']):
                regime_id = int(first_swing['hmm_regime'])
            
            regime_state = self.regime_states_map.get(regime_id) if regime_id is not None else None
            
            # Get regime label
            regime_label = None
            if hasattr(self.regime_system, 'regime_type_map') and regime_id is not None:
                regime_label = self.regime_system.regime_type_map.get(regime_id)
            
            if not regime_label and 'regime_type' in first_swing and pd.notna(first_swing.get('regime_type')):
                regime_label = first_swing['regime_type']
            
            if not regime_label and regime_state:
                regime_label = regime_state.name
            
            if not regime_label:
                regime_label = f"Regime {regime_id}" if regime_id is not None else "Unknown"
            
            # ====================================================================
            # Create AGGREGATED row (median values across all swings)
            # ====================================================================
            aggregated_row = pd.Series()
            
            for col in swing_df.columns:
                values = [s[col] for s in swings if pd.notna(s.get(col))]
                
                if len(values) == 0:
                    continue
                
                # Numeric: use median
                if pd.api.types.is_numeric_dtype(swing_df[col]):
                    aggregated_row[col] = np.median(values)
                # Categorical: use mode (most common)
                else:
                    from collections import Counter
                    counter = Counter(values)
                    aggregated_row[col] = counter.most_common(1)[0][0]
            
            # ====================================================================
            # Get confirmations from SINGLE aggregated row
            # ====================================================================
            indicators, patterns = [], []
            if regime_state:
                if regime_state.trend_direction == 'bull':
                    indicators, patterns = self.indicator_rules.get_bullish_confirmations(aggregated_row)
                elif regime_state.trend_direction == 'bear':
                    indicators, patterns = self.indicator_rules.get_bearish_confirmations(aggregated_row)
                else:
                    indicators, patterns = self.indicator_rules.get_ranging_confirmations(aggregated_row)
            else:
                indicators, patterns = self.indicator_rules.get_ranging_confirmations(aggregated_row)

            # --- STATISTICAL VALIDATION FILTER (NEW) ---
            # If we have a stats analyzer, verify these indicators actually matter for this regime
            if self.stats_analyzer:
                validated_indicators = []
                for ind in indicators:
                    # Strip value to get name "RSI > 55 (60.2)" -> "RSI"
                    ind_name = ind.split(' ')[0] 
                    # Run Kruskal-Wallis
                    stat_result = self.stats_analyzer.analyze_numeric_factor(ind_name)
                    if stat_result.get('significant', False): # Only keep if p < 0.05
                        validated_indicators.append(ind)
                
                if validated_indicators:
                    indicators = validated_indicators

            # Add to repository (sets auto-deduplicate)
            temp_repository[inst_key]['confirming_indicators'].update(indicators)
            temp_repository[inst_key]['strategy_patterns'].update(patterns)

            # Store metadata
            if inst_key not in instance_meta:
                instance_meta[inst_key] = {
                    'regime_type': regime_id,
                    'regime_label': regime_label,
                    'regime_name': regime_state.name if regime_state else f"R{regime_id}",
                    'trend_direction': regime_state.trend_direction if regime_state else 'unknown',
                    'volatility_level': regime_state.volatility_level if regime_state else 'unknown',
                }

        # ========================================================================
        # Build final repository
        # ========================================================================
        self.strategy_repository = {}
        for inst_key, sets in temp_repository.items():
            meta = instance_meta.get(inst_key, {})
            self.strategy_repository[inst_key] = {
                'regime_instance_id': inst_key,
                'regime_type': meta.get('regime_type'),
                'regime_label': meta.get('regime_label'),
                'regime_name': meta.get('regime_name'),
                'trend_direction': meta.get('trend_direction', 'unknown'),
                'volatility_level': meta.get('volatility_level', 'unknown'),
                'confirming_indicators': sorted(list(sets['confirming_indicators'])),
                'strategy_patterns': sorted(list(sets['strategy_patterns']))
            }

        return self.strategy_repository

    def get_repository(self) -> Dict[int, Dict]:
        """Returns the last built strategy repository."""
        return self.strategy_repository

    def print_repository_summary(self):
        """Prints a human-readable summary of the discovered playbook."""
        if not self.strategy_repository:
            print("Repository is empty. Run discover_strategies() first.")
            return
            
        print("\n" + "="*80)
        print("STRATEGY REPOSITORY SUMMARY (PLAYBOOK)")
        print("="*80)
        
        for regime_id, data in self.strategy_repository.items():
            # Handle None values safely
            regime_name = data.get('regime_name', 'Unknown')
            regime_label = data.get('regime_label', regime_name)
            trend = data.get('trend_direction', 'unknown')
            volatility = data.get('volatility_level', 'unknown')
            
            # Use regime_label if available, otherwise use regime_name
            display_name = regime_label if regime_label else regime_name
            if display_name is None:
                display_name = f"Regime {regime_id}"
            
            print(f"\n--- REGIME {regime_id}: {display_name.upper()} ---")
            print(f"    Type: {trend} trend, {volatility} volatility")
            
            print("\n    CONFIRMING INDICATORS:")
            indicators = data.get('confirming_indicators', [])
            if indicators:
                for ind in indicators:
                    print(f"      • {ind}")
            else:
                print("      (None found)")
                
            print("\n    STRATEGY & PRICE ACTION PATTERNS:")
            patterns = data.get('strategy_patterns', [])
            if patterns:
                for pat in patterns:
                    print(f"      • {pat}")
            else:
                print("      (None found)")
        
        print("\n" + "="*80)


# ============================================================================
# SECTION 3: USAGE EXAMPLE
# ============================================================================

def create_mock_indicator_data(length: int) -> pd.DataFrame:
    """Creates mock DataFrame with all required indicators."""
    print(f"  Generating {length} bars of mock indicator data...")
    dates = pd.date_range(start='2023-01-01', periods=length, freq='h')
    df = pd.DataFrame(index=dates)
    
    df['close'] = 100 + np.cumsum(np.random.randn(length) * 0.5)
    df['high'] = df['close'] + np.random.rand(length) * 2
    df['low'] = df['close'] - np.random.rand(length) * 2
    df['open'] = df['low'] + (df['high'] - df['low']) * np.random.rand(length)
    df['volume'] = np.random.randint(100, 1000, size=length)
    
    df['atr'] = np.random.rand(length) + 1
    df['bb_width'] = np.random.rand(length) * 0.05
    df['kc_width'] = np.random.rand(length) * 0.04
    df['rsi'] = np.random.randint(20, 80, size=length)
    df['roc'] = np.random.randn(length) * 0.1
    df['ppo'] = np.random.randn(length) * 0.5
    df['macd_hist'] = np.random.randn(length)
    df['adx'] = np.random.randint(10, 40, size=length)
    df['obv'] = np.cumsum(df['volume'] * np.sign(df['close'].diff().fillna(0)))
    df['sma_50'] = df['close'].rolling(50).mean().fillna(method='bfill')
    
    df['pullback_from_high_pct'] = np.random.rand(length) * 0.5
    df['structure_break_bullish'] = (np.random.rand(length) > 0.9).astype(int)
    df['structure_break_bearish'] = (np.random.rand(length) > 0.9).astype(int)
    df['pattern_hammer'] = (np.random.rand(length) > 0.95).astype(int)
    df['pattern_bearish_engulfing'] = (np.random.rand(length) > 0.95).astype(int)
    df['pattern_doji'] = (np.random.rand(length) > 0.9).astype(int)
    
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


if __name__ == "__main__":
    try:
        from .analysis_advanced_regime import HMM_AVAILABLE, XGB_AVAILABLE, SKLEARN_AVAILABLE
    except ImportError:
        HMM_AVAILABLE = False
        XGB_AVAILABLE = False
        SKLEARN_AVAILABLE = False

    if not HMM_AVAILABLE or not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("\n" + "="*80)
        print("WARNING: Mock example requires 'hmmlearn', 'xgboost', and 'scikit-learn'.")
        print("Please install them to run the full example.")
        print("pip install hmmlearn xgboost scikit-learn")
        print("="*80)
    else:
        print("Running Regime-Based Strategy Discovery Example...")
        print("="*80)
        
        mock_df = create_mock_indicator_data(length=1000)

        print("\n[PHASE 1: Running Advanced Regime Detection]")
        regime_detector = AdvancedRegimeDetectionSystem(n_regimes=9)  # Reduced to 9
        
        df_with_regimes = regime_detector.detect_advanced_market_regimes(mock_df)
        
        print("\n[PHASE 2: Running Strategy Discovery]")
        try:
            strategy_discoverer = RegimeStrategyDiscovery(regime_detector)
            strategy_playbook = strategy_discoverer.discover_strategies()
            strategy_discoverer.print_repository_summary()
            
            print("✅ Example complete.")

        except ValueError as e:
            print(f"\nExample failed: {e}")
            print("This can happen with mock data if HMM fails to converge.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()