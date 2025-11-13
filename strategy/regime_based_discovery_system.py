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

# We depend on the structures from the advanced regime system
# Make sure analysis_advanced_regime.py is in the same directory
try:
    from analysis_advanced_regime import AdvancedRegimeDetectionSystem, RegimeState
except ImportError:
    print("Error: Could not import 'AdvancedRegimeDetectionSystem'.")
    print("Please ensure 'analysis_advanced_regime.py' is in the same Python path.")
    # Define dummy classes for the script to be parsable
    class AdvancedRegimeDetectionSystem: pass
    class RegimeState: pass

# ============================================================================
# SECTION 1: INDICATOR & PATTERN LOGIC
# ============================================================================

class StrategyIndicatorRules:
    """
    A helper class that defines the "logic" for what constitutes a
    bullish, bearish, or ranging confirmation from indicators and
    price action features.
    
    This logic is applied to each *swing row* from the HybridSwingRegistry.
    """
    
    def __init__(self, rsi_bull: float = 55, rsi_bear: float = 45, adx_trend: float = 20):
        """
        Initialize the thresholds for indicators.
        
        Args:
            rsi_bull: RSI level considered bullish
            rsi_bear: RSI level considered bearish
            adx_trend: ADX level considered trending (either bull or bear)
        """
        self.rsi_bull = rsi_bull
        self.rsi_bear = rsi_bear
        self.adx_trend = adx_trend

    def get_bullish_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Find all indicators and patterns in a swing row that confirm a BULLISH bias.
        
        Args:
            row: A single row (pd.Series) from the swing_df
        
        Returns:
            Tuple[List[str], List[str]]: (confirming_indicators, strategy_patterns)
        """
        indicators: List[str] = []
        patterns: List[str] = []

        # Indicator Confirmations
        if 'rsi' in row and row['rsi'] > self.rsi_bull:
            indicators.append(f'RSI > {self.rsi_bull}')
        if 'macd_hist' in row and row['macd_hist'] > 0:
            indicators.append('MACD Hist > 0')
        if 'ppo' in row and row['ppo'] > 0:
            indicators.append('PPO > 0')
        if 'adx' in row and row['adx'] > self.adx_trend:
            indicators.append(f'ADX > {self.adx_trend} (Trending)')
        if 'volume_zscore' in row and row['volume_zscore'] > 0.5:
            indicators.append('Volume Z-Score > 0.5 (Confirming Volume)')
        
        # Price Action & Pattern Confirmations
        if 'structure_type' in row and row['structure_type'] == 'HH':
            patterns.append('Price Action: Higher High (HH)')
        if 'structure_type' in row and row['structure_type'] == 'HL':
            patterns.append('Price Action: Higher Low (HL)')
        if 'structure_break' in row and row['structure_break'] == 1:
            patterns.append('Price Action: Bullish Structure Break')
        if 'recent_bullish_patterns' in row and row['recent_bullish_patterns'] > 0:
            patterns.append('Candlestick: Recent Bullish')
        if 'pullback_depth' in row and row['swing_type'] == 'low' and row['pullback_depth'] > 0:
            patterns.append(f'Pullback {row["pullback_depth"]:.1%} Deep')

        return indicators, patterns

    def get_bearish_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Find all indicators and patterns in a swing row that confirm a BEARISH bias.
        
        Args:
            row: A single row (pd.Series) from the swing_df
        
        Returns:
            Tuple[List[str], List[str]]: (confirming_indicators, strategy_patterns)
        """
        indicators: List[str] = []
        patterns: List[str] = []

        # Indicator Confirmations
        if 'rsi' in row and row['rsi'] < self.rsi_bear:
            indicators.append(f'RSI < {self.rsi_bear}')
        if 'macd_hist' in row and row['macd_hist'] < 0:
            indicators.append('MACD Hist < 0')
        if 'ppo' in row and row['ppo'] < 0:
            indicators.append('PPO < 0')
        if 'adx' in row and row['adx'] > self.adx_trend:
            indicators.append(f'ADX > {self.adx_trend} (Trending)')
        if 'volume_zscore' in row and row['volume_zscore'] > 0.5:
            indicators.append('Volume Z-Score > 0.5 (Confirming Volume)')
            
        # Price Action & Pattern Confirmations
        if 'structure_type' in row and row['structure_type'] == 'LH':
            patterns.append('Price Action: Lower High (LH)')
        if 'structure_type' in row and row['structure_type'] == 'LL':
            patterns.append('Price Action: Lower Low (LL)')
        if 'structure_break' in row and row['structure_break'] == 1:
            patterns.append('Price Action: Bearish Structure Break')
        if 'recent_bearish_patterns' in row and row['recent_bearish_patterns'] > 0:
            patterns.append('Candlestick: Recent Bearish')

        return indicators, patterns

    def get_ranging_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Find all indicators and patterns in a swing row that confirm a RANGING bias.
        
        Args:
            row: A single row (pd.Series) from the swing_df
        
        Returns:
            Tuple[List[str], List[str]]: (confirming_indicators, strategy_patterns)
        """
        indicators: List[str] = []
        patterns: List[str] = []

        # Indicator Confirmations
        if 'rsi' in row and self.rsi_bear <= row['rsi'] <= self.rsi_bull:
            indicators.append(f'RSI Neutral ({self.rsi_bear}-{self.rsi_bull})')
        if 'adx' in row and row['adx'] < self.adx_trend:
            indicators.append(f'ADX < {self.adx_trend} (Ranging)')
        if 'bb_width' in row:
            # Note: We can't compare to mean here, but a signal finder
            # could check if the value is "low" relative to recent history
            indicators.append(f'BB Width: {row["bb_width"]:.4f}')
        if 'volume_zscore' in row and row['volume_zscore'] < -0.5:
            indicators.append('Volume Z-Score < -0.5 (Low Volume)')

        # Price Action & Pattern Confirmations
        if 'recent_doji' in row and row['recent_doji'] > 0:
            patterns.append('Candlestick: Recent Doji (Indecision)')
        if 'structure_type' in row and row['structure_type'] in ['LH', 'HL']:
            # A lower-high in a bull or higher-low in a bear can signify ranging
            patterns.append('Price Action: Failed Breakout (LH/HL)')

        return indicators, patterns


# ============================================================================
# SECTION 2: DISCOVERY SYSTEM
# ============================================================================

class RegimeStrategyDiscovery:
    """
    This class builds the "Repository of Strategies" by analyzing the
    output of an AdvancedRegimeDetectionSystem.
    
    It iterates through every swing, identifies its regime, and uses
    StrategyIndicatorRules to find all confirming evidence for that regime.
    """
    
    def __init__(self, regime_system: AdvancedRegimeDetectionSystem):
        """
        Initialize the discovery system.
        
        Args:
            regime_system: A *pre-run* instance of AdvancedRegimeDetectionSystem
                           that has already detected regimes.
        """
        if not regime_system.hmm_classifier or not regime_system.hmm_classifier.regime_states:
            raise ValueError("The AdvancedRegimeDetectionSystem has not been run or failed to train.")
        
        self.regime_system = regime_system
        self.regime_states_map: Dict[int, RegimeState] = regime_system.hmm_classifier.regime_states
        self.indicator_rules = StrategyIndicatorRules()
        
        # This is the "Repository of Strategies Discovered by Regime Status"
        self.strategy_repository: Dict[str, Dict] = {}
        print("✅ RegimeStrategyDiscovery initialized.")

    def discover_strategies(self) -> Dict[str, Dict]:
        """
        Analyzes the HybridSwingRegistry and builds the strategy repository.
        
        This is the main function to run.
        
        Returns:
            Dict[str, Dict]: The completed strategy repository.
        """
        print("  Discovering strategy playbook...")
        
        # 1. Get the swing data (which includes HMM labels)
        swing_df = self.regime_system.registry.to_dataframe()
        
        if 'hmm_regime' not in swing_df.columns:
            print("  ⚠️ 'hmm_regime' column not found in swing registry. Run HMM first.")
            return {}
        
        # 2. Initialize the repository structure from the HMM states
        # We use sets to store unique values
        temp_repository: Dict[str, Dict[str, Set]] = {}
        for regime_id, state in self.regime_states_map.items():
            temp_repository[state.name] = {
                'confirming_indicators': set(),
                'strategy_patterns': set()
            }

        # 3. Iterate over every swing and find confirming evidence
        for _, row in swing_df.iterrows():
            regime_id = row['hmm_regime']
            
            # Skip swings that couldn't be classified
            if regime_id is None or pd.isna(regime_id):
                continue
                
            regime_id = int(regime_id)
            regime_state = self.regime_states_map.get(regime_id)
            
            if not regime_state:
                continue
            
            repo_entry = temp_repository[regime_state.name]
            
            # 4. Use the rules to find confirmations
            indicators: List[str] = []
            patterns: List[str] = []
            
            if regime_state.trend_direction == 'bull':
                indicators, patterns = self.indicator_rules.get_bullish_confirmations(row)
            elif regime_state.trend_direction == 'bear':
                indicators, patterns = self.indicator_rules.get_bearish_confirmations(row)
            elif regime_state.trend_direction == 'neutral':
                indicators, patterns = self.indicator_rules.get_ranging_confirmations(row)
            
            # Add new findings to the sets
            repo_entry['confirming_indicators'].update(indicators)
            repo_entry['strategy_patterns'].update(patterns)

        # 5. Build the final, clean repository
        self.strategy_repository = {}
        for regime_id, state in self.regime_states_map.items():
            temp_entry = temp_repository[state.name]
            self.strategy_repository[state.name] = {
                'regime_id': regime_id,
                'trend_direction': state.trend_direction,
                'volatility_level': state.volatility_level,
                # Convert sets to sorted lists for clean, deterministic output
                'confirming_indicators': sorted(list(temp_entry['confirming_indicators'])),
                'strategy_patterns': sorted(list(temp_entry['strategy_patterns']))
            }
        
        print("  ✅ Strategy playbook discovery complete.")
        return self.strategy_repository

    def get_repository(self) -> Dict[str, Dict]:
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
        
        for regime_name, data in self.strategy_repository.items():
            print(f"\n--- {regime_name.upper()} (ID: {data['regime_id']}) ---")
            print(f"    Type: {data['trend_direction']} trend, {data['volatility_level']} volatility")
            
            print("\n    CONFIRMING INDICATORS:")
            if data['confirming_indicators']:
                for ind in data['confirming_indicators']:
                    print(f"      - {ind}")
            else:
                print("      (None found)")
                
            print("\n    STRATEGY & PRICE ACTION PATTERNS:")
            if data['strategy_patterns']:
                for pat in data['strategy_patterns']:
                    print(f"      - {pat}")
            else:
                print("      (None found)")
        
        print("\n" + "="*80)

# ============================================================================
# SECTION 3: USAGE EXAMPLE
# ============================================================================

def create_mock_indicator_data(length: int) -> pd.DataFrame:
    """
    Creates a mock DataFrame with all the indicators that
    AdvancedRegimeDetectionSystem expects to find.
    """
    print(f"  Generating {length} bars of mock indicator data...")
    dates = pd.date_range(start='2023-01-01', periods=length, freq='h')
    df = pd.DataFrame(index=dates)
    
    # Base prices
    df['close'] = 100 + np.cumsum(np.random.randn(length) * 0.5)
    df['high'] = df['close'] + np.random.rand(length) * 2
    df['low'] = df['close'] - np.random.rand(length) * 2
    df['open'] = df['low'] + (df['high'] - df['low']) * np.random.rand(length)
    df['volume'] = np.random.randint(100, 1000, size=length)
    
    # Indicators
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
    
    # Price Action / Patterns (mocked)
    df['pullback_from_high_pct'] = np.random.rand(length) * 0.5
    df['structure_break_bullish'] = (np.random.rand(length) > 0.9).astype(int)
    df['structure_break_bearish'] = (np.random.rand(length) > 0.9).astype(int)
    df['pattern_hammer'] = (np.random.rand(length) > 0.95).astype(int)
    df['pattern_bearish_engulfing'] = (np.random.rand(length) > 0.95).astype(int)
    df['pattern_doji'] = (np.random.rand(length) > 0.9).astype(int)
    
    # Fill NaNs
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

if __name__ == "__main__":
    
    # This check is necessary to use the mock example
    if not HMM_AVAILABLE or not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
        print("\n" + "="*80)
        print("WARNING: Mock example requires 'hmmlearn', 'xgboost', and 'scikit-learn'.")
        print("Please install them to run the full example.")
        print("pip install hmmlearn xgboost scikit-learn")
        print("="*80)
    
    else:
        print("Running Regime-Based Strategy Discovery Example...")
        print("="*80)
        
        # 1. Create mock data
        # In a real system, you would load your historical data and
        # run your full indicator pipeline here.
        mock_df = create_mock_indicator_data(length=1000)

        # 2. Run the Regime Detection System
        # This is the prerequisite. It must be run first.
        print("\n[PHASE 1: Running Advanced Regime Detection]")
        regime_detector = AdvancedRegimeDetectionSystem(n_regimes=6)
        
        # This function runs ZigZag, builds the registry, trains HMM,
        # and trains XGBoost all in one go.
        df_with_regimes = regime_detector.detect_advanced_market_regimes(mock_df)
        
        print("\n[PHASE 2: Running Strategy Discovery]")
        # 3. Initialize the Discovery System with the *trained* detector
        try:
            strategy_discoverer = RegimeStrategyDiscovery(regime_detector)
            
            # 4. Run the discovery process
            strategy_playbook = strategy_discoverer.discover_strategies()
            
            # 5. Print the results
            strategy_discoverer.print_repository_summary()
            
            # You can also get the dict directly
            # print("\nFull Repository (JSON format):")
            # print(json.dumps(strategy_playbook, indent=2))
            
            print("✅ Example complete.")

        except ValueError as e:
            print(f"\nExample failed: {e}")
            print("This can happen with mock data if HMM fails to converge.")
            print("Try re-running the script.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")