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
    from analysis_advanced_regime import AdvancedRegimeDetectionSystem, RegimeState
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
    Defines logic for bullish, bearish, or ranging confirmations from indicators
    and price action features.
    """
    
    def __init__(self, rsi_bull: float = 55, rsi_bear: float = 45, adx_trend: float = 20):
        self.rsi_bull = rsi_bull
        self.rsi_bear = rsi_bear
        self.adx_trend = adx_trend

    def get_bullish_confirmations(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        indicators: List[str] = []
        patterns: List[str] = []

        if 'rsi' in row and row['rsi'] > self.rsi_bull:
            indicators.append(f'RSI > {self.rsi_bull}')
        if 'macd_hist' in row and row['macd_hist'] > 0:
            indicators.append('MACD Hist > 0')
        if 'ppo' in row and row['ppo'] > 0:
            indicators.append('PPO > 0')
        if 'adx' in row and row['adx'] > self.adx_trend:
            indicators.append(f'ADX > {self.adx_trend} (Trending)')
        if 'volume_zscore' in row and row['volume_zscore'] > 0.5:
            indicators.append('Volume Z-Score > 0.5')
        
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
        indicators: List[str] = []
        patterns: List[str] = []

        if 'rsi' in row and row['rsi'] < self.rsi_bear:
            indicators.append(f'RSI < {self.rsi_bear}')
        if 'macd_hist' in row and row['macd_hist'] < 0:
            indicators.append('MACD Hist < 0')
        if 'ppo' in row and row['ppo'] < 0:
            indicators.append('PPO < 0')
        if 'adx' in row and row['adx'] > self.adx_trend:
            indicators.append(f'ADX > {self.adx_trend} (Trending)')
        if 'volume_zscore' in row and row['volume_zscore'] > 0.5:
            indicators.append('Volume Z-Score > 0.5')
            
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
        indicators: List[str] = []
        patterns: List[str] = []

        if 'rsi' in row and self.rsi_bear <= row['rsi'] <= self.rsi_bull:
            indicators.append(f'RSI Neutral ({self.rsi_bear}-{self.rsi_bull})')
        if 'adx' in row and row['adx'] < self.adx_trend:
            indicators.append(f'ADX < {self.adx_trend} (Ranging)')
        if 'bb_width' in row:
            indicators.append(f'BB Width: {row["bb_width"]:.4f}')
        if 'volume_zscore' in row and row['volume_zscore'] < -0.5:
            indicators.append('Volume Z-Score < -0.5 (Low Volume)')

        if 'recent_doji' in row and row['recent_doji'] > 0:
            patterns.append('Candlestick: Recent Doji (Indecision)')
        if 'structure_type' in row and row['structure_type'] in ['LH', 'HL']:
            patterns.append('Price Action: Failed Breakout (LH/HL)')

        return indicators, patterns


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
        print("✅ RegimeStrategyDiscovery initialized.")

    def discover_strategies(self) -> Dict[str, Dict]:
        """
        Analyzes the HybridSwingRegistry and builds the strategy repository per *regime instance*.
        Returns:
            Dict[str, Dict]: mapping regime_instance_id -> playbook metadata
        """
        print("  Discovering strategy playbook (instance-level)...")
        
        swing_df = self.regime_system.registry.to_dataframe()
        
        if 'hmm_regime' not in swing_df.columns:
            print("  ⚠️ 'hmm_regime' column not found in swing registry. Run HMM first.")
            return {}
        
        # Prefer instance-level key if available
        if 'regime_instance_id' in swing_df.columns:
            keys = swing_df['regime_instance_id'].fillna('UNKN').unique().tolist()
        else:
            # fallback to regime ids as strings
            keys = swing_df['hmm_regime'].fillna(-1).unique().astype(int).tolist()
            keys = [f"R{k}" for k in keys]

        # initialize temp repository per instance
        temp_repository: Dict[str, Dict[str, Set]] = {}
        # also store mapping to regime type metadata
        instance_meta: Dict[str, Dict] = {}
        
        # Build initial entries based on present instances
        if 'regime_instance_id' in swing_df.columns:
            for inst in keys:
                temp_repository[inst] = {
                    'confirming_indicators': set(),
                    'strategy_patterns': set()
                }
        else:
            for inst in keys:
                temp_repository[inst] = {
                    'confirming_indicators': set(),
                    'strategy_patterns': set()
                }

        # Iterate swings and assign to instance buckets
        for _, row in swing_df.iterrows():
            inst_key = None
            if 'regime_instance_id' in swing_df.columns and pd.notna(row.get('regime_instance_id')):
                inst_key = row['regime_instance_id']
            elif pd.notna(row.get('hmm_regime')):
                inst_key = f"R{int(row['hmm_regime'])}"
            else:
                continue

            # ensure present
            if inst_key not in temp_repository:
                temp_repository[inst_key] = {
                    'confirming_indicators': set(),
                    'strategy_patterns': set()
                }

            regime_id = None
            if 'hmm_regime' in row and pd.notna(row['hmm_regime']):
                regime_id = int(row['hmm_regime'])
            regime_state = self.regime_states_map.get(regime_id) if regime_id is not None else None

            # choose confirmation logic based on regime_state trend; if missing, use heuristics
            indicators, patterns = [], []
            if regime_state:
                if regime_state.trend_direction == 'bull':
                    indicators, patterns = self.indicator_rules.get_bullish_confirmations(row)
                elif regime_state.trend_direction == 'bear':
                    indicators, patterns = self.indicator_rules.get_bearish_confirmations(row)
                else:
                    indicators, patterns = self.indicator_rules.get_ranging_confirmations(row)
            else:
                # fallback: use neutral thresholds
                indicators, patterns = self.indicator_rules.get_ranging_confirmations(row)

            temp_repository[inst_key]['confirming_indicators'].update(indicators)
            temp_repository[inst_key]['strategy_patterns'].update(patterns)

            # store instance meta if not already
            if inst_key not in instance_meta:
                instance_meta[inst_key] = {
                    'regime_type': regime_id,
                    'regime_name': (self.regime_states_map.get(regime_id).name if regime_id in self.regime_states_map else f"R{regime_id}"),
                    'trend_direction': (self.regime_states_map.get(regime_id).trend_direction if regime_id in self.regime_states_map else 'unknown'),
                    'volatility_level': (self.regime_states_map.get(regime_id).volatility_level if regime_id in self.regime_states_map else 'unknown'),
                }

        # Build final repository keyed by instance_id
        self.strategy_repository = {}
        for inst_key, sets in temp_repository.items():
            meta = instance_meta.get(inst_key, {})
            self.strategy_repository[inst_key] = {
                'regime_instance_id': inst_key,
                'regime_type': meta.get('regime_type'),
                'regime_name': meta.get('regime_name'),
                'trend_direction': meta.get('trend_direction', 'unknown'),
                'volatility_level': meta.get('volatility_level', 'unknown'),
                'confirming_indicators': sorted(list(sets['confirming_indicators'])),
                'strategy_patterns': sorted(list(sets['strategy_patterns']))
            }

        print("  ✅ Strategy playbook discovery (instance-level) complete.")
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
            print(f"\n--- REGIME {regime_id}: {data['regime_name'].upper()} ---")
            print(f"    Type: {data['trend_direction']} trend, {data['volatility_level']} volatility")
            
            print("\n    CONFIRMING INDICATORS:")
            if data['confirming_indicators']:
                for ind in data['confirming_indicators']:
                    print(f"      • {ind}")
            else:
                print("      (None found)")
                
            print("\n    STRATEGY & PRICE ACTION PATTERNS:")
            if data['strategy_patterns']:
                for pat in data['strategy_patterns']:
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
        from analysis_advanced_regime import HMM_AVAILABLE, XGB_AVAILABLE, SKLEARN_AVAILABLE
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