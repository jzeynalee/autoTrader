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
from .reporting import ReportsMixin
from .patterns import PatternsMixin

try:
    from .discovery_mapping import (
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

class DiscoveryModesMixin(ReportsMixin, PatternsMixin):
    def _analyze_phase5_strategy_breakdown(self, strategies):
        """Analyze and print detailed Phase 5 strategy breakdown"""
        confluence_counts = 0
        structure_aligned_counts = 0
        alignment_scores = []
        confluence_scores = []
        
        for strategy in strategies:
            strategy_class = strategy.get('strategy_class', '')
            
            if 'confluence' in strategy_class:
                confluence_counts += 1
                confluence_scores.append(strategy.get('confluence_score', 0))
            elif 'structure_aligned' in strategy_class:
                structure_aligned_counts += 1
                alignment_scores.append(strategy.get('structure_alignment_score', 0))
        
        print(f"\n🎯 PHASE 5 STRATEGY DISCOVERY SUMMARY")
        print(f"   Total Strategies: {len(strategies)}")
        print(f"   Confluence Strategies: {confluence_counts}")
        print(f"   Structure-Aligned Strategies: {structure_aligned_counts}")
        
        if confluence_scores:
            avg_confluence = sum(confluence_scores) / len(confluence_scores)
            print(f"   Average Confluence Score: {avg_confluence:.3f}")
        
        if alignment_scores:
            avg_alignment = sum(alignment_scores) / len(alignment_scores)
            print(f"   Average Alignment Score: {avg_alignment:.3f}")
        
        # Print performance by strategy class
        class_performance = {}
        for strategy in strategies:
            strategy_class = strategy.get('strategy_class', 'unknown')
            accuracy = strategy.get('discovered_accuracy', 0)
            
            if strategy_class not in class_performance:
                class_performance[strategy_class] = []
            class_performance[strategy_class].append(accuracy)
        
        print(f"\n📊 AVERAGE PERFORMANCE BY STRATEGY CLASS:")
        for strategy_class, accuracies in class_performance.items():
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                print(f"   {strategy_class}: {avg_accuracy:.2%} ({len(accuracies)} strategies)")

    def discover_mtf_strategies_mode_a(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode A (Refactored): Strict hierarchical MTF cascade (HTF & TTF & LTF)
        Corrected for lookahead bias.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode A strategies for {group_name} (HTF & TTF & LTF cascade)...")

        if htf_df is None or ttf_df is None or ltf_df is None:
            return []

        strategies = []

        # --- Categorize indicators for each timeframe ---
        max_inds = getattr(self, "max_indicators", 6)

        # Fetch all signals (indicators + patterns)
        htf_all = self.categorize_columns(htf_df)
        htf_signals = (htf_all['indicators'] + htf_all['candlestick_patterns'] + htf_all['chart_patterns'])[:max_inds]
        
        ttf_all = self.categorize_columns(ttf_df)
        ttf_signals = (ttf_all['indicators'] + ttf_all['candlestick_patterns'] + ttf_all['chart_patterns'])[:max_inds]
        
        ltf_all = self.categorize_columns(ltf_df)
        ltf_signals = (ltf_all['indicators'] + ltf_all['candlestick_patterns'] + ltf_all['chart_patterns'])[:max_inds]
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        # Use the combined lists for the loops
        for htf_ind in htf_signals:
            htf_states = self.get_or_compute_states(htf_df, htf_ind, htf_pair_tf)
            if htf_states is None: continue

            for ttf_ind in ttf_signals:
                ttf_states = self.get_or_compute_states(ttf_df, ttf_ind, ttf_pair_tf)
                if ttf_states is None: continue

                for ltf_ind in ltf_signals:
                    ltf_states = self.get_or_compute_states(ltf_df, ltf_ind, ltf_pair_tf)
                    if ltf_states is None: continue
                    
                    # Bullish: HTF AND TTF AND LTF must all be bullish
                    bullish_mask = (htf_states == 'bullish') & (ttf_states == 'bullish') & (ltf_states == 'bullish')
                    
                    # Bearish: HTF AND TTF AND LTF must all be bearish
                    bearish_mask = (htf_states == 'bearish') & (ttf_states == 'bearish') & (ltf_states == 'bearish')

                    # --- Evaluate Bullish ---
                    if bullish_mask.sum() > 10: # Min sample size
                        aligned_returns = ltf_df.loc[bullish_mask, 'future_return']
                        win_rate = (aligned_returns > 0).mean()
                        if win_rate > 0.51:
                            strategies.append({
                                'type': 'mtf_mode_a',
                                'signal_type': 'MTF_COMPOSITE',
                                'group': group_name,
                                'pair_tf': f"{pair}_{ltf_tf}",
                                'direction': 'bullish',
                                'trade_direction': 'long',
                                'htf_indicator': htf_ind,
                                'ttf_indicator': ttf_ind,
                                'ltf_indicator': ltf_ind,
                                'htf_timeframe': htf_tf,
                                'ttf_timeframe': ttf_tf,
                                'ltf_timeframe': ltf_tf,
                                'discovered_accuracy': win_rate,
                                'sample_size': int(bullish_mask.sum()),
                                'performance_score': win_rate
                            })

                    # --- Evaluate Bearish ---
                    if bearish_mask.sum() > 10: # Min sample size
                        aligned_returns = ltf_df.loc[bearish_mask, 'future_return']
                        win_rate = (aligned_returns < 0).mean()
                        if win_rate > 0.51:
                            strategies.append({
                                'type': 'mtf_mode_a',
                                "signal_type": "MTF_COMPOSITE",
                                'group': group_name,
                                'pair_tf': f"{pair}_{ltf_tf}",
                                'direction': 'bearish',
                                'trade_direction': 'short',
                                'htf_indicator': htf_ind,
                                'ttf_indicator': ttf_ind,
                                'ltf_indicator': ltf_ind,
                                'htf_timeframe': htf_tf,
                                'ttf_timeframe': ttf_tf,
                                'ltf_timeframe': ltf_tf,
                                'discovered_accuracy': win_rate,
                                'sample_size': int(bearish_mask.sum()),
                                'performance_score': win_rate
                            })

        return strategies


    def discover_mtf_strategies_mode_b(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode B (Corrected): Flexible MTF Confluence (2-of-3)
        Corrected for lookahead bias.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode B strategies for {group_name} (2-of-3 confluence)...")

        if any(df is None for df in (htf_df, ttf_df, ltf_df)):
            return []

        strategies = []

        max_inds = getattr(self, "max_indicators", 6)
        # Fetch all signals (indicators + patterns)
        htf_all = self.categorize_columns(htf_df)
        htf_signals = (htf_all['indicators'] + htf_all['candlestick_patterns'] + htf_all['chart_patterns'])[:max_inds]
        
        ttf_all = self.categorize_columns(ttf_df)
        ttf_signals = (ttf_all['indicators'] + ttf_all['candlestick_patterns'] + ttf_all['chart_patterns'])[:max_inds]
        
        ltf_all = self.categorize_columns(ltf_df)
        ltf_signals = (ltf_all['indicators'] + ltf_all['candlestick_patterns'] + ltf_all['chart_patterns'])[:max_inds]
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        # Use the combined lists for the loops
        for htf_ind in htf_signals:
            htf_states = self.get_or_compute_states(htf_df, htf_ind, htf_pair_tf)
            if htf_states is None: continue

            for ttf_ind in ttf_signals:
                ttf_states = self.get_or_compute_states(ttf_df, ttf_ind, ttf_pair_tf)
                if ttf_states is None: continue

                for ltf_ind in ltf_signals:
                    ltf_states = self.get_or_compute_states(ltf_df, ltf_ind, ltf_pair_tf)
                    if ltf_states is None: continue

                    # === Correct Point-in-Time Logic ===
                    htf_bull = (htf_states == 'bullish')
                    ttf_bull = (ttf_states == 'bullish')
                    ltf_bull = (ltf_states == 'bullish')

                    htf_bear = (htf_states == 'bearish')
                    ttf_bear = (ttf_states == 'bearish')
                    ltf_bear = (ltf_states == 'bearish')

                    # 2-of-3 confluence masks
                    bullish_masks = {
                        "HTF+TTF": htf_bull & ttf_bull,
                        "HTF+LTF": htf_bull & ltf_bull,
                        "TTF+LTF": ttf_bull & ltf_bull,
                    }
                    bearish_masks = {
                        "HTF+TTF": htf_bear & ttf_bear,
                        "HTF+LTF": htf_bear & ltf_bear,
                        "TTF+LTF": ttf_bear & ltf_bear,
                    }

                    for combo, mask in bullish_masks.items():
                        if mask.sum() > 10:
                            ret = ltf_df.loc[mask, "future_return"]
                            win_rate = (ret > 0).mean()
                            if win_rate > 0.51:
                                strategies.append({
                                    "type": "mtf_mode_b",
                                    "signal_type": "MTF_COMPOSITE",
                                    "group": group_name,
                                    "pair_tf": f"{pair}_{ltf_tf}", 
                                    "direction": "bullish",
                                    "trade_direction": "long", 
                                    "confluence_type": combo,
                                    "htf_indicator": htf_ind, 
                                    "ttf_indicator": ttf_ind,
                                    "ltf_indicator": ltf_ind, 
                                    "htf_timeframe": htf_tf,
                                    "ttf_timeframe": ttf_tf, 
                                    "ltf_timeframe": ltf_tf,
                                    "discovered_accuracy": win_rate, 
                                    "sample_size": int(mask.sum()),
                                    "performance_score": win_rate * 0.9,
                                })

                    for combo, mask in bearish_masks.items():
                        if mask.sum() > 25:
                            ret = ltf_df.loc[mask, "future_return"]
                            win_rate = (ret < 0).mean()
                            if win_rate > 0.51:
                                strategies.append({
                                    "type": "mtf_mode_b", 
                                    "signal_type": "MTF_COMPOSITE",
                                    "group": group_name,
                                    "pair_tf": f"{pair}_{ltf_tf}", "direction": "bearish",
                                    "trade_direction": "short", "confluence_type": combo,
                                    "htf_indicator": htf_ind, "ttf_indicator": ttf_ind,
                                    "ltf_indicator": ltf_ind, "htf_timeframe": htf_tf,
                                    "ttf_timeframe": ttf_tf, "ltf_timeframe": ltf_tf,
                                    "discovered_accuracy": win_rate, "sample_size": int(mask.sum()),
                                    "performance_score": win_rate * 0.9,
                                })
        return strategies

    def discover_mtf_strategies_mode_c(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode C (Refactored): Weighted MTF Scoring
        Corrected for lookahead bias.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode C strategies for {group_name} (weighted cascade)...")

        if any(df is None for df in (htf_df, ttf_df, ltf_df)):
            return []

        strategies = []

        max_inds = getattr(self, "max_indicators", 6)
        '''htf_inds = self.categorize_columns(htf_df)["indicators"][:max_inds]
        ttf_inds = self.categorize_columns(ttf_df)["indicators"][:max_inds]
        ltf_inds = self.categorize_columns(ltf_df)["indicators"][:max_inds]'''

        # Fetch all signals (indicators + patterns)
        htf_all = self.categorize_columns(htf_df)
        htf_signals = (htf_all['indicators'] + htf_all['candlestick_patterns'] + htf_all['chart_patterns'])[:max_inds]
        
        ttf_all = self.categorize_columns(ttf_df)
        ttf_signals = (ttf_all['indicators'] + ttf_all['candlestick_patterns'] + ttf_all['chart_patterns'])[:max_inds]
        
        ltf_all = self.categorize_columns(ltf_df)
        ltf_signals = (ltf_all['indicators'] + ltf_all['candlestick_patterns'] + ltf_all['chart_patterns'])[:max_inds]
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        # Use the combined lists for the loops
        for htf_ind in htf_signals:
            htf_states = self.get_or_compute_states(htf_df, htf_ind, htf_pair_tf)
            if htf_states is None: continue

            for ttf_ind in ttf_signals:
                ttf_states = self.get_or_compute_states(ttf_df, ttf_ind, ttf_pair_tf)
                if ttf_states is None: continue

                for ltf_ind in ltf_signals:
                    ltf_states = self.get_or_compute_states(ltf_df, ltf_ind, ltf_pair_tf)
                    if ltf_states is None: continue

                    # === Correct Point-in-Time Logic ===
                    htf_scores = htf_states.map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)
                    ttf_scores = ttf_states.map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)
                    ltf_scores = ltf_states.map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)

                    weighted_score = (
                        htf_scores * 0.5 +
                        ttf_scores * 0.3 +
                        ltf_scores * 0.2
                    )

                    for threshold in [0.6, 0.7, 0.8]:
                        bullish_mask = weighted_score >= threshold
                        bearish_mask = weighted_score <= -threshold

                        if bullish_mask.sum() > 12:
                            aligned = ltf_df.loc[bullish_mask, "future_return"]
                            win_rate = (aligned > 0).mean()
                            if win_rate > 0.51:
                                strategies.append({
                                    "type": "mtf_mode_c", 
                                    "signal_type": "MTF_COMPOSITE",
                                    "group": group_name,
                                    "pair_tf": f"{pair}_{ltf_tf}", "direction": "bullish",
                                    "trade_direction": "long", "htf_indicator": htf_ind,
                                    "ttf_indicator": ttf_ind, "ltf_indicator": ltf_ind,
                                    "score_threshold": threshold, "htf_timeframe": htf_tf,
                                    "ttf_timeframe": ttf_tf, "ltf_timeframe": ltf_tf,
                                    "discovered_accuracy": win_rate, "sample_size": int(bullish_mask.sum()),
                                    "performance_score": win_rate * (1 + threshold),
                                })

                        if bearish_mask.sum() > 12:
                            aligned = ltf_df.loc[bearish_mask, "future_return"]
                            win_rate = (aligned < 0).mean()
                            if win_rate > 0.51:
                                strategies.append({
                                    "type": "mtf_mode_c", 
                                    "signal_type": "MTF_COMPOSITE",
                                    "group": group_name,
                                    "pair_tf": f"{pair}_{ltf_tf}", "direction": "bearish",
                                    "trade_direction": "short", "htf_indicator": htf_ind,
                                    "ttf_indicator": ttf_ind, "ltf_indicator": ltf_ind,
                                    "score_threshold": threshold, "htf_timeframe": htf_tf,
                                    "ttf_timeframe": ttf_tf, "ltf_timeframe": ltf_tf,
                                    "discovered_accuracy": win_rate, "sample_size": int(bearish_mask.sum()),
                                    "performance_score": win_rate * (1 + threshold),
                                })
        return strategies

    def discover_mtf_strategies_mode_d(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode D(Refactored): Pure Price Action MTF Strategy
        Uses only swing points, pullbacks, and market structure - NO traditional indicators
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode D (Price Action) strategies for {group_name}...")
        
        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # Define price action signal combinations
        price_action_signals = {
            # Bullish price action setups
            'bullish_setup_1': {
                'htf': ['higher_lows_pattern', 'trend_medium'],
                'ttf': ['pullback_complete_bull', 'healthy_bull_pullback'], 
                'ltf': ['swing_low', 'near_fib_618']
            },
            'bullish_setup_2': {
                'htf': ['trend_medium', 'abc_pullback_bull'],
                'ttf': ['pullback_stage_resumption_bull', 'volume_decreasing'],
                'ltf': ['swing_low', 'sr_confluence_score']
            },
            
            # Bearish price action setups  
            'bearish_setup_1': {
                'htf': ['lower_highs_pattern', 'trend_medium'],
                'ttf': ['pullback_complete_bear', 'healthy_bear_pullback'],
                'ltf': ['swing_high', 'near_fib_382']
            },
            'bearish_setup_2': {
                'htf': ['trend_medium', 'abc_pullback_bear'],
                'ttf': ['pullback_stage_resumption_bear', 'volume_decreasing'],
                'ltf': ['swing_high', 'sr_confluence_score']
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        for setup_name, signals in price_action_signals.items():
            direction = 'bullish' if 'bullish' in setup_name else 'bearish'
            
            # Get states for all signals in this setup
            htf_states = {}
            for sig in signals['htf']:
                if sig in htf_df.columns:
                    htf_states[sig] = self.get_or_compute_states(htf_df, sig, htf_pair_tf)
            
            ttf_states = {}
            for sig in signals['ttf']:
                if sig in ttf_df.columns:
                    ttf_states[sig] = self.get_or_compute_states(ttf_df, sig, ttf_pair_tf)
                    
            ltf_states = {}
            for sig in signals['ltf']:
                if sig in ltf_df.columns:
                    ltf_states[sig] = self.get_or_compute_states(ltf_df, sig, ltf_pair_tf)
            
            # Create combined mask (all signals must align)
            if htf_states and ttf_states and ltf_states:
                combined_mask = pd.Series(True, index=ltf_df.index)
                
                # HTF signals must be bullish for bullish setup, etc.
                for sig, states in htf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            combined_mask &= (states == 'bullish')
                        else:
                            combined_mask &= (states == 'bearish')
                
                for sig, states in ttf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            combined_mask &= (states == 'bullish')
                        else:
                            combined_mask &= (states == 'bearish')
                            
                for sig, states in ltf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            combined_mask &= (states == 'bullish')
                        else:
                            combined_mask &= (states == 'bearish')
                
                if combined_mask.sum() > 10:
                    aligned_returns = ltf_df.loc[combined_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    if win_rate > 0.51:
                        strategies.append({
                            'type': 'mtf_mode_d', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'setup_name': setup_name,
                            'htf_signals': signals['htf'],
                            'ttf_signals': signals['ttf'],
                            'ltf_signals': signals['ltf'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(combined_mask.sum()),
                            'performance_score': win_rate,
                            'strategy_class': 'pure_price_action'
                        })
        
        return strategies

    def discover_mtf_strategies_mode_e(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode E(Refactored): Hybrid Price Action + Indicator Strategy
        Combines the best of price structure with traditional indicators
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode E (Hybrid) strategies for {group_name}...")
        
        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # Define hybrid signal combinations (price action + indicators)
        hybrid_setups = {
            # Bullish hybrid setups
            'hybrid_bull_1': {
                'price_action': {
                    'htf': ['trend_medium', 'higher_lows_pattern'],
                    'ttf': ['pullback_complete_bull'],
                    'ltf': ['swing_low', 'near_fib_618']
                },
                'indicators': {
                    'htf': ['rsi'],
                    'ttf': ['macd'],
                    'ltf': ['bb_pct']
                }
            },
            'hybrid_bull_2': {
                'price_action': {
                    'htf': ['abc_pullback_bull'],
                    'ttf': ['healthy_bull_pullback'], 
                    'ltf': ['sr_confluence_score']
                },
                'indicators': {
                    'htf': ['ema_50'],
                    'ttf': ['stoch_k'],
                    'ltf': ['rsi']
                }
            },
            
            # Bearish hybrid setups
            'hybrid_bear_1': {
                'price_action': {
                    'htf': ['trend_medium', 'lower_highs_pattern'],
                    'ttf': ['pullback_complete_bear'],
                    'ltf': ['swing_high', 'near_fib_382']
                },
                'indicators': {
                    'htf': ['rsi'],
                    'ttf': ['macd'],
                    'ltf': ['bb_pct']
                }
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        for setup_name, setup_config in hybrid_setups.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Check price action signals
            pa_mask = pd.Series(True, index=ltf_df.index)
            
            for tf, signals in setup_config['price_action'].items():
                df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df

                current_pair_tf = htf_pair_tf if tf == 'htf' else (ttf_pair_tf if tf == 'ttf' else ltf_pair_tf)
                
                for sig in signals:
                    if sig in df.columns:
                        states = self.get_or_compute_states(df, sig, current_pair_tf)
                        if states is not None:
                            if direction == 'bullish':
                                pa_mask &= (states == 'bullish')
                            else:
                                pa_mask &= (states == 'bearish')
            
            # Check indicator signals  
            indicator_mask = pd.Series(True, index=ltf_df.index)
            
            for tf, signals in setup_config['indicators'].items():
                df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df

                current_pair_tf = htf_pair_tf if tf == 'htf' else (ttf_pair_tf if tf == 'ttf' else ltf_pair_tf)
                
                for sig in signals:
                    if sig in df.columns:
                        states = self.get_or_compute_states(df, sig, current_pair_tf)
                        if states is not None:
                            if direction == 'bullish':
                                indicator_mask &= (states == 'bullish')
                            else:
                                indicator_mask &= (states == 'bearish')
            
            # Combined mask (both price action AND indicators must agree)
            combined_mask = pa_mask & indicator_mask
            
            if combined_mask.sum() > 8:
                aligned_returns = ltf_df.loc[combined_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate > 0.51:  # Higher threshold for hybrid strategies
                    strategies.append({
                        'type': 'mtf_mode_e', 
                        "signal_type": "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'price_action_signals': setup_config['price_action'],
                        'indicator_signals': setup_config['indicators'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(combined_mask.sum()),
                        'performance_score': win_rate * 1.1,  # Bonus for hybrid approach
                        'strategy_class': 'hybrid_price_action'
                    })
        
        return strategies

    def discover_mtf_strategies_mode_f(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode F(Refactored): Advanced Structure Breakout Strategy
        Focuses on structural breaks with volume and momentum confirmation
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode F (Structure Breakout) strategies for {group_name}...")
        
        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
            
        # ========== DEBUG: CHECK WHAT COLUMNS EXIST ==========
        '''print(f"    [DEBUG Mode F] HTF columns: {list(htf_df.columns)}")
        print(f"    [DEBUG Mode F] TTF columns: {list(ttf_df.columns)}")
        print(f"    [DEBUG Mode F] LTF columns: {list(ltf_df.columns)}")'''
        
        # Check for required pattern columns
        required_patterns = [
            'trend_structure', 'structure_break_bullish', 'structure_break_bearish',
            'false_breakout_bullish', 'false_breakout_bearish', 
            'momentum_continuation', 'volume_breakout_confirmation',
            'higher_highs_lower_lows'
        ]
        
        missing_patterns = []
        for pattern in required_patterns:
            if pattern not in htf_df.columns:
                missing_patterns.append(f"HTF:{pattern}")
            if pattern not in ttf_df.columns:
                missing_patterns.append(f"TTF:{pattern}")
            if pattern not in ltf_df.columns:
                missing_patterns.append(f"LTF:{pattern}")
        
        if missing_patterns:
            print(f"    [DEBUG Mode F] ❌ MISSING PATTERNS: {missing_patterns[:10]}")
            print(f"    [DEBUG Mode F] Total missing: {len(missing_patterns)}")
            return []  # Early return if patterns are missing
        else:
            print(f"    [DEBUG Mode F] ✅ All required patterns found!")

        strategies = []
        
        # Advanced breakout setups
        advanced_setups = {
            # Bullish structural breakouts
            'structural_breakout_bull': {
                'description': 'HTF trend + TTF structure break + LTF volume confirmation',
                'htf': ['trend_structure'],
                'ttf': ['structure_break_bullish'],
                'ltf': ['momentum_continuation']
            },
            'false_breakout_reversal_bull': {
                'description': 'False bearish breakout followed by bullish reversal',
                'htf': ['trend_structure'],
                'ttf': ['false_breakout_bearish'],
                'ltf': ['structure_break_bullish']
            },
            
            # Bearish structural breakouts  
            'structural_breakout_bear': {
                'description': 'HTF downtrend + TTF structure break + LTF volume confirmation',
                'htf': ['trend_structure'],
                'ttf': ['higher_highs_lower_lows'],
                'ltf': ['momentum_continuation']
            },
            'false_breakout_reversal_bear': {
                'description': 'False bullish breakout followed by bearish reversal',
                'htf': ['trend_structure'],
                'ttf': ['false_breakout_bullish'],
                'ltf': ['volume_breakout_confirmation']
            }
        }
        

        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        for setup_name, setup_config in advanced_setups.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Get states for all timeframes
            htf_states = {}
            for sig in setup_config['htf']:
                if sig in htf_df.columns:
                    htf_states[sig] = self.get_or_compute_states(htf_df, sig, htf_pair_tf)
            
            ttf_states = {}
            for sig in setup_config['ttf']:
                if sig in ttf_df.columns:
                    ttf_states[sig] = self.get_or_compute_states(ttf_df, sig, ttf_pair_tf)
                    
            ltf_states = {}
            for sig in setup_config['ltf']:
                if sig in ltf_df.columns:
                    ltf_states[sig] = self.get_or_compute_states(ltf_df, sig, ltf_pair_tf)
            
            # Create advanced confluence mask
            if htf_states and ttf_states and ltf_states:
                advanced_mask = pd.Series(True, index=ltf_df.index)
                
                # HTF must show appropriate trend structure
                for sig, states in htf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            advanced_mask &= (states == 'bullish')
                        else:
                            advanced_mask &= (states == 'bearish')
                
                # TTF must show structural signals
                for sig, states in ttf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            advanced_mask &= (states == 'bullish')
                        else:
                            advanced_mask &= (states == 'bearish')
                            
                # LTF must show confirmation signals
                for sig, states in ltf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            advanced_mask &= (states == 'bullish')
                        else:
                            advanced_mask &= (states == 'bearish')
                
                if advanced_mask.sum() > 5:
                    aligned_returns = ltf_df.loc[advanced_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    if win_rate > 0.51:  # Higher threshold for advanced strategies
                        strategies.append({
                            'type': 'mtf_mode_f', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'setup_name': setup_name,
                            'description': setup_config['description'],
                            'htf_signals': setup_config['htf'],
                            'ttf_signals': setup_config['ttf'],
                            'ltf_signals': setup_config['ltf'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(advanced_mask.sum()),
                            'performance_score': win_rate * 1.2,  # Bonus for advanced patterns
                            'strategy_class': 'advanced_structure_breakout'
                        })
        
        return strategies

    def discover_mtf_strategies_mode_g(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode G(Refactored): Momentum & Divergence Strategy
        Focuses on momentum shifts and divergence patterns across timeframes.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode G (Momentum Divergence) strategies for {group_name}...")

        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        # Momentum and divergence setups
        momentum_setups = {
            # Bullish momentum divergence
            'momentum_divergence_bull': {
                'description': 'Price making lower lows but momentum showing bullish divergence',
                'htf': ['trend_structure'],
                'ttf': ['momentum_divergence_bullish'],
                'ltf': ['false_breakout_bearish']
            },
            'momentum_reversal_bull': {
                'description': 'Momentum shift from bearish to bullish across timeframes',
                'htf': ['trend_structure'],
                'ttf': ['momentum_divergence_bullish'],
                'ltf': ['structure_break_bullish']
            },
            
            # Bearish momentum divergence
            'momentum_divergence_bear': {
                'description': 'Price making higher highs but momentum showing bearish divergence',
                'htf': ['trend_structure'],
                'ttf': ['momentum_divergence_bearish'],
                'ltf': ['false_breakout_bullish']
            },
            'momentum_reversal_bear': {
                'description': 'Momentum shift from bullish to bearish across timeframes',
                'htf': ['trend_structure'],
                'ttf': ['momentum_divergence_bearish'],
                'ltf': ['structure_break_bearish']
            }
        }

        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        for setup_name, setup_config in momentum_setups.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Get states for momentum patterns
            htf_states = {}
            for sig in setup_config['htf']:
                if sig in htf_df.columns:
                    htf_states[sig] = self.get_or_compute_states(htf_df, sig, htf_pair_tf)
            
            ttf_states = {}
            for sig in setup_config['ttf']:
                if sig in ttf_df.columns:
                    ttf_states[sig] = self.get_or_compute_states(ttf_df, sig, ttf_pair_tf)
                    
            ltf_states = {}
            for sig in setup_config['ltf']:
                if sig in ltf_df.columns:
                    ltf_states[sig] = self.get_or_compute_states(ltf_df, sig, ltf_pair_tf)
            
            # Create momentum confluence mask
            if htf_states and ttf_states and ltf_states:
                momentum_mask = pd.Series(True, index=ltf_df.index)
                
                # HTF context
                for sig, states in htf_states.items():
                    if states is not None:
                        # For momentum strategies, we're more flexible on HTF trend
                        if 'trend_structure' in sig:
                            if direction == 'bullish':
                                momentum_mask &= (states != 'bearish')
                            else:
                                momentum_mask &= (states != 'bullish')
                        else:
                            if direction == 'bullish':
                                momentum_mask &= (states == 'bullish')
                            else:
                                momentum_mask &= (states == 'bearish')
                
                # TTF momentum signals (most important)
                for sig, states in ttf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            momentum_mask &= (states == 'bullish')
                        else:
                            momentum_mask &= (states == 'bearish')
                            
                # LTF confirmation
                for sig, states in ltf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            momentum_mask &= (states == 'bullish')
                        else:
                            momentum_mask &= (states == 'bearish')
                
                if momentum_mask.sum() > 5:  # Even fewer signals for momentum strategies
                    aligned_returns = ltf_df.loc[momentum_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    if win_rate > 0.51:
                        strategies.append({
                            'type': 'mtf_mode_g', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'setup_name': setup_name,
                            'description': setup_config['description'],
                            'htf_signals': setup_config['htf'],
                            'ttf_signals': setup_config['ttf'],
                            'ltf_signals': setup_config['ltf'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(momentum_mask.sum()),
                            'performance_score': win_rate * 1.15,
                            'strategy_class': 'momentum_divergence'
                        })
        
        return strategies


    def discover_mtf_strategies_mode_h(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode H (Refactored): Trend-Context Strategy
        Uses comprehensive trend analysis to filter high-probability setups
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode H (Trend-Context) strategies for {group_name}...")
        
        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        # Get comprehensive trend analysis
        trend_analysis = self.trend_analyzer.classify_mtf_trend_strength(htf_df, ttf_df, ltf_df)
        
        # Only proceed if we have good trend structure
        if (trend_analysis['overall_trend']['trend_strength'] < 60 or 
            trend_analysis['structure_quality_score'] < 50):
            return strategies
        
        # Define trend-context setups
        trend_context_setups = {
            # Strong uptrend setups
            'strong_uptrend_pullback': {
                'required_trend': 'strong_uptrend',
                'description': 'Pullback in strong uptrend with multiple confirmations',
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['pullback_complete_bull'],
                    'ltf': ['swing_low']
                }
            },
            'uptrend_structure_break': {
                'required_trend': 'uptrend',
                'description': 'Structure break in established uptrend',
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['structure_break_bullish'],
                    'ltf': ['higher_highs_lower_lows']
                }
            },
            
            # Strong downtrend setups
            'strong_downtrend_rally': {
                'required_trend': 'strong_downtrend', 
                'description': 'Rally in strong downtrend with multiple confirmations',
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['pullback_complete_bear'],
                    'ltf': ['swing_high']
                }
            },
            'downtrend_structure_break': {
                'required_trend': 'downtrend',
                'description': 'Structure break in established downtrend',
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['structure_break_bearish'],
                    'ltf': ['lower_highs_lower_lows']
                }
            }
        }

        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        current_trend = trend_analysis['overall_trend']['primary_trend']
        
        for setup_name, setup_config in trend_context_setups.items():
            if setup_config['required_trend'] not in current_trend:
                continue
                
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Get signal states
            htf_states = {}
            for sig in setup_config['signals']['htf']:
                if sig in htf_df.columns:
                    htf_states[sig] = self.get_or_compute_states(htf_df, sig, htf_pair_tf)
            
            ttf_states = {}
            for sig in setup_config['signals']['ttf']:
                if sig in ttf_df.columns:
                    ttf_states[sig] = self.get_or_compute_states(ttf_df, sig, ttf_pair_tf)
                    
            ltf_states = {}
            for sig in setup_config['signals']['ltf']:
                if sig in ltf_df.columns:
                    ltf_states[sig] = self.get_or_compute_states(ltf_df, sig, ltf_pair_tf)
            
            # Create trend-context mask
            if htf_states and ttf_states and ltf_states:
                trend_mask = pd.Series(True, index=ltf_df.index)
                
                for sig, states in htf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            trend_mask &= (states == 'bullish')
                        else:
                            trend_mask &= (states == 'bearish')
                
                for sig, states in ttf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            trend_mask &= (states == 'bullish')
                        else:
                            trend_mask &= (states == 'bearish')
                            
                for sig, states in ltf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            trend_mask &= (states == 'bullish')
                        else:
                            trend_mask &= (states == 'bearish')
                
                if trend_mask.sum() > 8:
                    aligned_returns = ltf_df.loc[trend_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    if win_rate > 0.51:  # High threshold for trend-context strategies
                        # Calculate pullback quality for additional filtering
                        pullback_quality = self.pullback_analyzer.analyze_pullback_quality(
                            ltf_df, direction
                        )
                        
                        if pullback_quality['overall_score'] >= 70:
                            strategies.append({
                                'type': 'mtf_mode_h', 
                                "signal_type": "MTF_COMPOSITE",
                                'group': group_name,
                                'pair_tf': f"{pair}_{ltf_tf}",
                                'direction': direction,
                                'trade_direction': 'long' if direction == 'bullish' else 'short',
                                'setup_name': setup_name,
                                'description': setup_config['description'],
                                'trend_context': current_trend,
                                'trend_strength': trend_analysis['overall_trend']['trend_strength'],
                                'pullback_quality': pullback_quality['overall_score'],
                                'htf_signals': setup_config['signals']['htf'],
                                'ttf_signals': setup_config['signals']['ttf'],
                                'ltf_signals': setup_config['signals']['ltf'],
                                'htf_timeframe': htf_tf,
                                'ttf_timeframe': ttf_tf,
                                'ltf_timeframe': ltf_tf,
                                'discovered_accuracy': win_rate,
                                'sample_size': int(trend_mask.sum()),
                                'performance_score': win_rate * (1 + pullback_quality['overall_score'] / 100),
                                'strategy_class': 'trend_context'
                            })
        
        return strategies

    def discover_mtf_strategies_mode_i(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode I (Refactored): High-Quality Pullback Strategy
        Focuses exclusively on high-scoring pullback setups with trend confirmation
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode I (High-Quality Pullback) strategies for {group_name}...")
        
        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # Get trend analysis for context
        trend_analysis = self.trend_analyzer.classify_mtf_trend_strength(htf_df, ttf_df, ltf_df)
        
        # High-quality pullback setups
        hq_pullback_setups = {
            'hq_bullish_pullback': {
                'min_pullback_score': 80,
                'required_trend': ['strong_uptrend', 'uptrend'],
                'description': 'High-quality bullish pullback in uptrend',
                'confirmation_signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['pullback_complete_bull'],
                    'ltf': ['momentum_divergence_bullish']
                }
            },
            'hq_bearish_pullback': {
                'min_pullback_score': 80,
                'required_trend': ['strong_downtrend', 'downtrend'],
                'description': 'High-quality bearish pullback in downtrend',
                'confirmation_signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['pullback_complete_bear'],
                    'ltf': ['momentum_divergence_bearish']
                }
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        current_trend = trend_analysis['overall_trend']['primary_trend']
        
        for setup_name, setup_config in hq_pullback_setups.items():
            if current_trend not in setup_config['required_trend']:
                continue
                
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Analyze pullback quality on LTF (entry timeframe)
            pullback_quality = self.pullback_analyzer.analyze_pullback_quality(ltf_df, direction)
            
            if pullback_quality['overall_score'] < setup_config['min_pullback_score']:
                continue
            
            # Get confirmation signals
            htf_states = {}
            for sig in setup_config['confirmation_signals']['htf']:
                if sig in htf_df.columns:
                    htf_states[sig] = self.get_or_compute_states(htf_df, sig, htf_pair_tf)
            
            ttf_states = {}
            for sig in setup_config['confirmation_signals']['ttf']:
                if sig in ttf_df.columns:
                    ttf_states[sig] = self.get_or_compute_states(ttf_df, sig, ttf_pair_tf)
                    
            ltf_states = {}
            for sig in setup_config['confirmation_signals']['ltf']:
                if sig in ltf_df.columns:
                    ltf_states[sig] = self.get_or_compute_states(ltf_df, sig, ltf_pair_tf)
            
            # Create high-quality pullback mask
            if htf_states and ttf_states and ltf_states:
                hq_mask = pd.Series(True, index=ltf_df.index)
                
                # All confirmation signals must align
                for sig, states in htf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            hq_mask &= (states == 'bullish')
                        else:
                            hq_mask &= (states == 'bearish')
                
                for sig, states in ttf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            hq_mask &= (states == 'bullish')
                        else:
                            hq_mask &= (states == 'bearish')
                            
                for sig, states in ltf_states.items():
                    if states is not None:
                        if direction == 'bullish':
                            hq_mask &= (states == 'bullish')
                        else:
                            hq_mask &= (states == 'bearish')
                
                if hq_mask.sum() > 5:
                    aligned_returns = ltf_df.loc[hq_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    if win_rate > 0.51:
                        strategies.append({
                            'type': 'mtf_mode_i', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'setup_name': setup_name,
                            'description': setup_config['description'],
                            'trend_context': current_trend,
                            'pullback_quality_score': pullback_quality['overall_score'],
                            'pullback_quality_grade': pullback_quality['quality_grade'],
                            'trend_strength': trend_analysis['overall_trend']['trend_strength'],
                            'confirmation_signals': setup_config['confirmation_signals'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(hq_mask.sum()),
                            'performance_score': win_rate * (1 + pullback_quality['overall_score'] / 100),
                            'strategy_class': 'high_quality_pullback'
                        })
        
        return strategies

    def discover_mtf_strategies_mode_j(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode J (Refactored): Regime-Optimized Strategy Discovery
        
        This version is fixed to use a 'historical_regime' column,
        testing strategies *only* on the bars where their compatible
        regime was active.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode J (Regime-Optimized) strategies for {group_name}...")
        
        try:
            if htf_df is None:
                return []
            
            htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
            ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
            ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
            
            if 'historical_regime' not in ltf_df.columns:
                print(f"  ⚠️  Mode J SKIPPED: 'historical_regime' column not found in LFT dataframe.")
                return []
            
            strategies = []
            
            regime_strategies = {
                'trend_following_momentum': {
                    'compatible_regimes': ['strong_trend_high_vol', 'strong_trend_normal_vol', 'weak_trend'],
                    'description': 'Momentum-based trend following in trending markets',
                    'signals': {
                        'htf': ['trend_structure'],
                        'ttf': ['momentum_continuation'],
                        'ltf': ['pullback_complete_bull']
                    }
                },
                'trend_pullback_entries': {
                    'compatible_regimes': ['strong_trend_normal_vol', 'weak_trend'],
                    'description': 'Pullback entries in established trends',
                    'signals': {
                        'htf': ['trend_structure'],
                        'ttf': ['pullback_complete_bull'],
                        'ltf': ['momentum_divergence_bullish']
                    }
                },
                'range_boundary_trading': {
                    'compatible_regimes': ['ranging_high_vol', 'ranging_normal_vol'],
                    'description': 'Trading range boundaries with mean reversion',
                    'signals': {
                        'htf': ['market_structure'],
                        'ttf': ['swing_high', 'swing_low'],
                        'ltf': ['rsi']
                    }
                },
                'breakout_anticipation': {
                    'compatible_regimes': ['ranging_low_vol', 'transition_normal_vol'],
                    'description': 'Anticipating breakouts from low volatility ranges',
                    'signals': {
                        'htf': ['bb_width'],
                        'ttf': ['momentum_continuation'],
                        'ltf': ['structure_break_bullish']
                    }
                }
            }
        
            for strategy_name, strategy_config in regime_strategies.items():
                for direction in ['bullish', 'bearish']:
                    
                    # 1. Build the Signal Mask
                    signal_mask = pd.Series(True, index=ltf_df.index)
                    direction_str = 'bullish' if direction == 'bullish' else 'bearish'

                    for tf, signals in strategy_config['signals'].items():
                        df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                        current_pair_tf = f"{pair}_{htf_tf}" if tf == 'htf' else (f"{pair}_{ttf_tf}" if tf == 'ttf' else f"{pair}_{ltf_tf}")
                        for sig in signals:
                            if sig in df.columns:
                                states = self.get_or_compute_states(df, sig, current_pair_tf)
                                if states is not None:
                                    signal_mask &= (states == direction_str)
                    
                    # 2. Build the Regime Mask (THE CRITICAL FIX)
                    compatible_regimes = strategy_config['compatible_regimes']
                    regime_mask = ltf_df['historical_regime'].isin(compatible_regimes)
                    
                    # 3. Final Mask is the intersection
                    final_mask = signal_mask & regime_mask
                    
                    if final_mask.sum() > 10: # Sample size check
                        active_returns = ltf_df.loc[final_mask, 'future_return']
                        
                        if direction == 'bullish':
                            win_rate = (active_returns > 0).mean()
                        else:
                            win_rate = (active_returns < 0).mean()
                        
                        # --- DISCOVERY THRESHOLD FIX ---
                        # Lowered from 0.56+ to 0.51 to *collect* signals.
                        # The *reporting* phase will filter for 0.60+.
                        if win_rate > 0.51:
                            # Get regime info for the *last bar* for metadata
                            last_bar_regime = ltf_df['historical_regime'].iloc[-1]
                            
                            strategies.append({
                                'type': 'mtf_mode_j', 
                                "signal_type": "MTF_COMPOSITE",
                                'group': group_name,
                                'pair_tf': f"{pair}_{ltf_tf}",
                                'direction': direction,
                                'trade_direction': 'long' if direction == 'bullish' else 'short',
                                'strategy_name': strategy_name,
                                'description': strategy_config['description'],
                                'optimized_regime': ", ".join(compatible_regimes),
                                'current_bar_regime': last_bar_regime,
                                'htf_signals': strategy_config['signals']['htf'],
                                'ttf_signals': strategy_config['signals']['ttf'],
                                'ltf_signals': strategy_config['signals']['ltf'],
                                'htf_timeframe': htf_tf,
                                'ttf_timeframe': ttf_tf,
                                'ltf_timeframe': ltf_tf,
                                'discovered_accuracy': win_rate,
                                'sample_size': int(final_mask.sum()),
                                'performance_score': win_rate * (1 + (final_mask.sum() / len(ltf_df))),
                                'strategy_class': 'regime_optimized'
                            })
                            
            return strategies

        except Exception as e:
            import traceback
            print(f"❌ [discover_mtf_strategies_mode_j] TRACEBACK: {traceback.format_exc()}")
            return []

    def discover_mtf_strategies_mode_k(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode K (Refactored): Adaptive Multi-Regime Strategy Discovery
        Discovers strategies that work across multiple regimes with adaptive parameters
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode K (Adaptive Multi-Regime) strategies for {group_name}...")

        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        # Multi-regime strategy configurations
        multi_regime_strategies = {
            'universal_momentum_capture': {
                'description': 'Momentum capture strategy that adapts to multiple regimes',
                'core_signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['momentum_continuation'],
                    'ltf': ['structure_break_bullish']
                },
                'regime_adaptations': {
                    'trending': {
                        'additional_signals': ['higher_highs_lower_lows', 'pullback_complete_bull'],
                        'parameter_modifier': 1.2
                    },
                    'ranging': {
                        'additional_signals': ['equal_highs_lows', 'false_breakout_bullish'],
                        'parameter_modifier': 0.8
                    },
                    'transition': {
                        'additional_signals': ['swing_failure', 'momentum_divergence_bullish'],
                        'parameter_modifier': 1.5
                    }
                }
            },
            'structure_based_breakout': {
                'description': 'Structure-based breakout strategy with regime adaptation',
                'core_signals': {
                    'htf': ['equal_highs_lows'],
                    'ttf': ['structure_break_bullish', 'structure_break_bearish'],
                    'ltf': ['momentum_continuation']
                },
                'regime_adaptations': {
                    'trending': {
                        'additional_signals': ['trend_structure', 'higher_highs_lower_lows'],
                        'parameter_modifier': 1.1
                    },
                    'ranging': {
                        'additional_signals': ['swing_high', 'swing_low'],
                        'parameter_modifier': 0.9
                    },
                    'transition': {
                        'additional_signals': ['swing_failure', 'volume_divergence'],
                        'parameter_modifier': 1.3
                    }
                }
            }
        }
        
        for strategy_name, strategy_config in multi_regime_strategies.items():
            # Test across different regime contexts
            regime_contexts = self._identify_regime_contexts(htf_df, ttf_df, ltf_df)
            
            for regime_context in regime_contexts:
                for direction in ['bullish', 'bearish']:
                    # Build adaptive signal set based on regime context
                    adaptive_signals = self._build_adaptive_signal_set(
                        strategy_config, regime_context, htf_df, ttf_df, ltf_df
                    )
                    
                    # Create adaptive mask
                    adaptive_mask = self._create_adaptive_signal_mask(
                        adaptive_signals, htf_df, ttf_df, ltf_df, direction, 
                        pair, htf_tf, ttf_tf, ltf_tf
                    )
                    
                    if adaptive_mask.sum() > 6:
                        aligned_returns = ltf_df.loc[adaptive_mask, 'future_return']
                        
                        if direction == 'bullish':
                            win_rate = (aligned_returns > 0).mean()
                        else:
                            win_rate = (aligned_returns < 0).mean()
                        
                        # Calculate regime-adaptive performance score
                        '''regime_modifier = strategy_config['regime_adaptations'][regime_context]['parameter_modifier']'''
                        regime_adaptation = self.get_regime_adaptation(strategy_config, regime_context)
                        regime_modifier = regime_adaptation['parameter_modifier']
                        adaptive_score = win_rate * regime_modifier
                        
                        if win_rate > 0.51:  # Lower threshold for multi-regime strategies
                            strategies.append({
                                'type': 'mtf_mode_k', 
                                "signal_type": "MTF_COMPOSITE",
                                'group': group_name,
                                'pair_tf': f"{pair}_{ltf_tf}",
                                'direction': direction,
                                'trade_direction': 'long' if direction == 'bullish' else 'short',
                                'strategy_name': strategy_name,
                                'description': strategy_config['description'],
                                'regime_context': regime_context,
                                'adaptive_signal_set': adaptive_signals,
                                'regime_modifier': regime_modifier,
                                'htf_timeframe': htf_tf,
                                'ttf_timeframe': ttf_tf,
                                'ltf_timeframe': ltf_tf,
                                'discovered_accuracy': win_rate,
                                'sample_size': int(adaptive_mask.sum()),
                                'performance_score': adaptive_score,
                                'strategy_class': 'adaptive_multi_regime'
                            })
        
        return strategies

    def discover_mtf_strategies_mode_l(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode L (Refactored): Advanced Structure-Aligned Strategy Discovery
        Uses comprehensive MTF structure alignment for high-confidence setups.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode L (Structure-Aligned) strategies for {group_name}...")

        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # Get comprehensive structure alignment analysis
        structure_alignment = self.get_mtf_structure_alignment(
            htf_df, ttf_df, ltf_df, pair, htf_tf, ttf_tf, ltf_tf
            )
        
        # Only proceed with good or excellent alignment
        if structure_alignment['alignment_quality'] in ['poor', 'fair']:
            return strategies
        
        # Define structure-aligned setups
        structure_setups = [
            {
                'name': 'perfect_alignment_breakout',
                'description': 'Perfect MTF alignment with breakout confirmation',
                'required_alignment': 'excellent',
                'min_alignment_score': 0.8,
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['structure_break_bullish'],
                    'ltf': ['higher_highs_lower_lows']
                }
            },
            {
                'name': 'aligned_pullback_entry',
                'description': 'Aligned pullback across timeframes with structure support',
                'required_alignment': 'good',
                'min_alignment_score': 0.6,
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['pullback_complete_bull'],
                    'ltf': ['swing_low']
                }
            }
        ]
        
        current_alignment_quality = structure_alignment['alignment_quality']
        current_alignment_score = structure_alignment['overall_alignment_score']
        
        for setup in structure_setups:
            if (current_alignment_quality == setup['required_alignment'] and 
                current_alignment_score >= setup['min_alignment_score']):
                
                for direction in ['bullish', 'bearish']:
                    # Get signal states
                    signal_mask = pd.Series(True, index=ltf_df.index)
                    
                    for tf, signals in setup['signals'].items():
                        df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                        current_pair_tf = f"{pair}_{htf_tf}" if tf == 'htf' else (f"{pair}_{ttf_tf}" if tf == 'ttf' else f"{pair}_{ltf_tf}")
                        
                        for signal in signals:
                            if signal in df.columns:
                                states = self.get_or_compute_states(df, signal, current_pair_tf)
                                if states is not None:
                                    if direction == 'bullish':
                                        signal_mask &= (states == 'bullish')
                                    else:
                                        signal_mask &= (states == 'bearish')
                    
                    if signal_mask.sum() > 5:
                        aligned_returns = ltf_df.loc[signal_mask, 'future_return']
                        
                        if direction == 'bullish':
                            win_rate = (aligned_returns > 0).mean()
                        else:
                            win_rate = (aligned_returns < 0).mean()
                        
                        if win_rate > 0.51:
                            strategies.append({
                                'type': 'mtf_mode_l', 
                                "signal_type": "MTF_COMPOSITE",
                                'group': group_name,
                                'pair_tf': f"{pair}_{ltf_tf}",
                                'direction': direction,
                                'trade_direction': 'long' if direction == 'bullish' else 'short',
                                'setup_name': setup['name'],
                                'description': setup['description'],
                                'structure_alignment_score': current_alignment_score,
                                'alignment_quality': current_alignment_quality,
                                'alignment_metrics': structure_alignment,
                                'signals': setup['signals'],
                                'htf_timeframe': htf_tf,
                                'ttf_timeframe': ttf_tf,
                                'ltf_timeframe': ltf_tf,
                                'discovered_accuracy': win_rate,
                                'sample_size': int(signal_mask.sum()),
                                'performance_score': win_rate * (1 + current_alignment_score),
                                'strategy_class': 'structure_aligned'
                            })
        
        return strategies

    def discover_mtf_strategies_mode_m(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode M (Refactored): Volatility-Adaptive MTF Strategies
        Combines volatility regimes with MTF structure for adaptive strategy selection.
        Receives dataframes as arguments to prevent redundant loads.
        """
        print(f"  Discovering Mode M (Volatility-Adaptive) strategies for {group_name}...")

        if htf_df is None:
            return []
        
        # Enhance dataframes with advanced patterns
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # Get volatility regime analysis for all timeframes
        htf_volatility = self._analyze_volatility_characteristics(htf_df)
        ttf_volatility = self._analyze_volatility_characteristics(ttf_df)
        ltf_volatility = self._analyze_volatility_characteristics(ltf_df)
        
        # Define volatility-adaptive strategy templates
        volatility_strategies = {
            # HIGH VOLATILITY STRATEGIES
            'high_vol_breakout_momentum': {
                'volatility_regime': 'high_volatility',
                'description': 'Breakout momentum strategy for high volatility periods',
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['structure_break_bullish'],
                    'ltf': ['momentum_continuation']
                },
                'adaptive_params': {
                    'risk_multiplier': 0.7,
                    'stop_loss_pct': 0.03,
                    'take_profit_ratio': 3.0,
                    'position_size': 'reduced'
                }
            },
            'high_vol_range_expansion': {
                'volatility_regime': 'high_volatility', 
                'description': 'Range expansion plays in high volatility',
                'signals': {
                    'htf': ['market_structure'],
                    'ttf': ['equal_highs_lows'],
                    'ltf': ['false_breakout_bullish', 'false_breakout_bearish']
                },
                'adaptive_params': {
                    'risk_multiplier': 0.5,
                    'stop_loss_pct': 0.04,
                    'take_profit_ratio': 2.5,
                    'position_size': 'minimal'
                }
            },
            
            # LOW VOLATILITY STRATEGIES
            'low_vol_compression_breakout': {
                'volatility_regime': 'low_volatility',
                'description': 'Breakout from low volatility compression',
                'signals': {
                    'htf': ['market_structure'],
                    'ttf': ['bb_squeeze'],
                    'ltf': ['structure_break_bullish']
                },
                'adaptive_params': {
                    'risk_multiplier': 1.2,
                    'stop_loss_pct': 0.015,
                    'take_profit_ratio': 4.0,
                    'position_size': 'normal'
                }
            },
            'low_vol_mean_reversion': {
                'volatility_regime': 'low_volatility',
                'description': 'Mean reversion in low volatility ranges',
                'signals': {
                    'htf': ['equal_highs_lows'],
                    'ttf': ['rsi_extreme'],
                    'ltf': ['momentum_divergence_bullish']
                },
                'adaptive_params': {
                    'risk_multiplier': 1.0,
                    'stop_loss_pct': 0.01,
                    'take_profit_ratio': 1.5,
                    'position_size': 'normal'
                }
            },
            
            # NORMAL VOLATILITY STRATEGIES
            'normal_vol_trend_following': {
                'volatility_regime': 'normal_volatility',
                'description': 'Standard trend following in normal volatility',
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['pullback_complete_bull'],
                    'ltf': ['momentum_confirmation']
                },
                'adaptive_params': {
                    'risk_multiplier': 1.0,
                    'stop_loss_pct': 0.02,
                    'take_profit_ratio': 2.0,
                    'position_size': 'normal'
                }
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"

        # Use LTF volatility for primary strategy selection
        current_vol_regime = ltf_volatility['volatility_regime']
        
        for strategy_name, strategy_config in volatility_strategies.items():
            if strategy_config['volatility_regime'] != current_vol_regime:
                continue
                
            for direction in ['bullish', 'bearish']:
                # Get signal states
                vol_mask = pd.Series(True, index=ltf_df.index)
                
                for tf, signals in strategy_config['signals'].items():
                    df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                    current_pair_tf = htf_pair_tf if tf == 'htf' else (ttf_pair_tf if tf == 'ttf' else ltf_pair_tf)
                    
                    for signal in signals:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, current_pair_tf)
                            if states is not None:
                                if direction == 'bullish':
                                    vol_mask &= (states == 'bullish')
                                else:
                                    vol_mask &= (states == 'bearish')
                
                if vol_mask.sum() > 10:
                    aligned_returns = ltf_df.loc[vol_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    # Volatility-specific performance thresholds
                    vol_thresholds = {
                        'high_volatility': 0.56,
                        'normal_volatility': 0.58,
                        'low_volatility': 0.60
                    }
                    
                    required_threshold = vol_thresholds.get(current_vol_regime, 0.60)
                    
                    if win_rate > required_threshold:
                        strategies.append({
                            'type': 'mtf_mode_m', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'strategy_name': strategy_name,
                            'description': strategy_config['description'],
                            'volatility_regime': current_vol_regime,
                            'volatility_score': ltf_volatility['volatility_score'],
                            'adaptive_parameters': strategy_config['adaptive_params'],
                            'signals': strategy_config['signals'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(vol_mask.sum()),
                            'performance_score': win_rate * (1 + ltf_volatility['volatility_score'] / 100),
                            'strategy_class': 'volatility_adaptive'
                        })
        
        return strategies

    def discover_mtf_strategies_mode_n(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
            """
            Mode N: Momentum Cascade Strategies
            Momentum that flows from HTF → TTF → LTF with confirmation.
            Receives dataframes as arguments to prevent redundant loads.
            """
            print(f"  Discovering Mode N (Momentum Cascade) strategies for {group_name}...")
      
            if htf_df is None:
                return []
            
            strategies = []
            
            # Analyze momentum cascade across timeframes
            momentum_cascade = self._analyze_momentum_cascade(htf_df, ttf_df, ltf_df)
            
            # Define momentum cascade patterns
            cascade_patterns = {
                'bullish_momentum_cascade': {
                    'description': 'Bullish momentum flowing from HTF to LTF',
                    'required_flow': 'htf_to_ltf',
                    'momentum_strength': 'strong',
                    'signals': {
                        'htf': ['trend_structure'],
                        'ttf': ['pullback_complete_bull'],
                        'ltf': ['structure_break_bullish']
                    },
                    'cascade_requirements': {
                        'htf_momentum': 'bullish',
                        'ttf_momentum': 'bullish', 
                        'ltf_momentum': 'bullish',
                    }
                },
                'bearish_momentum_cascade': {
                    'description': 'Bearish momentum flowing from HTF to LTF',
                    'required_flow': 'htf_to_ltf',
                    'momentum_strength': 'strong',
                    'signals': {
                        'htf': ['trend_structure'],
                        'ttf': ['pullback_complete_bear'],
                        'ltf': ['structure_break_bearish']
                    },
                    'cascade_requirements': {
                        'htf_momentum': 'bearish',
                        'ttf_momentum': 'bearish',
                        'ltf_momentum': 'bearish',
                    }
                },
            }
            
            htf_pair_tf = f"{pair}_{htf_tf}"
            ttf_pair_tf = f"{pair}_{ttf_tf}"
            ltf_pair_tf = f"{pair}_{ltf_tf}"

            current_flow = momentum_cascade['primary_flow_direction']
            current_strength = momentum_cascade['momentum_strength']
            
            for pattern_name, pattern_config in cascade_patterns.items():
                if (pattern_config['required_flow'] != current_flow or 
                    pattern_config['momentum_strength'] != current_strength):
                    continue
                    
                direction = 'bullish' if 'bullish' in pattern_name else 'bearish'
                
                # Check cascade requirements
                requirements_met = True
                
                # --- START OF THE FIX ---
                # The original loop was flawed and used the wrong key
                reqs = pattern_config['cascade_requirements']
                
                if 'htf_momentum' in reqs and reqs['htf_momentum'] != momentum_cascade['htf_momentum']:
                    requirements_met = False
                if 'ttf_momentum' in reqs and reqs['ttf_momentum'] != momentum_cascade['ttf_momentum']:
                    requirements_met = False
                if 'ltf_momentum' in reqs and reqs['ltf_momentum'] != momentum_cascade['ltf_momentum']:
                    requirements_met = False
                # --- END OF THE FIX ---
                
                if not requirements_met:
                    continue
                
                # Get signal states
                cascade_mask = pd.Series(True, index=ltf_df.index)
                
                for tf, signals in pattern_config['signals'].items():
                    df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                    current_pair_tf = htf_pair_tf if tf == 'htf' else (ttf_pair_tf if tf == 'ttf' else ltf_pair_tf)
                    
                    for signal in signals:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, current_pair_tf)
                            if states is not None:
                                if direction == 'bullish':
                                    cascade_mask &= (states == 'bullish')
                                else:
                                    cascade_mask &= (states == 'bearish')
                
                if cascade_mask.sum() > 5:
                    aligned_returns = ltf_df.loc[cascade_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    # Use the 0.51 threshold
                    if win_rate > 0.51:
                        strategies.append({
                            'type': 'mtf_mode_n', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'pattern_name': pattern_name,
                            'description': pattern_config['description'],
                            'momentum_flow': current_flow,
                            'momentum_strength': current_strength,
                            'cascade_score': momentum_cascade['cascade_score'],
                            'signals': pattern_config['signals'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(cascade_mask.sum()),
                            'performance_score': win_rate * (1 + momentum_cascade['cascade_score'] / 100),
                            'strategy_class': 'momentum_cascade'
                        })
            
            return strategies
    
    def discover_mtf_strategies_mode_confluence(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode Confluence (Refactored):
        
        Finds combinations where Price Action and Indicators align,
        using flexible "N-of-M" (e.g., 60%) confluence logic
        instead of the original strict "all-or-nothing" approach.
        
        Also uses historical regime filtering.
        Receives dataframes as arguments to prevent redundant loads.
        """
        try:
            print(f"  Discovering Mode Confluence (Flexible) strategies for {group_name}...")
                
            # --- START OF FIX (Q4) ---
            # Remove redundant call to get_mtf_dataframes.
            # The DFs are already passed in as arguments.
            if htf_df is None:
                return []            
            
            htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
            ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
            ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
            
            # --- CRITICAL FIX (Same as Mode J) ---
            if 'historical_regime' not in ltf_df.columns:
                print(f"  ⚠️  Mode Confluence SKIPPED: 'historical_regime' column not found.")
                return []
            # --- END FIX ---
            
            htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
            ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
            ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
            
            strategies = []
            
            confluence_patterns = [
                {
                    'name': 'bullish_trend_resumption',
                    'description': 'Strong trend + Pullback completion + Momentum confirmation',
                    'price_action': {
                        'htf': ['trend_structure'],
                        'ttf': ['pullback_complete_bull'],
                        'ltf': ['structure_break_bullish']
                    },
                    'indicators': {
                        'htf': ['adx', 'ema_50'],
                        'ttf': ['rsi', 'macd'],
                        'ltf': ['bb_pct', 'stoch_k']
                    },
                    'regime': ['strong_trend_high_vol', 'strong_trend_normal_vol', 'weak_trend'],
                    'confluence_threshold_pct': 0.6  # Require 60% of signals to be active
                },
                {
                    'name': 'bearish_reversal_confluence',
                    'description': 'Overbought conditions + Bearish divergence + Structure break',
                    'price_action': {
                        'htf': ['market_structure'],
                        'ttf': ['false_breakout_bullish'],
                        'ltf': ['structure_break_bearish']
                    },
                    'indicators': {
                        'htf': ['rsi'],
                        'ttf': ['macd', 'stoch_k'],
                        'ltf': ['rsi', 'bb_pct']
                    },
                    'regime': ['ranging_high_vol', 'ranging_normal_vol', 'transition_normal_vol'],
                    'confluence_threshold_pct': 0.6
                }
            ]
            
            for pattern in confluence_patterns:
                direction = 'bullish' if 'bullish' in pattern['name'] else 'bearish'
                direction_str = 'bullish' if direction == 'bullish' else 'bearish'
                
                # --- FLEXIBLE CONFLUENCE LOGIC ---
                
                # 1. Collect all state Series
                pa_states = []
                for tf, signals in pattern['price_action'].items():
                    df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                    current_pair_tf = f"{pair}_{htf_tf}" if tf == 'htf' else (f"{pair}_{ttf_tf}" if tf == 'ttf' else f"{pair}_{ltf_tf}")
                    for signal in signals:
                        if signal in df.columns:
                            pa_states.append(self.get_or_compute_states(df, signal, current_pair_tf))

                indicator_states = []
                for tf, signals in pattern['indicators'].items():
                    df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                    current_pair_tf = f"{pair}_{htf_tf}" if tf == 'htf' else (f"{pair}_{ttf_tf}" if tf == 'ttf' else f"{pair}_{ltf_tf}")
                    for signal in signals:
                        if signal in df.columns:
                            indicator_states.append(self.get_or_compute_states(df, signal, current_pair_tf))

                # 2. Convert states to boolean masks for the correct direction
                pa_masks = [(s == direction_str) for s in pa_states if s is not None]
                indicator_masks = [(s == direction_str) for s in indicator_states if s is not None]

                if not pa_masks or not indicator_masks:
                    continue # Not enough signals to test

                # 3. Sum the masks to get a *score Series* (e.g., 0 to 5)
                # sum() on a list of Series performs element-wise addition
                pa_score_series = sum(pa_masks)
                indicator_score_series = sum(indicator_masks)

                # 4. Determine the *required* score
                pa_min_required = int(np.ceil(len(pa_masks) * pattern['confluence_threshold_pct']))
                indicator_min_required = int(np.ceil(len(indicator_masks) * pattern['confluence_threshold_pct']))

                # 5. Create the flexible masks
                pa_mask = (pa_score_series >= pa_min_required)
                indicator_mask = (indicator_score_series >= indicator_min_required)
                
                # --- END FLEXIBLE LOGIC ---

                # 6. Create the Regime Mask
                regime_mask = ltf_df['historical_regime'].isin(pattern['regime'])

                # 7. Final Mask & Backtest
                confluence_mask = pa_mask & indicator_mask & regime_mask
                
                if confluence_mask.sum() > 10:
                    aligned_returns = ltf_df.loc[confluence_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    # --- DISCOVERY THRESHOLD FIX ---
                    if win_rate > 0.51:
                        
                        # Use the dedicated Confluence Scorer for *last bar* metadata
                        last_bar_confluence = self.confluence_scorer.calculate_mtf_confluence_score(
                            htf_df, ttf_df, ltf_df, 
                            {'htf': pattern['price_action']['htf'] + pattern['indicators']['htf'],
                             'ttf': pattern['price_action']['ttf'] + pattern['indicators']['ttf'],
                             'ltf': pattern['price_action']['ltf'] + pattern['indicators']['ltf']}, 
                            direction
                        )
                        
                        strategies.append({
                            'type': 'mtf_confluence', 
                            "signal_type": "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'pattern_name': pattern['name'],
                            'description': pattern['description'],
                            'regime_context': ", ".join(pattern['regime']),
                            'price_action_signals': pattern['price_action'],
                            'indicator_signals': pattern['indicators'],
                            'confluence_score_last_bar': last_bar_confluence['overall_score'],
                            'confluence_grade_last_bar': last_bar_confluence['grade'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(confluence_mask.sum()),
                            'performance_score': win_rate * (1 + last_bar_confluence['overall_score'] / 100),
                            'strategy_class': 'price_action_indicator_confluence'
                        })
            
            return strategies    
        except Exception as e:
            import traceback
            print(f"❌ [discover_mtf_strategies_mode_confluence] TRACEBACK: {traceback.format_exc()}")
            return []

    def discover_mtf_strategies(self):
        """
        Comprehensive MTF strategy discovery with all advanced modes including Phase 5
        """
        print("\n" + "="*80)
        print("ADVANCED MULTI-TIMEFRAME STRATEGY DISCOVERY WITH PHASE 5 FEATURES")
        print("="*80)
        
        all_mtf_strategies = []
        strategy_id = len(self.strategy_pool) + 1
        
        # Define proper timeframe hierarchy order (HTF to LTF)
        timeframe_hierarchy = {
            '4h' : 0,
            '1h' : 1,
            '15m': 2,
            '5m' : 3,
            '1m' : 4
        }
        
        # Sort groups by their highest timeframe (HTF) to maintain proper analysis order
        def get_group_order(group_name):
            config = TIMEFRAME_GROUPS[group_name]
            htf = config['HTF']
            return timeframe_hierarchy.get(htf, 999)
        
        sorted_groups = sorted(TIMEFRAME_GROUPS.keys(), key=get_group_order)
        
        print(f"Analysis order: {' -> '.join(sorted_groups)}")

        pair = "btc_usdt" # Define the pair once for the entire run
        
        for group_name in sorted_groups:
            print(f"\nAnalyzing {group_name}.")

            # -----------------------------
            # Enforce HTF -> TTF -> LTF order
            # -----------------------------
            # Compute & cache HTF/TTF/LTF first (this ensures HTF is analyzed before lower TFs)
            try:
                group_cfg = TIMEFRAME_GROUPS[group_name]
                htf_tf = group_cfg["HTF"]
                ttf_tf = group_cfg["TTF"]
                ltf_tf = group_cfg["LTF"]
            except Exception:
                # fallback: just run original per-group flow if config missing
                htf_tf = ttf_tf = ltf_tf = None

            # fetch dataframes (use existing helper if available)
            try:
                # many existing helper names exist in the file: try the enhanced getter first,
                # then fall back to simpler getter patterns used elsewhere.
                htf_df, ttf_df, ltf_df = self.get_mtf_dataframes(pair, htf_tf, ttf_tf, ltf_tf)

                if htf_df is None or ttf_df is None or ltf_df is None:
                    print(f"  ⚠️  Skipping {group_name}: Could not load all dataframes.")
                    continue

            except Exception:
                htf_df = ttf_df = ltf_df = None
                print(f"  ⚠️  Skipping {group_name}: Error during dataframe loading.")
                continue

            # run discovery modes sequentially (HTF-first is guaranteed because htf_df computed above)
            # NOTE: we call modes one-by-one rather than in parallel to ensure order and reproducibility
            strategy_results = {}
            for mode_label, mode_fn in [
                ('A', self.discover_mtf_strategies_mode_a),
                ('B', self.discover_mtf_strategies_mode_b),
                ('C', self.discover_mtf_strategies_mode_c),
                ('D', self.discover_mtf_strategies_mode_d),
                ('E', self.discover_mtf_strategies_mode_e),
                ('F', self.discover_mtf_strategies_mode_f),
                ('G', self.discover_mtf_strategies_mode_g),
                ('H', self.discover_mtf_strategies_mode_h),
                ('I', self.discover_mtf_strategies_mode_i),
                ('J', self.discover_mtf_strategies_mode_j),
                ('K', self.discover_mtf_strategies_mode_k),
                ('L', self.discover_mtf_strategies_mode_l),
                ('M', getattr(self, 'discover_mtf_strategies_mode_m', lambda *args: [])),
                ('N', getattr(self, 'discover_mtf_strategies_mode_n', lambda *args: [])),
                ('Confluence', self.discover_mtf_strategies_mode_confluence)
            ]:
                try:
                    # Each mode may internally fetch dataframes again; but because we cached the
                    # computed HTF/TTF/LTF above, modes can (optionally) reuse those to guarantee HTF-first logic.
                    res = mode_fn(group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf)
                    strategy_results[mode_label] = res if res is not None else []
                except Exception as e:
                    print(f"  ! Mode {mode_label} failed for {group_name}: {e}")
                    #print(f"  ! Mode {mode_label} failed for {group_name}: {e}\n{traceback.format_exc()}") # Added full traceback
                    strategy_results[mode_label] = []

            # Combine all results
            group_strategies = []
            strategy_counts = {}

            for mode, strategies in strategy_results.items():
                group_strategies.extend(strategies)
                strategy_counts[mode] = len(strategies)

            all_mtf_strategies.extend(group_strategies)

            # Print detailed breakdown
            print(f"  ✓ {group_name}: ", end="")
            for mode, count in strategy_counts.items():
                print(f"{count}{mode} ", end="")
            print()


        
        # Add to strategy pool
        for strategy in all_mtf_strategies:
            strategy_key = f"MTF_{strategy_id:04d}"
            strategy['id'] = strategy_id
            self.strategy_pool[strategy_key] = strategy
            strategy_id += 1
        
        # Generate comprehensive strategy analysis
        self._analyze_phase5_strategy_breakdown(all_mtf_strategies)

        print(f"\n🎯 DISCOVERY COMPLETE: {len(all_mtf_strategies)} strategies found")
        
        # Generate report
        self.generate_strategy_report()
        
        # Save to file
        self.save_strategies_to_file()
        
        # Export to CSV
        self.export_strategies_to_csv()
        
        return all_mtf_strategies
    
    def build_strategy_correlation_matrix(self, min_signals=30):
            """Analyze correlation between strategies to build diversified portfolios"""
            print("\nAnalyzing strategy correlations...")
            
            valid_strategies = {
                k: v for k, v in self.strategy_pool.items()
                if v.get('backtest_total_signals', 0) >= min_signals
            }
            
            if len(valid_strategies) < 2:
                print("Insufficient strategies for correlation analysis")
                return None
            
            strategy_returns = {}
            
            for strat_id, strategy in valid_strategies.items():
                returns = self._extract_strategy_returns(strategy)
                if returns is not None and len(returns) > 0:
                    strategy_returns[strat_id] = returns
            
            if len(strategy_returns) < 2:
                return None
            
            returns_df = pd.DataFrame(strategy_returns)
            correlation_matrix = returns_df.corr()
            
            diversified_portfolios = self._find_diversified_combinations(
                correlation_matrix, valid_strategies
            )
            
            return {
                'correlation_matrix': correlation_matrix,
                'diversified_portfolios': diversified_portfolios,
                'strategy_returns': returns_df
            }
