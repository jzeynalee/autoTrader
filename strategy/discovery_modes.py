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

    def _calculate_strategy_quality_score(self, strategy, ltf_df, final_mask):
        """
        Calculate comprehensive quality score for a strategy.
        Returns score 0-100 based on multiple factors.
        """
        if final_mask.sum() < 10:
            return 0
        
        active_bars = ltf_df.loc[final_mask]
        
        # Factor 1: Win Rate (0-40 points)
        win_rate = strategy.get('discovered_accuracy', 0)
        win_rate_score = min(40, (win_rate - 0.5) * 200)  # 51% = 2pts, 70% = 40pts
        
        # Factor 2: Sample Size (0-20 points)
        sample_size = final_mask.sum()
        sample_score = min(20, (sample_size / 100) * 20)  # 100 samples = 20pts
        
        # Factor 3: Profit Factor (0-20 points)
        returns = active_bars['future_return']
        direction = strategy['direction']
        
        if direction == 'bullish':
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns <= 0]
        else:
            winning_returns = returns[returns < 0]
            losing_returns = returns[returns >= 0]
        
        if len(losing_returns) > 0 and losing_returns.sum() != 0:
            profit_factor = abs(winning_returns.sum() / losing_returns.sum())
            pf_score = min(20, (profit_factor - 1) * 10)  # PF of 3 = 20pts
        else:
            pf_score = 0
        
        # Factor 4: Consistency (0-20 points)
        # Check if strategy works across different market conditions
        if 'historical_regime' in ltf_df.columns:
            regimes = active_bars['historical_regime'].unique()
            consistency_score = min(20, len(regimes) * 5)  # Works in 4+ regimes = 20pts
        else:
            consistency_score = 10  # Neutral score
        
        total_score = win_rate_score + sample_score + pf_score + consistency_score
        
        return {
            'total_quality_score': total_score,
            'win_rate_component': win_rate_score,
            'sample_size_component': sample_score,
            'profit_factor_component': pf_score,
            'consistency_component': consistency_score,
        }

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
        Mode D: ADVANCED Price Action with Multi-Signal Confluence
        Uses 3-5 signals per timeframe for high-quality setups
        """
        print(f"  Discovering Mode D (Advanced Price Action) strategies for {group_name}...")
        
        if htf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # *** ADVANCED MULTI-SIGNAL SETUPS ***
        advanced_setups = {
            # === BULLISH SETUPS (Multi-Layer Filtering) ===
            'bullish_trend_pullback_entry': {
                'description': 'HTF uptrend + TTF pullback complete + LTF entry confirmation',
                'htf': {
                    'trend_context': ['trend_medium', 'higher_lows_pattern'],  # Trend must be up
                    'structure': ['higher_highs'],  # Structure confirmation
                },
                'ttf': {
                    'pullback_signals': ['pullback_complete_bull', 'healthy_bull_pullback'],
                    'volume_context': ['volume_decreasing'],  # Volume should decrease on pullback
                },
                'ltf': {
                    'entry_triggers': ['swing_low', 'near_fib_618', 'near_fib_500'],  # Multiple fib levels
                    'confirmation': ['sr_confluence_score'],  # Support/Resistance confluence
                    'momentum': ['rsi'],  # RSI not overbought
                },
                'min_confluence': 3,  # At least 3 signals must align
            },
            
            'bullish_breakout_structure': {
                'description': 'Multi-timeframe structure break with volume',
                'htf': {
                    'trend_context': ['trend_long', 'trend_medium'],
                    'pattern': ['abc_pullback_bull'],
                },
                'ttf': {
                    'structure': ['structure_break_bullish'],
                    'confirmation': ['higher_highs'],
                },
                'ltf': {
                    'entry_triggers': ['volume_breakout', 'momentum_continuation'],
                    'indicators': ['rsi', 'macd_hist'],
                },
                'min_confluence': 3,
            },
            
            'bullish_reversal_confluence': {
                'description': 'Reversal setup with divergence + structure',
                'htf': {
                    'reversal_context': ['failed_pullback_bear', 'swing_failure'],
                },
                'ttf': {
                    'divergence': ['momentum_divergence_bullish'],
                    'structure': ['false_breakout_bearish'],
                },
                'ltf': {
                    'entry_triggers': ['swing_low', 'structure_break_bullish'],
                    'volume': ['volume_breakout'],
                },
                'min_confluence': 2,
            },
            
            # === BEARISH SETUPS (Mirror Logic) ===
            'bearish_trend_pullback_entry': {
                'description': 'HTF downtrend + TTF rally exhaustion + LTF entry',
                'htf': {
                    'trend_context': ['trend_medium', 'lower_highs_pattern'],
                    'structure': ['lower_lows'],
                },
                'ttf': {
                    'pullback_signals': ['pullback_complete_bear', 'healthy_bear_pullback'],
                    'volume_context': ['volume_decreasing'],
                },
                'ltf': {
                    'entry_triggers': ['swing_high', 'near_fib_382', 'near_fib_236'],
                    'confirmation': ['sr_confluence_score'],
                    'momentum': ['rsi'],
                },
                'min_confluence': 3,
            },
            
            'bearish_breakout_structure': {
                'description': 'Multi-timeframe bearish structure break',
                'htf': {
                    'trend_context': ['trend_long', 'trend_medium'],
                    'pattern': ['abc_pullback_bear'],
                },
                'ttf': {
                    'structure': ['structure_break_bearish'],
                    'confirmation': ['lower_lows'],
                },
                'ltf': {
                    'entry_triggers': ['volume_breakout', 'momentum_continuation'],
                    'indicators': ['rsi', 'macd_hist'],
                },
                'min_confluence': 3,
            },
            
            'bearish_reversal_confluence': {
                'description': 'Reversal with divergence + failed breakout',
                'htf': {
                    'reversal_context': ['failed_pullback_bull', 'swing_failure'],
                },
                'ttf': {
                    'divergence': ['momentum_divergence_bearish'],
                    'structure': ['false_breakout_bullish'],
                },
                'ltf': {
                    'entry_triggers': ['swing_high', 'structure_break_bearish'],
                    'volume': ['volume_breakout'],
                },
                'min_confluence': 2,
            },
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        # === PROCESS EACH SETUP ===
        for setup_name, setup_config in advanced_setups.items():
            direction = 'bullish' if 'bullish' in setup_name else 'bearish'
            
            # Build multi-signal mask for each timeframe
            htf_masks = []
            ttf_masks = []
            ltf_masks = []
            
            # === HTF: Collect all signal masks ===
            for category, signals in setup_config['htf'].items():
                for sig in signals:
                    if sig in htf_df.columns:
                        states = self.get_or_compute_states(htf_df, sig, htf_pair_tf)
                        if states is not None:
                            mask = (states == direction)
                            htf_masks.append(mask)
            
            # === TTF: Collect all signal masks ===
            for category, signals in setup_config['ttf'].items():
                for sig in signals:
                    if sig in ttf_df.columns:
                        states = self.get_or_compute_states(ttf_df, sig, ttf_pair_tf)
                        if states is not None:
                            mask = (states == direction)
                            ttf_masks.append(mask)
            
            # === LTF: Collect all signal masks ===
            for category, signals in setup_config['ltf'].items():
                for sig in signals:
                    if sig in ltf_df.columns:
                        states = self.get_or_compute_states(ltf_df, sig, ltf_pair_tf)
                        if states is not None:
                            # Special handling for RSI (check for oversold/overbought)
                            if sig == 'rsi':
                                if direction == 'bullish':
                                    mask = (states == 'bullish') | (ltf_df['rsi'] < 50)
                                else:
                                    mask = (states == 'bearish') | (ltf_df['rsi'] > 50)
                            else:
                                mask = (states == direction)
                            ltf_masks.append(mask)
            
            # === CONFLUENCE CHECK: Sum how many signals are active ===
            if len(htf_masks) == 0 or len(ttf_masks) == 0 or len(ltf_masks) == 0:
                continue
            
            # Calculate confluence score (how many signals align)
            htf_confluence = sum(htf_masks)  # Element-wise sum
            ttf_confluence = sum(ttf_masks)
            ltf_confluence = sum(ltf_masks)
            
            # Require minimum confluence on each timeframe
            min_conf = setup_config['min_confluence']
            final_mask = (
                (htf_confluence >= min(min_conf, len(htf_masks))) &
                (ttf_confluence >= min(min_conf, len(ttf_masks))) &
                (ltf_confluence >= min(min_conf, len(ltf_masks)))
            )
            
            # === BACKTEST ===
            if final_mask.sum() > 10:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate > 0.51:
                    # Calculate average confluence score for this strategy
                    avg_confluence = (
                        htf_confluence.loc[final_mask].mean() +
                        ttf_confluence.loc[final_mask].mean() +
                        ltf_confluence.loc[final_mask].mean()
                    ) / 3
                    
                    strategies.append({
                        'type': 'mtf_mode_d',
                        'signal_type': "MTF_COMPOSITE",
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
                        'sample_size': int(final_mask.sum()),
                        'avg_confluence_score': float(avg_confluence),
                        'min_required_confluence': min_conf,
                        'performance_score': win_rate * (1 + avg_confluence / 10),  # Bonus for high confluence
                        'strategy_class': 'advanced_multi_signal'
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

    # ============================================================================
    # COMPLETE FIX FOR MODES F-N + CONFLUENCE 11/07/25-09:47
    # ============================================================================
    # The issue is that the discovery is TOO STRICT. Here's what to do:

    # 1. LOWER THE MINIMUM SAMPLE SIZE from 5 to 3
    # 2. LOWER THE WIN RATE THRESHOLD from 0.51 to 0.48 for initial discovery
    # 3. ADD FALLBACK PATTERNS when primary patterns don't exist

    # Apply these changes to ALL non-working modes:

    # ============================================================================
    # MODE F FIX
    # ============================================================================
    def discover_mtf_strategies_mode_f(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode F: Structure Breakout Strategies
        FIXED: Safe index alignment + relaxed thresholds + better pattern detection
        """
        print(f"  Discovering Mode F (Structure Breakout) strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        # === RELAXED SETUPS with FALLBACKS ===
        advanced_setups = {
            'structural_breakout_bull': {
                'description': 'Bullish structure break with momentum',
                'htf': {
                    'primary': ['higher_highs', 'trend_structure'],
                    'fallback': ['ema_50', 'adx']  # Use if primary missing
                },
                'ttf': {
                    'primary': ['momentum_divergence_bullish', 'structure_break_bullish'],
                    'fallback': ['rsi', 'macd']
                },
                'ltf': {
                    'primary': ['volume_breakout', 'momentum_continuation'],
                    'fallback': ['volume', 'close']  # Simple volume spike
                }
            },
            'structural_breakout_bear': {
                'description': 'Bearish structure break with momentum',
                'htf': {
                    'primary': ['lower_lows', 'trend_structure'],
                    'fallback': ['ema_50', 'adx']
                },
                'ttf': {
                    'primary': ['momentum_divergence_bearish', 'structure_break_bearish'],
                    'fallback': ['rsi', 'macd']
                },
                'ltf': {
                    'primary': ['volume_breakout', 'momentum_continuation'],
                    'fallback': ['volume', 'close']
                }
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        for setup_name, setup_config in advanced_setups.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # === BUILD MASKS WITH SAFE INDEX ALIGNMENT ===
            
            # Start with LTF index (the shortest timeframe)
            final_mask = pd.Series(True, index=ltf_df.index)
            used_signals = {'htf': [], 'ttf': [], 'ltf': []}
            
            # Process each timeframe
            for tf_label, df, pair_tf_str in [
                ('htf', htf_df, htf_pair_tf),
                ('ttf', ttf_df, ttf_pair_tf),
                ('ltf', ltf_df, ltf_pair_tf)
            ]:
                signals_config = setup_config[tf_label]
                
                # Try primary signals first
                for signal in signals_config['primary']:
                    if signal in df.columns:
                        states = self.get_or_compute_states(df, signal, pair_tf_str)
                        if states is not None:
                            # *** CRITICAL: Align to LTF index ***
                            states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                            
                            if direction == 'bullish':
                                mask = (states_aligned == 'bullish') | (states_aligned == 1)
                            else:
                                mask = (states_aligned == 'bearish') | (states_aligned == -1)
                            
                            final_mask &= mask
                            used_signals[tf_label].append(signal)
                            break  # Use first available primary signal
                
                # If no primary signals worked, try fallback
                if not used_signals[tf_label]:
                    for signal in signals_config['fallback']:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, pair_tf_str)
                            if states is not None:
                                states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                
                                if direction == 'bullish':
                                    mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                else:
                                    mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                
                                final_mask &= mask
                                used_signals[tf_label].append(f"{signal}_fallback")
                                break
            
            # === RELAXED THRESHOLDS ===
            min_samples = 5  # Reduced from 10
            min_win_rate = 0.50  # Reduced from 0.51
            
            if final_mask.sum() >= min_samples:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate >= min_win_rate:
                    strategies.append({
                        'type': 'mtf_mode_f',
                        'signal_type': "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'description': setup_config['description'],
                        'htf_signals': used_signals['htf'],
                        'ttf_signals': used_signals['ttf'],
                        'ltf_signals': used_signals['ltf'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(final_mask.sum()),
                        'performance_score': win_rate * 1.2,
                        'strategy_class': 'advanced_structure_breakout'
                    })
        
        return strategies

    # ============================================================================
    # APPLY THE SAME PATTERN TO MODES G, H, I, M, N, CONFLUENCE
    # ============================================================================

    # Key changes:
    # 1. Use simpler patterns from vectorized analysis (higher_highs, lower_lows, etc.)
    # 2. Lower minimum sample size to 3
    # 3. Lower win rate threshold to 0.48
    # 4. Support both string ('bullish') and numeric (1) state formats
    # 5. Skip gracefully if patterns don't exist

    # For the remaining modes, the pattern is:
    # - Use patterns that ARE created by optimized_detect_advanced_price_patterns
    # - Don't rely on patterns from calculator.py that might not exist
    # - Be flexible with state matching ((states == 'bullish') | (states == 1))

    def discover_mtf_strategies_mode_g(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode G: Momentum Divergence (FIXED)
        """
        print(f"  Discovering Mode G (Momentum Divergence) strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        momentum_setups = {
            'momentum_divergence_bull': {
                'description': 'Bullish momentum divergence with structure support',
                'htf': {
                    'primary': ['trend_structure'],
                    'fallback': ['ema_50']
                },
                'ttf': {
                    'primary': ['momentum_divergence_bullish'],
                    'fallback': ['rsi']  # RSI < 40
                },
                'ltf': {
                    'primary': ['false_breakout_bearish', 'structure_break_bullish'],
                    'fallback': ['swing_low']
                }
            },
            'momentum_divergence_bear': {
                'description': 'Bearish momentum divergence with structure resistance',
                'htf': {
                    'primary': ['trend_structure'],
                    'fallback': ['ema_50']
                },
                'ttf': {
                    'primary': ['momentum_divergence_bearish'],
                    'fallback': ['rsi']  # RSI > 60
                },
                'ltf': {
                    'primary': ['false_breakout_bullish', 'structure_break_bearish'],
                    'fallback': ['swing_high']
                }
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        for setup_name, setup_config in momentum_setups.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Build mask with LTF index
            final_mask = pd.Series(True, index=ltf_df.index)
            used_signals = {'htf': [], 'ttf': [], 'ltf': []}
            
            for tf_label, df, pair_tf_str in [
                ('htf', htf_df, htf_pair_tf),
                ('ttf', ttf_df, ttf_pair_tf),
                ('ltf', ltf_df, ltf_pair_tf)
            ]:
                signals_config = setup_config[tf_label]
                
                # Try primary, then fallback
                for signal_list in [signals_config['primary'], signals_config['fallback']]:
                    signal_found = False
                    for signal in signal_list:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, pair_tf_str)
                            if states is not None:
                                states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                
                                # Special handling for RSI fallback
                                if signal == 'rsi':
                                    if direction == 'bullish':
                                        mask = df['rsi'].reindex(final_mask.index) < 40
                                    else:
                                        mask = df['rsi'].reindex(final_mask.index) > 60
                                else:
                                    if direction == 'bullish':
                                        mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                    else:
                                        mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                
                                final_mask &= mask
                                used_signals[tf_label].append(signal)
                                signal_found = True
                                break
                    
                    if signal_found:
                        break
            
            # Relaxed thresholds
            if final_mask.sum() >= 5:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate >= 0.50:
                    strategies.append({
                        'type': 'mtf_mode_g',
                        'signal_type': "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'description': setup_config['description'],
                        'htf_signals': used_signals['htf'],
                        'ttf_signals': used_signals['ttf'],
                        'ltf_signals': used_signals['ltf'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(final_mask.sum()),
                        'performance_score': win_rate * 1.15,
                        'strategy_class': 'momentum_divergence'
                    })
        
        return strategies

    
    def discover_mtf_strategies_mode_h(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode H: Trend-Context Strategy (FIXED)
        """
        print(f"  Discovering Mode H (Trend-Context) strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        # Get trend analysis (with error handling)
        try:
            trend_analysis = self.trend_analyzer.classify_mtf_trend_strength(htf_df, ttf_df, ltf_df)
            current_trend = trend_analysis['overall_trend']['primary_trend']
            trend_strength = trend_analysis['overall_trend']['trend_strength']
        except Exception as e:
            print(f"  ⚠️  Trend analysis failed: {e}")
            return strategies
        
        # Skip weak trends
        if trend_strength < 50:
            return strategies
        
        trend_context_setups = {
            'uptrend_pullback': {
                'required_trend': ['strong_uptrend', 'uptrend'],
                'description': 'Pullback entry in established uptrend',
                'htf': {
                    'primary': ['trend_structure'],
                    'fallback': ['ema_50']
                },
                'ttf': {
                    'primary': ['pullback_complete_bull', 'healthy_bull_pullback'],
                    'fallback': ['swing_low']
                },
                'ltf': {
                    'primary': ['structure_break_bullish', 'momentum_continuation'],
                    'fallback': ['rsi']
                }
            },
            'downtrend_rally': {
                'required_trend': ['strong_downtrend', 'downtrend'],
                'description': 'Rally fade in established downtrend',
                'htf': {
                    'primary': ['trend_structure'],
                    'fallback': ['ema_50']
                },
                'ttf': {
                    'primary': ['pullback_complete_bear', 'healthy_bear_pullback'],
                    'fallback': ['swing_high']
                },
                'ltf': {
                    'primary': ['structure_break_bearish', 'momentum_continuation'],
                    'fallback': ['rsi']
                }
            }
        }
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        for setup_name, setup_config in trend_context_setups.items():
            # Check if current trend matches required trend
            if not any(trend in current_trend for trend in setup_config['required_trend']):
                continue
            
            direction = 'bullish' if 'uptrend' in setup_name else 'bearish'
            
            # Build mask
            final_mask = pd.Series(True, index=ltf_df.index)
            used_signals = {'htf': [], 'ttf': [], 'ltf': []}
            
            for tf_label, df, pair_tf_str in [
                ('htf', htf_df, htf_pair_tf),
                ('ttf', ttf_df, ttf_pair_tf),
                ('ltf', ltf_df, ltf_pair_tf)
            ]:
                signals_config = setup_config[tf_label]
                
                for signal_list in [signals_config['primary'], signals_config['fallback']]:
                    signal_found = False
                    for signal in signal_list:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, pair_tf_str)
                            if states is not None:
                                states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                
                                if direction == 'bullish':
                                    mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                else:
                                    mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                
                                final_mask &= mask
                                used_signals[tf_label].append(signal)
                                signal_found = True
                                break
                    
                    if signal_found:
                        break
            
            if final_mask.sum() >= 5:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate >= 0.50:
                    strategies.append({
                        'type': 'mtf_mode_h',
                        'signal_type': "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'description': setup_config['description'],
                        'trend_context': current_trend,
                        'trend_strength': trend_strength,
                        'htf_signals': used_signals['htf'],
                        'ttf_signals': used_signals['ttf'],
                        'ltf_signals': used_signals['ltf'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(final_mask.sum()),
                        'performance_score': win_rate * 1.1,
                        'strategy_class': 'trend_context'
                    })
        
        return strategies
    # ============================================================================
    # TEMPLATE FOR REMAINING MODES (I, K, M, N)
    # ============================================================================
    MODE_I_SETUPS = {
        'hq_bullish_pullback': {
            'description': 'High-quality bullish pullback in uptrend',
            'htf': {
                'primary': ['trend_structure'],
                'fallback': ['ema_50']
            },
            'ttf': {
                'primary': ['pullback_complete_bull', 'healthy_bull_pullback'],
                'fallback': ['swing_low']
            },
            'ltf': {
                'primary': ['momentum_divergence_bullish', 'structure_break_bullish'],
                'fallback': ['rsi', 'volume_breakout']
            },
            'bonus_multiplier': 1.2,
            'strategy_class': 'high_quality_pullback'
        },
        'hq_bearish_pullback': {
            'description': 'High-quality bearish pullback in downtrend',
            'htf': {
                'primary': ['trend_structure'],
                'fallback': ['ema_50']
            },
            'ttf': {
                'primary': ['pullback_complete_bear', 'healthy_bear_pullback'],
                'fallback': ['swing_high']
            },
            'ltf': {
                'primary': ['momentum_divergence_bearish', 'structure_break_bearish'],
                'fallback': ['rsi', 'volume_breakout']
            },
            'bonus_multiplier': 1.2,
            'strategy_class': 'high_quality_pullback'
        }
    }

    # how to call:
    # discover_mtf_strategies_mode_template(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, 'I', MODE_I_SETUPS)
    def discover_mtf_strategies_mode_template(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, mode_name, setups_config):
        """
        UNIVERSAL TEMPLATE for Modes I, K, M, N
        Just pass different setups_config for each mode
        """
        print(f"  Discovering Mode {mode_name} strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        for setup_name, setup_config in setups_config.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Build mask with safe alignment
            final_mask = pd.Series(True, index=ltf_df.index)
            used_signals = {'htf': [], 'ttf': [], 'ltf': []}
            
            for tf_label, df, pair_tf_str in [
                ('htf', htf_df, htf_pair_tf),
                ('ttf', ttf_df, ttf_pair_tf),
                ('ltf', ltf_df, ltf_pair_tf)
            ]:
                signals_config = setup_config[tf_label]
                
                for signal_list in [signals_config.get('primary', []), signals_config.get('fallback', [])]:
                    signal_found = False
                    for signal in signal_list:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, pair_tf_str)
                            if states is not None:
                                states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                
                                if direction == 'bullish':
                                    mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                else:
                                    mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                
                                final_mask &= mask
                                used_signals[tf_label].append(signal)
                                signal_found = True
                                break
                    
                    if signal_found:
                        break
            
            # Relaxed thresholds
            if final_mask.sum() >= 5:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate >= 0.50:
                    strategies.append({
                        'type': f'mtf_mode_{mode_name.lower()}',
                        'signal_type': "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'description': setup_config.get('description', ''),
                        'htf_signals': used_signals['htf'],
                        'ttf_signals': used_signals['ttf'],
                        'ltf_signals': used_signals['ltf'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(final_mask.sum()),
                        'performance_score': win_rate * setup_config.get('bonus_multiplier', 1.0),
                        'strategy_class': setup_config.get('strategy_class', 'advanced_mtf')
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

    #_________________________________________
    MODE_K_SETUPS = {
        'universal_momentum_bull': {
            'description': 'Adaptive momentum capture for multiple market regimes (Bullish)',
            'regime_compatible': ['trending', 'ranging', 'transition'],  # Works in all regimes
            'htf': {
                'primary': ['trend_structure', 'market_structure'],
                'fallback': ['ema_50', 'adx']
            },
            'ttf': {
                'primary': ['momentum_continuation', 'structure_break_bullish'],
                'fallback': ['macd', 'rsi']
            },
            'ltf': {
                'primary': ['volume_breakout', 'momentum_divergence_bullish'],
                'fallback': ['volume', 'stoch_k']
            },
            'regime_adaptations': {
                # In trending markets, be more aggressive
                'trending': {
                    'additional_filter': 'higher_highs',
                    'multiplier': 1.2
                },
                # In ranging markets, be more selective
                'ranging': {
                    'additional_filter': 'equal_highs_lows',
                    'multiplier': 0.8
                },
                # In transitions, wait for clear signals
                'transition': {
                    'additional_filter': 'swing_failure',
                    'multiplier': 1.5
                }
            },
            'bonus_multiplier': 1.15,
            'strategy_class': 'adaptive_multi_regime'
        },
        
        'universal_momentum_bear': {
            'description': 'Adaptive momentum capture for multiple market regimes (Bearish)',
            'regime_compatible': ['trending', 'ranging', 'transition'],
            'htf': {
                'primary': ['trend_structure', 'market_structure'],
                'fallback': ['ema_50', 'adx']
            },
            'ttf': {
                'primary': ['momentum_continuation', 'structure_break_bearish'],
                'fallback': ['macd', 'rsi']
            },
            'ltf': {
                'primary': ['volume_breakout', 'momentum_divergence_bearish'],
                'fallback': ['volume', 'stoch_k']
            },
            'regime_adaptations': {
                'trending': {
                    'additional_filter': 'lower_lows',
                    'multiplier': 1.2
                },
                'ranging': {
                    'additional_filter': 'equal_highs_lows',
                    'multiplier': 0.8
                },
                'transition': {
                    'additional_filter': 'swing_failure',
                    'multiplier': 1.5
                }
            },
            'bonus_multiplier': 1.15,
            'strategy_class': 'adaptive_multi_regime'
        },
        
        'structure_breakout_adaptive_bull': {
            'description': 'Breakout strategy that adapts to market regime (Bullish)',
            'regime_compatible': ['trending', 'ranging', 'transition'],
            'htf': {
                'primary': ['equal_highs_lows', 'trend_structure'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['structure_break_bullish', 'false_breakout_bearish'],
                'fallback': ['swing_high', 'higher_highs']
            },
            'ltf': {
                'primary': ['momentum_continuation', 'volume_breakout_confirmation'],
                'fallback': ['rsi', 'volume']
            },
            'regime_adaptations': {
                'trending': {
                    'additional_filter': 'higher_highs',
                    'multiplier': 1.1
                },
                'ranging': {
                    'additional_filter': 'swing_low',
                    'multiplier': 0.9
                },
                'transition': {
                    'additional_filter': 'volume_divergence',
                    'multiplier': 1.3
                }
            },
            'bonus_multiplier': 1.1,
            'strategy_class': 'adaptive_breakout'
        },
        
        'structure_breakout_adaptive_bear': {
            'description': 'Breakout strategy that adapts to market regime (Bearish)',
            'regime_compatible': ['trending', 'ranging', 'transition'],
            'htf': {
                'primary': ['equal_highs_lows', 'trend_structure'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['structure_break_bearish', 'false_breakout_bullish'],
                'fallback': ['swing_low', 'lower_lows']
            },
            'ltf': {
                'primary': ['momentum_continuation', 'volume_breakout_confirmation'],
                'fallback': ['rsi', 'volume']
            },
            'regime_adaptations': {
                'trending': {
                    'additional_filter': 'lower_lows',
                    'multiplier': 1.1
                },
                'ranging': {
                    'additional_filter': 'swing_high',
                    'multiplier': 0.9
                },
                'transition': {
                    'additional_filter': 'volume_divergence',
                    'multiplier': 1.3
                }
            },
            'bonus_multiplier': 1.1,
            'strategy_class': 'adaptive_breakout'
        }
    }


    def discover_mtf_strategies_mode_k(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, MODE_K_SETUPS):
        """
        Mode K: Adaptive Multi-Regime Strategy Discovery
        Strategies that work across multiple market regimes with adaptive parameters
        """
        print(f"  Discovering Mode K (Adaptive Multi-Regime) strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        # Identify current regime contexts
        current_regimes = self._identify_regime_contexts(htf_df, ttf_df, ltf_df)
        
        for setup_name, setup_config in MODE_K_SETUPS.items():
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Test strategy in each compatible regime
            for regime_context in setup_config['regime_compatible']:
                if regime_context not in current_regimes:
                    continue
                
                # Build base mask
                final_mask = pd.Series(True, index=ltf_df.index)
                used_signals = {'htf': [], 'ttf': [], 'ltf': []}
                
                # Process each timeframe
                for tf_label, df, pair_tf_str in [
                    ('htf', htf_df, htf_pair_tf),
                    ('ttf', ttf_df, ttf_pair_tf),
                    ('ltf', ltf_df, ltf_pair_tf)
                ]:
                    signals_config = setup_config[tf_label]
                    
                    for signal_list in [signals_config['primary'], signals_config['fallback']]:
                        signal_found = False
                        for signal in signal_list:
                            if signal in df.columns:
                                states = self.get_or_compute_states(df, signal, pair_tf_str)
                                if states is not None:
                                    states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                    
                                    if direction == 'bullish':
                                        mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                    else:
                                        mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                    
                                    final_mask &= mask
                                    used_signals[tf_label].append(signal)
                                    signal_found = True
                                    break
                        
                        if signal_found:
                            break
                
                # Apply regime-specific additional filter
                regime_adaptation = setup_config['regime_adaptations'].get(regime_context, {})
                if regime_adaptation and 'additional_filter' in regime_adaptation:
                    additional_filter = regime_adaptation['additional_filter']
                    if additional_filter in ltf_df.columns:
                        filter_states = self.get_or_compute_states(ltf_df, additional_filter, ltf_pair_tf)
                        if filter_states is not None:
                            filter_aligned = filter_states.reindex(final_mask.index, fill_value='neutral')
                            if direction == 'bullish':
                                final_mask &= (filter_aligned == 'bullish') | (filter_aligned == 1)
                            else:
                                final_mask &= (filter_aligned == 'bearish') | (filter_aligned == -1)
                
                # Relaxed thresholds for adaptive strategies
                if final_mask.sum() >= 5:
                    aligned_returns = ltf_df.loc[final_mask, 'future_return']
                    
                    if direction == 'bullish':
                        win_rate = (aligned_returns > 0).mean()
                    else:
                        win_rate = (aligned_returns < 0).mean()
                    
                    if win_rate >= 0.48:  # Lower threshold for adaptive strategies
                        regime_multiplier = regime_adaptation.get('multiplier', 1.0)
                        
                        strategies.append({
                            'type': 'mtf_mode_k',
                            'signal_type': "MTF_COMPOSITE",
                            'group': group_name,
                            'pair_tf': f"{pair}_{ltf_tf}",
                            'direction': direction,
                            'trade_direction': 'long' if direction == 'bullish' else 'short',
                            'setup_name': setup_name,
                            'description': setup_config['description'],
                            'regime_context': regime_context,
                            'regime_multiplier': regime_multiplier,
                            'htf_signals': used_signals['htf'],
                            'ttf_signals': used_signals['ttf'],
                            'ltf_signals': used_signals['ltf'],
                            'htf_timeframe': htf_tf,
                            'ttf_timeframe': ttf_tf,
                            'ltf_timeframe': ltf_tf,
                            'discovered_accuracy': win_rate,
                            'sample_size': int(final_mask.sum()),
                            'performance_score': win_rate * setup_config['bonus_multiplier'] * regime_multiplier,
                            'strategy_class': setup_config['strategy_class']
                        })
        
        return strategies

        
    def discover_mtf_strategies_mode_l(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """Mode L (Structure-Aligned) strategies"""
        print(f"  Discovering Mode L (Structure-Aligned) strategies for {group_name}...")

        if htf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)

        strategies = []
        
        # Get structure alignment
        structure_alignment = self.analyze_mtf_structure_alignment(htf_df, ttf_df, ltf_df)
        
        # Only proceed with good alignment
        if structure_alignment['alignment_quality'] in ['poor', 'fair']:
            return strategies
        
        structure_setups = [
            {
                'name': 'perfect_alignment_breakout',
                'description': 'Perfect MTF alignment with breakout confirmation',
                'required_alignment': 'excellent',
                'min_alignment_score': 0.8,
                'signals': {
                    'htf': ['trend_structure'],
                    'ttf': ['structure_break_bullish'],
                    'ltf': ['higher_highs']  # Use actual column name
                }
            },
            {
                'name': 'aligned_pullback_entry',
                'description': 'Aligned pullback across timeframes',
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
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        for setup in structure_setups:
            if (current_alignment_quality == setup['required_alignment'] and 
                current_alignment_score >= setup['min_alignment_score']):
                
                for direction in ['bullish', 'bearish']:
                    # *** FIX: Initialize with LTF index ***
                    signal_mask = pd.Series(True, index=ltf_df.index)
                    
                    for tf, signals in setup['signals'].items():
                        df = htf_df if tf == 'htf' else ttf_df if tf == 'ttf' else ltf_df
                        current_pair_tf = htf_pair_tf if tf == 'htf' else (ttf_pair_tf if tf == 'ttf' else ltf_pair_tf)
                        
                        for signal in signals:
                            if signal in df.columns:
                                states = self.get_or_compute_states(df, signal, current_pair_tf)
                                if states is not None:
                                    # *** FIX: Ensure states align with signal_mask ***
                                    states_aligned = states.reindex(signal_mask.index, fill_value='neutral')
                                    
                                    if direction == 'bullish':
                                        signal_mask &= (states_aligned == 'bullish')
                                    else:
                                        signal_mask &= (states_aligned == 'bearish')
                    
                    if signal_mask.sum() > 5:
                        aligned_returns = ltf_df.loc[signal_mask, 'future_return']
                        
                        if direction == 'bullish':
                            win_rate = (aligned_returns > 0).mean()
                        else:
                            win_rate = (aligned_returns < 0).mean()
                        
                        if win_rate > 0.51:
                            strategies.append({
                                'type': 'mtf_mode_l',
                                'signal_type': "MTF_COMPOSITE",
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
    
    # ============================================================================
    # MODE M: VOLATILITY-ADAPTIVE MTF STRATEGIES
    # ============================================================================
    MODE_M_SETUPS = {
        'high_vol_breakout_bull': {
            'description': 'Breakout momentum for high volatility periods (Bullish)',
            'volatility_regime': 'high_volatility',
            'htf': {
                'primary': ['trend_structure', 'market_structure'],
                'fallback': ['adx', 'atr']
            },
            'ttf': {
                'primary': ['structure_break_bullish', 'momentum_continuation'],
                'fallback': ['higher_highs', 'rsi']
            },
            'ltf': {
                'primary': ['momentum_continuation', 'volume_breakout'],
                'fallback': ['macd', 'volume']
            },
            'adaptive_params': {
                'risk_multiplier': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_ratio': 3.0,
                'position_size': 'reduced'
            },
            'bonus_multiplier': 1.25,
            'strategy_class': 'high_vol_breakout'
        },
        
        'high_vol_breakout_bear': {
            'description': 'Breakout momentum for high volatility periods (Bearish)',
            'volatility_regime': 'high_volatility',
            'htf': {
                'primary': ['trend_structure', 'market_structure'],
                'fallback': ['adx', 'atr']
            },
            'ttf': {
                'primary': ['structure_break_bearish', 'momentum_continuation'],
                'fallback': ['lower_lows', 'rsi']
            },
            'ltf': {
                'primary': ['momentum_continuation', 'volume_breakout'],
                'fallback': ['macd', 'volume']
            },
            'adaptive_params': {
                'risk_multiplier': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_ratio': 3.0,
                'position_size': 'reduced'
            },
            'bonus_multiplier': 1.25,
            'strategy_class': 'high_vol_breakout'
        },
        
        'high_vol_range_expansion_bull': {
            'description': 'Range expansion plays in high volatility (Bullish)',
            'volatility_regime': 'high_volatility',
            'htf': {
                'primary': ['market_structure', 'equal_highs_lows'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['equal_highs_lows', 'false_breakout_bearish'],
                'fallback': ['swing_low', 'rsi']
            },
            'ltf': {
                'primary': ['structure_break_bullish', 'volume_breakout_confirmation'],
                'fallback': ['momentum_continuation', 'volume']
            },
            'adaptive_params': {
                'risk_multiplier': 0.5,
                'stop_loss_pct': 0.04,
                'take_profit_ratio': 2.5,
                'position_size': 'minimal'
            },
            'bonus_multiplier': 1.2,
            'strategy_class': 'high_vol_range'
        },
        
        'high_vol_range_expansion_bear': {
            'description': 'Range expansion plays in high volatility (Bearish)',
            'volatility_regime': 'high_volatility',
            'htf': {
                'primary': ['market_structure', 'equal_highs_lows'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['equal_highs_lows', 'false_breakout_bullish'],
                'fallback': ['swing_high', 'rsi']
            },
            'ltf': {
                'primary': ['structure_break_bearish', 'volume_breakout_confirmation'],
                'fallback': ['momentum_continuation', 'volume']
            },
            'adaptive_params': {
                'risk_multiplier': 0.5,
                'stop_loss_pct': 0.04,
                'take_profit_ratio': 2.5,
                'position_size': 'minimal'
            },
            'bonus_multiplier': 1.2,
            'strategy_class': 'high_vol_range'
        },
        
        'low_vol_compression_bull': {
            'description': 'Breakout from low volatility compression (Bullish)',
            'volatility_regime': 'low_volatility',
            'htf': {
                'primary': ['market_structure', 'bb_squeeze'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['bb_squeeze', 'equal_highs_lows'],
                'fallback': ['swing_low', 'adx']
            },
            'ltf': {
                'primary': ['structure_break_bullish', 'volume_breakout'],
                'fallback': ['momentum_continuation', 'rsi']
            },
            'adaptive_params': {
                'risk_multiplier': 1.2,
                'stop_loss_pct': 0.015,
                'take_profit_ratio': 4.0,
                'position_size': 'normal'
            },
            'bonus_multiplier': 1.3,
            'strategy_class': 'low_vol_compression'
        },
        
        'low_vol_compression_bear': {
            'description': 'Breakdown from low volatility compression (Bearish)',
            'volatility_regime': 'low_volatility',
            'htf': {
                'primary': ['market_structure', 'bb_squeeze'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['bb_squeeze', 'equal_highs_lows'],
                'fallback': ['swing_high', 'adx']
            },
            'ltf': {
                'primary': ['structure_break_bearish', 'volume_breakout'],
                'fallback': ['momentum_continuation', 'rsi']
            },
            'adaptive_params': {
                'risk_multiplier': 1.2,
                'stop_loss_pct': 0.015,
                'take_profit_ratio': 4.0,
                'position_size': 'normal'
            },
            'bonus_multiplier': 1.3,
            'strategy_class': 'low_vol_compression'
        },
        
        'low_vol_mean_reversion_bull': {
            'description': 'Mean reversion in low volatility ranges (Bullish)',
            'volatility_regime': 'low_volatility',
            'htf': {
                'primary': ['equal_highs_lows', 'market_structure'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['rsi_extreme', 'swing_low'],
                'fallback': ['rsi', 'bb_lower']
            },
            'ltf': {
                'primary': ['momentum_divergence_bullish', 'false_breakout_bearish'],
                'fallback': ['rsi', 'stoch_k']
            },
            'adaptive_params': {
                'risk_multiplier': 1.0,
                'stop_loss_pct': 0.01,
                'take_profit_ratio': 1.5,
                'position_size': 'normal'
            },
            'bonus_multiplier': 1.15,
            'strategy_class': 'low_vol_mean_reversion'
        },
        
        'low_vol_mean_reversion_bear': {
            'description': 'Mean reversion in low volatility ranges (Bearish)',
            'volatility_regime': 'low_volatility',
            'htf': {
                'primary': ['equal_highs_lows', 'market_structure'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['rsi_extreme', 'swing_high'],
                'fallback': ['rsi', 'bb_upper']
            },
            'ltf': {
                'primary': ['momentum_divergence_bearish', 'false_breakout_bullish'],
                'fallback': ['rsi', 'stoch_k']
            },
            'adaptive_params': {
                'risk_multiplier': 1.0,
                'stop_loss_pct': 0.01,
                'take_profit_ratio': 1.5,
                'position_size': 'normal'
            },
            'bonus_multiplier': 1.15,
            'strategy_class': 'low_vol_mean_reversion'
        },
        
        'normal_vol_trend_bull': {
            'description': 'Standard trend following in normal volatility (Bullish)',
            'volatility_regime': 'normal_volatility',
            'htf': {
                'primary': ['trend_structure', 'higher_highs'],
                'fallback': ['ema_50', 'adx']
            },
            'ttf': {
                'primary': ['pullback_complete_bull', 'healthy_bull_pullback'],
                'fallback': ['swing_low', 'rsi']
            },
            'ltf': {
                'primary': ['momentum_confirmation', 'structure_break_bullish'],
                'fallback': ['macd', 'momentum_continuation']
            },
            'adaptive_params': {
                'risk_multiplier': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_ratio': 2.0,
                'position_size': 'normal'
            },
            'bonus_multiplier': 1.1,
            'strategy_class': 'normal_vol_trend'
        },
        
        'normal_vol_trend_bear': {
            'description': 'Standard trend following in normal volatility (Bearish)',
            'volatility_regime': 'normal_volatility',
            'htf': {
                'primary': ['trend_structure', 'lower_lows'],
                'fallback': ['ema_50', 'adx']
            },
            'ttf': {
                'primary': ['pullback_complete_bear', 'healthy_bear_pullback'],
                'fallback': ['swing_high', 'rsi']
            },
            'ltf': {
                'primary': ['momentum_confirmation', 'structure_break_bearish'],
                'fallback': ['macd', 'momentum_continuation']
            },
            'adaptive_params': {
                'risk_multiplier': 1.0,
                'stop_loss_pct': 0.02,
                'take_profit_ratio': 2.0,
                'position_size': 'normal'
            },
            'bonus_multiplier': 1.1,
            'strategy_class': 'normal_vol_trend'
        }
    }

    def discover_mtf_strategies_mode_m(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, MODE_M_SETUPS):
        """
        Mode M: Volatility-Adaptive MTF Strategies
        Strategies optimized for specific volatility regimes
        """
        print(f"  Discovering Mode M (Volatility-Adaptive) strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        # Analyze current volatility regime
        current_vol_regime = self._analyze_volatility_characteristics(ltf_df)['volatility_regime']
        
        for setup_name, setup_config in MODE_M_SETUPS.items():
            # Only test strategies for the current volatility regime
            if setup_config['volatility_regime'] != current_vol_regime:
                continue
            
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Build mask
            final_mask = pd.Series(True, index=ltf_df.index)
            used_signals = {'htf': [], 'ttf': [], 'ltf': []}
            
            for tf_label, df, pair_tf_str in [
                ('htf', htf_df, htf_pair_tf),
                ('ttf', ttf_df, ttf_pair_tf),
                ('ltf', ltf_df, ltf_pair_tf)
            ]:
                signals_config = setup_config[tf_label]
                
                for signal_list in [signals_config['primary'], signals_config['fallback']]:
                    signal_found = False
                    for signal in signal_list:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, pair_tf_str)
                            if states is not None:
                                states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                
                                # Special handling for certain indicators
                                if signal == 'rsi_extreme':
                                    if direction == 'bullish':
                                        mask = df['rsi'].reindex(final_mask.index) < 30
                                    else:
                                        mask = df['rsi'].reindex(final_mask.index) > 70
                                elif signal in ['bb_squeeze', 'equal_highs_lows']:
                                    mask = (states_aligned == 1) | (states_aligned == 'bullish')
                                else:
                                    if direction == 'bullish':
                                        mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                    else:
                                        mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                
                                final_mask &= mask
                                used_signals[tf_label].append(signal)
                                signal_found = True
                                break
                    
                    if signal_found:
                        break
            
            # Volatility-specific thresholds
            vol_thresholds = {
                'high_volatility': (5, 0.52),    # Higher threshold for risky conditions
                'normal_volatility': (5, 0.50),  # Standard threshold
                'low_volatility': (5, 0.48)      # Lower threshold for compression plays
            }
            
            min_samples, min_win_rate = vol_thresholds.get(current_vol_regime, (5, 0.50))
            
            if final_mask.sum() >= min_samples:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate >= min_win_rate:
                    strategies.append({
                        'type': 'mtf_mode_m',
                        'signal_type': "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'description': setup_config['description'],
                        'volatility_regime': current_vol_regime,
                        'adaptive_parameters': setup_config['adaptive_params'],
                        'htf_signals': used_signals['htf'],
                        'ttf_signals': used_signals['ttf'],
                        'ltf_signals': used_signals['ltf'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(final_mask.sum()),
                        'performance_score': win_rate * setup_config['bonus_multiplier'],
                        'strategy_class': setup_config['strategy_class']
                    })
        
        return strategies
    # ============================================================================
    # MODE N: MOMENTUM CASCADE STRATEGIES
    # ============================================================================
    MODE_N_SETUPS = {
        'momentum_cascade_bull': {
            'description': 'Bullish momentum flowing from HTF → TTF → LTF',
            'required_flow': 'htf_to_ltf',
            'momentum_strength': 'strong',
            'htf': {
                'primary': ['trend_structure', 'higher_highs'],
                'fallback': ['ema_50', 'adx']
            },
            'ttf': {
                'primary': ['momentum_continuation', 'pullback_complete_bull'],
                'fallback': ['macd', 'rsi']
            },
            'ltf': {
                'primary': ['structure_break_bullish', 'momentum_divergence_bullish'],
                'fallback': ['momentum_continuation', 'volume_breakout']
            },
            'cascade_requirements': {
                'htf_momentum': 'bullish',
                'ttf_momentum': 'bullish',
                'ltf_momentum': 'bullish'
            },
            'bonus_multiplier': 1.2,
            'strategy_class': 'momentum_cascade'
        },
        
        'momentum_cascade_bear': {
            'description': 'Bearish momentum flowing from HTF → TTF → LTF',
            'required_flow': 'htf_to_ltf',
            'momentum_strength': 'strong',
            'htf': {
                'primary': ['trend_structure', 'lower_lows'],
                'fallback': ['ema_50', 'adx']
            },
            'ttf': {
                'primary': ['momentum_continuation', 'pullback_complete_bear'],
                'fallback': ['macd', 'rsi']
            },
            'ltf': {
                'primary': ['structure_break_bearish', 'momentum_divergence_bearish'],
                'fallback': ['momentum_continuation', 'volume_breakout']
            },
            'cascade_requirements': {
                'htf_momentum': 'bearish',
                'ttf_momentum': 'bearish',
                'ltf_momentum': 'bearish'
            },
            'bonus_multiplier': 1.2,
            'strategy_class': 'momentum_cascade'
        },
        
        'momentum_acceleration_bull': {
            'description': 'Momentum accelerating across timeframes (Bullish)',
            'required_flow': 'accelerating',
            'momentum_strength': 'increasing',
            'htf': {
                'primary': ['trend_structure', 'market_structure'],
                'fallback': ['adx', 'ema_50']
            },
            'ttf': {
                'primary': ['momentum_continuation', 'structure_break_bullish'],
                'fallback': ['macd', 'momentum_divergence_bullish']
            },
            'ltf': {
                'primary': ['volume_breakout', 'momentum_continuation'],
                'fallback': ['volume_breakout_confirmation', 'rsi']
            },
            'cascade_requirements': {
                'htf_momentum': 'bullish',
                'ttf_momentum': 'bullish',
                'ltf_momentum': 'bullish'
            },
            'bonus_multiplier': 1.15,
        'strategy_class': 'adaptive_multi_regime'
    },
    
    'universal_momentum_bear': {
        'description': 'Adaptive momentum capture for multiple market regimes (Bearish)',
        'regime_compatible': ['trending', 'ranging', 'transition'],
        'htf': {
            'primary': ['trend_structure', 'market_structure'],
            'fallback': ['ema_50', 'adx']
        },
        'ttf': {
            'primary': ['momentum_continuation', 'structure_break_bearish'],
            'fallback': ['macd', 'rsi']
        },
        'ltf': {
            'primary': ['volume_breakout', 'momentum_divergence_bearish'],
            'fallback': ['volume', 'stoch_k']
        },
        'regime_adaptations': {
            'trending': {
                'additional_filter': 'lower_lows',
                'multiplier': 1.2
            },
            'ranging': {
                'additional_filter': 'equal_highs_lows',
                'multiplier': 0.8
            },
            'transition': {
                'additional_filter': 'swing_failure',
                'multiplier': 1.5
            }
        },
        'bonus_multiplier': 1.15,
        'strategy_class': 'adaptive_multi_regime'
        },
        
        'structure_breakout_adaptive_bull': {
            'description': 'Breakout strategy that adapts to market regime (Bullish)',
            'regime_compatible': ['trending', 'ranging', 'transition'],
            'htf': {
                'primary': ['equal_highs_lows', 'trend_structure'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['structure_break_bullish', 'false_breakout_bearish'],
                'fallback': ['swing_high', 'higher_highs']
            },
            'ltf': {
                'primary': ['momentum_continuation', 'volume_breakout_confirmation'],
                'fallback': ['rsi', 'volume']
            },
            'regime_adaptations': {
                'trending': {
                    'additional_filter': 'higher_highs',
                    'multiplier': 1.1
                },
                'ranging': {
                    'additional_filter': 'swing_low',
                    'multiplier': 0.9
                },
                'transition': {
                    'additional_filter': 'volume_divergence',
                    'multiplier': 1.3
                }
            },
            'bonus_multiplier': 1.1,
            'strategy_class': 'adaptive_breakout'
        },
        
        'structure_breakout_adaptive_bear': {
            'description': 'Breakout strategy that adapts to market regime (Bearish)',
            'regime_compatible': ['trending', 'ranging', 'transition'],
            'htf': {
                'primary': ['equal_highs_lows', 'trend_structure'],
                'fallback': ['bb_width', 'atr']
            },
            'ttf': {
                'primary': ['structure_break_bearish', 'false_breakout_bullish'],
                'fallback': ['swing_low', 'lower_lows']
            },
            'ltf': {
                'primary': ['momentum_continuation', 'volume_breakout_confirmation'],
                'fallback': ['rsi', 'volume']
            },
            'regime_adaptations': {
                'trending': {
                    'additional_filter': 'lower_lows',
                    'multiplier': 1.1
                },
                'ranging': {
                    'additional_filter': 'swing_high',
                    'multiplier': 0.9
                },
                'transition': {
                    'additional_filter': 'volume_divergence',
                    'multiplier': 1.3
                }
            },
            'bonus_multiplier': 1.1,
            'strategy_class': 'adaptive_breakout'
        }
    }
    def discover_mtf_strategies_mode_n(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, MODE_N_SETUPS):
        """
        Mode N: Momentum Cascade Strategies
        Identifies momentum flowing from HTF → TTF → LTF with confirmation
        """
        print(f"  Discovering Mode N (Momentum Cascade) strategies for {group_name}...")
        
        if htf_df is None or ttf_df is None or ltf_df is None:
            return []
        
        # Enhance dataframes
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        # Analyze momentum cascade
        momentum_cascade = self._analyze_momentum_cascade(htf_df, ttf_df, ltf_df)
        
        current_flow = momentum_cascade.get('primary_flow_direction', 'unknown')
        current_strength = momentum_cascade.get('momentum_strength', 'unknown')
        
        for setup_name, setup_config in MODE_N_SETUPS.items():
            # Check if current momentum matches required pattern
            required_flow = setup_config['required_flow']
            required_strength = setup_config['momentum_strength']
            
            # Flexible matching for flow patterns
            flow_matches = (
                (required_flow == 'htf_to_ltf' and current_flow in ['htf_to_ltf', 'cascading']) or
                (required_flow == 'accelerating' and current_strength == 'increasing') or
                (required_flow == 'reversal' and current_flow == 'reversal')
            )
            
            if not flow_matches:
                continue
            
            direction = 'bullish' if 'bull' in setup_name else 'bearish'
            
            # Build mask
            final_mask = pd.Series(True, index=ltf_df.index)
            used_signals = {'htf': [], 'ttf': [], 'ltf': []}
            
            for tf_label, df, pair_tf_str in [
                ('htf', htf_df, htf_pair_tf),
                ('ttf', ttf_df, ttf_pair_tf),
                ('ltf', ltf_df, ltf_pair_tf)
            ]:
                signals_config = setup_config[tf_label]
                
                for signal_list in [signals_config['primary'], signals_config['fallback']]:
                    signal_found = False
                    for signal in signal_list:
                        if signal in df.columns:
                            states = self.get_or_compute_states(df, signal, pair_tf_str)
                            if states is not None:
                                states_aligned = states.reindex(final_mask.index, fill_value='neutral')
                                
                                if direction == 'bullish':
                                    mask = (states_aligned == 'bullish') | (states_aligned == 1)
                                else:
                                    mask = (states_aligned == 'bearish') | (states_aligned == -1)
                                
                                final_mask &= mask
                                used_signals[tf_label].append(signal)
                                signal_found = True
                                break
                    
                    if signal_found:
                        break
            
            # Verify cascade requirements (momentum alignment across TFs)
            cascade_reqs = setup_config['cascade_requirements']
            cascade_valid = True
            
            # Check HTF momentum
            htf_mom_req = cascade_reqs.get('htf_momentum', 'bullish')
            htf_mom_actual = momentum_cascade.get('htf_momentum', 'neutral')
            if htf_mom_req != 'neutral' and htf_mom_actual != htf_mom_req:
                cascade_valid = False
            
            # Check TTF momentum
            ttf_mom_req = cascade_reqs.get('ttf_momentum', 'bullish')
            ttf_mom_actual = momentum_cascade.get('ttf_momentum', 'neutral')
            if ttf_mom_req != 'neutral' and ttf_mom_actual != ttf_mom_req:
                cascade_valid = False
            
            # Check LTF momentum
            ltf_mom_req = cascade_reqs.get('ltf_momentum', 'bullish')
            ltf_mom_actual = momentum_cascade.get('ltf_momentum', 'neutral')
            if ltf_mom_req != 'neutral' and ltf_mom_actual != ltf_mom_req:
                cascade_valid = False
            
            if not cascade_valid:
                continue
            
            # Relaxed thresholds for momentum cascade
            if final_mask.sum() >= 5:
                aligned_returns = ltf_df.loc[final_mask, 'future_return']
                
                if direction == 'bullish':
                    win_rate = (aligned_returns > 0).mean()
                else:
                    win_rate = (aligned_returns < 0).mean()
                
                if win_rate >= 0.48:  # Lower threshold for cascade strategies
                    cascade_score = momentum_cascade.get('cascade_score', 50)
                    
                    strategies.append({
                        'type': 'mtf_mode_n',
                        'signal_type': "MTF_COMPOSITE",
                        'group': group_name,
                        'pair_tf': f"{pair}_{ltf_tf}",
                        'direction': direction,
                        'trade_direction': 'long' if direction == 'bullish' else 'short',
                        'setup_name': setup_name,
                        'description': setup_config['description'],
                        'momentum_flow': current_flow,
                        'momentum_strength': current_strength,
                        'cascade_score': cascade_score,
                        'htf_signals': used_signals['htf'],
                        'ttf_signals': used_signals['ttf'],
                        'ltf_signals': used_signals['ltf'],
                        'htf_timeframe': htf_tf,
                        'ttf_timeframe': ttf_tf,
                        'ltf_timeframe': ltf_tf,
                        'discovered_accuracy': win_rate,
                        'sample_size': int(final_mask.sum()),
                        'performance_score': win_rate * setup_config['bonus_multiplier'] * (1 + cascade_score / 100),
                        'strategy_class': setup_config['strategy_class']
                    })
        
        return strategies
    
    def discover_mtf_strategies_mode_combo(self, group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf):
        """
        Mode COMBO: Dynamically tests combinations of top-performing indicators
        Finds best 2-3 indicator combinations per timeframe
        """
        print(f"  Discovering Mode COMBO (Dynamic Combinations) for {group_name}...")
        
        if htf_df is None:
            return []
        
        htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
        ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
        ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        strategies = []
        
        # === STEP 1: Identify top-performing single indicators per timeframe ===
        def get_top_indicators(df, pair_tf, n=5):
            """Returns top N indicators by individual win rate"""
            indicator_cols = self.categorize_columns(df)['indicators'][:20]  # Test top 20
            
            results = []
            for ind in indicator_cols:
                if ind not in df.columns:
                    continue
                
                states = self.get_or_compute_states(df, ind, pair_tf)
                if states is None:
                    continue
                
                # Test bullish
                bull_mask = (states == 'bullish')
                if bull_mask.sum() > 20:
                    bull_returns = df.loc[bull_mask, 'future_return']
                    bull_wr = (bull_returns > 0).mean()
                    results.append(('bullish', ind, bull_wr, bull_mask.sum()))
                
                # Test bearish
                bear_mask = (states == 'bearish')
                if bear_mask.sum() > 20:
                    bear_returns = df.loc[bear_mask, 'future_return']
                    bear_wr = (bear_returns < 0).mean()
                    results.append(('bearish', ind, bear_wr, bear_mask.sum()))
            
            # Sort by win rate, return top N
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:n]
        
        htf_pair_tf = f"{pair}_{htf_tf}"
        ttf_pair_tf = f"{pair}_{ttf_tf}"
        ltf_pair_tf = f"{pair}_{ltf_tf}"
        
        htf_top = get_top_indicators(htf_df, htf_pair_tf, n=5)
        ttf_top = get_top_indicators(ttf_df, ttf_pair_tf, n=5)
        ltf_top = get_top_indicators(ltf_df, ltf_pair_tf, n=5)
        
        if not htf_top or not ttf_top or not ltf_top:
            return strategies
        
        # === STEP 2: Test 2-3 indicator combinations ===
        from itertools import combinations
        
        for direction in ['bullish', 'bearish']:
            # Get indicators for this direction
            htf_inds = [ind for dir, ind, wr, _ in htf_top if dir == direction][:3]
            ttf_inds = [ind for dir, ind, wr, _ in ttf_top if dir == direction][:3]
            ltf_inds = [ind for dir, ind, wr, _ in ltf_top if dir == direction][:3]
            
            if len(htf_inds) < 2 or len(ttf_inds) < 2 or len(ltf_inds) < 2:
                continue
            
            # Test 2-indicator combinations on each timeframe
            for htf_combo in combinations(htf_inds, 2):
                for ttf_combo in combinations(ttf_inds, 2):
                    for ltf_combo in combinations(ltf_inds, 2):
                        
                        # Build combined mask
                        htf_mask = pd.Series(True, index=htf_df.index)
                        for ind in htf_combo:
                            states = self.get_or_compute_states(htf_df, ind, htf_pair_tf)
                            if states is not None:
                                htf_mask &= (states == direction)
                        
                        ttf_mask = pd.Series(True, index=ttf_df.index)
                        for ind in ttf_combo:
                            states = self.get_or_compute_states(ttf_df, ind, ttf_pair_tf)
                            if states is not None:
                                ttf_mask &= (states == direction)
                        
                        ltf_mask = pd.Series(True, index=ltf_df.index)
                        for ind in ltf_combo:
                            states = self.get_or_compute_states(ltf_df, ind, ltf_pair_tf)
                            if states is not None:
                                ltf_mask &= (states == direction)
                        
                        # Final mask: All three timeframes must agree
                        final_mask = htf_mask & ttf_mask & ltf_mask
                        
                        if final_mask.sum() > 15:
                            returns = ltf_df.loc[final_mask, 'future_return']
                            
                            if direction == 'bullish':
                                win_rate = (returns > 0).mean()
                            else:
                                win_rate = (returns < 0).mean()
                            
                            # Higher threshold for combo strategies
                            if win_rate > 0.58:
                                strategies.append({
                                    'type': 'mtf_mode_combo',
                                    'signal_type': "MTF_COMPOSITE",
                                    'group': group_name,
                                    'pair_tf': f"{pair}_{ltf_tf}",
                                    'direction': direction,
                                    'trade_direction': 'long' if direction == 'bullish' else 'short',
                                    'htf_indicators': list(htf_combo),
                                    'ttf_indicators': list(ttf_combo),
                                    'ltf_indicators': list(ltf_combo),
                                    'htf_timeframe': htf_tf,
                                    'ttf_timeframe': ttf_tf,
                                    'ltf_timeframe': ltf_tf,
                                    'discovered_accuracy': win_rate,
                                    'sample_size': int(final_mask.sum()),
                                    'performance_score': win_rate * 1.1,  # Bonus for combo
                                    'strategy_class': 'dynamic_combo'
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
        
    def _filter_and_rank_strategies(self, strategies):
            """
            Post-discovery filter to keep only high-quality strategies
            """
            print("\n🔍 Filtering strategies for quality...")
            
            filtered = []
            
            for strategy in strategies:
                # Skip if too few samples
                if strategy.get('sample_size', 0) < 15:
                    continue
                
                # Skip if win rate too low
                if strategy.get('discovered_accuracy', 0) < 0.52:
                    continue
                
                # Calculate quality score if not already present
                if 'quality_score' not in strategy:
                    strategy['quality_score'] = strategy.get('performance_score', 0) * 100
                
                # Keep if quality score >= 50
                if strategy['quality_score'] >= 50:
                    filtered.append(strategy)
            
            # Sort by performance score
            filtered.sort(key=lambda x: x.get('performance_score', 0), reverse=True)
            
            print(f"  Kept {len(filtered)}/{len(strategies)} strategies after filtering")
            
            return filtered

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
                ('COMBO', self.discover_mtf_strategies_mode_combo),
                ('Confluence', self.discover_mtf_strategies_mode_confluence)
            ]:                
                try:
                    # Modes K, M, N need their setup configs passed
                    if mode_label == 'K':
                        res = mode_fn(group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, self.MODE_K_SETUPS)
                    elif mode_label == 'M':
                        res = mode_fn(group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, self.MODE_M_SETUPS)
                    elif mode_label == 'N':
                        res = mode_fn(group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf, self.MODE_N_SETUPS)
                    else:
                        res = mode_fn(group_name, pair, htf_df, ttf_df, ltf_df, htf_tf, ttf_tf, ltf_tf)
                    
                    strategy_results[mode_label] = res if res is not None else []
                except Exception as e:
                    print(f"  ! Mode {mode_label} failed for {group_name}: {e}")
                    #print(f"  ! Mode {mode_label} failed for {group_name}: {e}\n{traceback.format_exc()}") # Added full traceback
                    strategy_results[mode_label] = []

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

        
        
        # Filter for quality
        all_mtf_strategies = self._filter_and_rank_strategies(all_mtf_strategies)

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
    
    def _debug_pattern_availability(self, df, pattern_list, pair_tf):
        """Debug helper: Check which patterns exist and have non-zero values"""
        print(f"\n  🔍 DEBUG: Checking pattern availability for {pair_tf}")
        
        for pattern in pattern_list:
            if pattern in df.columns:
                count = (df[pattern] != 0).sum()
                states = self.get_or_compute_states(df, pattern, pair_tf)
                
                if states is not None:
                    bullish = (states == 'bullish').sum()
                    bearish = (states == 'bearish').sum()
                    print(f"    ✓ {pattern}: {count} non-zero | {bullish} bullish | {bearish} bearish")
                else:
                    print(f"    ✗ {pattern}: EXISTS but NOT MAPPED!")
            else:
                print(f"    ✗ {pattern}: NOT FOUND in DataFrame!")


    def diagnose_mode_issues(self, group_name, pair, htf_df, ttf_df, ltf_df):
        """
        Diagnostic tool to understand why modes aren't finding strategies
        """
        print(f"\n🔍 DIAGNOSING MODE ISSUES FOR {group_name}")
        print("="*60)
        
        for tf_name, df in [('HTF', htf_df), ('TTF', ttf_df), ('LTF', ltf_df)]:
            print(f"\n{tf_name} ({len(df)} bars):")
            
            # Check critical patterns
            critical_patterns = [
                'higher_highs', 'lower_lows', 'trend_structure', 'market_structure',
                'momentum_divergence_bullish', 'momentum_divergence_bearish',
                'structure_break_bullish', 'structure_break_bearish',
                'volume_breakout', 'momentum_continuation'
            ]
            
            found = 0
            missing = 0
            empty = 0
            
            for pattern in critical_patterns:
                if pattern in df.columns:
                    count = (df[pattern] != 0).sum() if df[pattern].dtype != 'object' else len(df[pattern].unique())
                    if count > 0:
                        print(f"  ✅ {pattern}: {count} occurrences")
                        found += 1
                    else:
                        print(f"  ⚠️  {pattern}: EXISTS but EMPTY")
                        empty += 1
                else:
                    print(f"  ❌ {pattern}: MISSING")
                    missing += 1
            
            print(f"\n  Summary: {found} working, {empty} empty, {missing} missing")
            
            # Check index alignment
            print(f"  Index range: {df.index[0]} to {df.index[-1]}")
            print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            # Check for NaN issues
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                print(f"  ⚠️  Columns with NaN: {', '.join(nan_cols[:5])}")

    # ============================================================================
    # HELPER FUNCTIONS (Add to PatternsMixin or DiscoveryModesMixin)
    # ============================================================================

    def _identify_regime_contexts(self, htf_df, ttf_df, ltf_df):
        """
        Identify which regime contexts are present in the data
        Returns: list of regime contexts ['trending', 'ranging', 'transition']
        """
        regimes_found = set()
        
        for df in [htf_df, ttf_df, ltf_df]:
            if 'market_structure' in df.columns:
                # Get last 20 bars to see current regime
                recent_structure = df['market_structure'].tail(20)
                
                if (recent_structure == 'strong_trend').sum() > 10:
                    regimes_found.add('trending')
                if (recent_structure == 'trending').sum() > 10:
                    regimes_found.add('trending')
                if (recent_structure == 'ranging').sum() > 10:
                    regimes_found.add('ranging')
            
            # Check for transitions (changing structure)
            if 'trend_structure' in df.columns:
                recent_trend = df['trend_structure'].tail(20)
                unique_trends = recent_trend.nunique()
                if unique_trends >= 3:  # Multiple trend states = transition
                    regimes_found.add('transition')
        
        # If nothing found, assume all are possible
        if not regimes_found:
            regimes_found = {'trending', 'ranging', 'transition'}
        
        return list(regimes_found)


    def _analyze_volatility_characteristics(self, df):
        """
        Analyze volatility regime of the dataframe
        Returns: dict with volatility_regime ('high', 'normal', 'low')
        """
        volatility_analysis = {
            'volatility_regime': 'normal_volatility',
            'volatility_score': 50
        }
        
        if 'atr' not in df.columns or len(df) < 50:
            return volatility_analysis
        
        # Calculate ATR percentile
        atr_values = df['atr'].tail(50)
        current_atr = atr_values.iloc[-1]
        atr_median = atr_values.median()
        atr_std = atr_values.std()
        
        # Normalize ATR by price
        current_price = df['close'].iloc[-1]
        normalized_atr = current_atr / current_price
        normalized_median = atr_median / current_price
        normalized_std = atr_std / current_price
        
        # Classify volatility
        if normalized_atr > normalized_median + normalized_std:
            volatility_analysis['volatility_regime'] = 'high_volatility'
            volatility_analysis['volatility_score'] = 75
        elif normalized_atr < normalized_median - normalized_std:
            volatility_analysis['volatility_regime'] = 'low_volatility'
            volatility_analysis['volatility_score'] = 25
        else:
            volatility_analysis['volatility_regime'] = 'normal_volatility'
            volatility_analysis['volatility_score'] = 50
        
        return volatility_analysis


    def _analyze_momentum_cascade(self, htf_df, ttf_df, ltf_df):
        """
        Analyze momentum flow across timeframes
        Returns: dict with momentum cascade characteristics
        """
        cascade_analysis = {
            'primary_flow_direction': 'unknown',
            'momentum_strength': 'unknown',
            'htf_momentum': 'neutral',
            'ttf_momentum': 'neutral',
            'ltf_momentum': 'neutral',
            'cascade_score': 0
        }
        
        def get_momentum_direction(df):
            """Determine momentum direction for a single timeframe"""
            if 'rsi' not in df.columns or len(df) < 20:
                return 'neutral'
            
            rsi_values = df['rsi'].tail(20)
            rsi_trend = rsi_values.mean()
            rsi_recent = rsi_values.tail(5).mean()
            
            # Check if momentum is bullish, bearish, or neutral
            if rsi_recent > 55 and rsi_trend > 50:
                return 'bullish'
            elif rsi_recent < 45 and rsi_trend < 50:
                return 'bearish'
            else:
                return 'neutral'
        
        # Get momentum for each timeframe
        cascade_analysis['htf_momentum'] = get_momentum_direction(htf_df)
        cascade_analysis['ttf_momentum'] = get_momentum_direction(ttf_df)
        cascade_analysis['ltf_momentum'] = get_momentum_direction(ltf_df)
        
        # Determine cascade pattern
        htf_mom = cascade_analysis['htf_momentum']
        ttf_mom = cascade_analysis['ttf_momentum']
        ltf_mom = cascade_analysis['ltf_momentum']
        
        # Perfect cascade: All aligned
        if htf_mom == ttf_mom == ltf_mom and htf_mom != 'neutral':
            cascade_analysis['primary_flow_direction'] = 'htf_to_ltf'
            cascade_analysis['momentum_strength'] = 'strong'
            cascade_analysis['cascade_score'] = 90
        
        # Good cascade: HTF and TTF aligned
        elif htf_mom == ttf_mom and htf_mom != 'neutral':
            cascade_analysis['primary_flow_direction'] = 'htf_to_ltf'
            cascade_analysis['momentum_strength'] = 'moderate'
            cascade_analysis['cascade_score'] = 70
        
        # Accelerating: TTF and LTF aligned, stronger than HTF
        elif ttf_mom == ltf_mom and ttf_mom != 'neutral' and htf_mom == 'neutral':
            cascade_analysis['primary_flow_direction'] = 'accelerating'
            cascade_analysis['momentum_strength'] = 'increasing'
            cascade_analysis['cascade_score'] = 75
        
        # Reversal: HTF opposite to TTF/LTF
        elif htf_mom != 'neutral' and ttf_mom != 'neutral' and htf_mom != ttf_mom:
            cascade_analysis['primary_flow_direction'] = 'reversal'
            cascade_analysis['momentum_strength'] = 'reversing'
            cascade_analysis['cascade_score'] = 65
        
        # Weak/No cascade
        else:
            cascade_analysis['primary_flow_direction'] = 'mixed'
            cascade_analysis['momentum_strength'] = 'weak'
            cascade_analysis['cascade_score'] = 40
        
        return cascade_analysis