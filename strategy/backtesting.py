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

class BacktestingMixin:
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 1: MTF Backtesting Functions (MODES A, B, C)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def backtest_mtf_strategy(self, strategy):
        """
        Enhanced backtesting for legacy MTF strategies (Modes A, B, C)
        FIXED: Now correctly calculates avg_win, avg_loss, and profit_factor.
        FIXED: Passes 'pair_tf' to get_or_compute_states.
        FIXED: Dynamically gets 'pair' from strategy.
        """
        if not strategy['type'].startswith('mtf_'):
            return strategy
        
        # --- START OF FIX: DYNAMIC PAIR ---
        # Get pair from strategy (e.g., 'btc_usdt_1m' -> 'btc_usdt')
        pair_tf = strategy.get('pair_tf', 'btc_usdt_1m')
        pair = "_".join(pair_tf.split('_')[:2])
        # --- END OF FIX ---

        # Get the aligned dataframes
        htf_df, ttf_df, ltf_df = self.get_mtf_dataframes(
            pair,
            strategy['htf_timeframe'],
            strategy['ttf_timeframe'], 
            strategy['ltf_timeframe']
        )
        
        if htf_df is None:
            return strategy
        
        # --- START OF FIX: PASS pair_tf ---
        htf_pair_tf = f"{pair}_{strategy['htf_timeframe']}"
        ttf_pair_tf = f"{pair}_{strategy['ttf_timeframe']}"
        ltf_pair_tf = f"{pair}_{strategy['ltf_timeframe']}"

        htf_states = self.get_or_compute_states(htf_df, strategy['htf_indicator'], htf_pair_tf)
        ttf_states = self.get_or_compute_states(ttf_df, strategy['ttf_indicator'], ttf_pair_tf)
        ltf_states = self.get_or_compute_states(ltf_df, strategy['ltf_indicator'], ltf_pair_tf)
        # --- END OF FIX ---
        
        if any(states is None for states in [htf_states, ttf_states, ltf_states]):
            return strategy
        
        active_mask = pd.Series(False, index=ltf_df.index) # Default empty mask
        
        # Apply the specific MTF mode logic
        if strategy['type'] == 'mtf_mode_a':
            if strategy['direction'] == 'bullish':
                active_mask = (
                    (htf_states == 'bullish') & 
                    (ttf_states == 'bullish') & 
                    (ltf_states == 'bullish')
                )
            else:
                active_mask = (
                    (htf_states == 'bearish') & 
                    (ttf_states == 'bearish') & 
                    (ltf_states == 'bearish')
                )
        
        elif strategy['type'] == 'mtf_mode_b':
            confluence_type = strategy['confluence_type']
            htf_bull = htf_states == 'bullish'
            ttf_bull = ttf_states == 'bullish' 
            ltf_bull = ltf_states == 'bullish'
            htf_bear = htf_states == 'bearish'
            ttf_bear = ttf_states == 'bearish'
            ltf_bear = ltf_states == 'bearish'
            
            if strategy['direction'] == 'bullish':
                if confluence_type == 'HTF+TTF': active_mask = htf_bull & ttf_bull
                elif confluence_type == 'HTF+LTF': active_mask = htf_bull & ltf_bull
                else: active_mask = ttf_bull & ltf_bull # TTF+LTF
            else:
                if confluence_type == 'HTF+TTF': active_mask = htf_bear & ttf_bear
                elif confluence_type == 'HTF+LTF': active_mask = htf_bear & ltf_bear
                else: active_mask = ttf_bear & ltf_bear # TTF+LTF
        
        elif strategy['type'] == 'mtf_mode_c':
            htf_scores = htf_states.map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)
            ttf_scores = ttf_states.map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)
            ltf_scores = ltf_states.map({'bullish': 1, 'bearish': -1, 'neutral': 0}).fillna(0)
            
            weighted_score = (htf_scores * 0.5 + ttf_scores * 0.3 + ltf_scores * 0.2)
            threshold = strategy['score_threshold']
            
            if strategy['direction'] == 'bullish':
                active_mask = weighted_score >= threshold
            else:
                active_mask = weighted_score <= -threshold
        
        # Calculate performance metrics
        active_returns = ltf_df.loc[active_mask, 'future_return']
        
        if len(active_returns) == 0:
            return strategy
        
        if strategy['direction'] == 'bullish':
            wins = (active_returns > 0).sum()
            winning_returns = active_returns[active_returns > 0]
            losing_returns = active_returns[active_returns <= 0]
        else:
            wins = (active_returns < 0).sum()
            winning_returns = -active_returns[active_returns < 0]
            losing_returns = -active_returns[active_returns >= 0]
        
        total_signals = len(active_returns)
        win_rate = wins / total_signals if total_signals > 0 else 0
        avg_return = active_returns.mean()
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        profit_factor = abs(winning_returns.sum() / losing_returns.sum()) if losing_returns.sum() != 0 else np.inf
        
        strategy.update({
            'backtest_win_rate': win_rate,
            'backtest_total_signals': int(total_signals),
            'backtest_wins': int(wins),
            'backtest_losses': int(total_signals - wins),
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'performance_score': win_rate
        })
        
        return strategy

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 2: MTF Backtesting Functions (MODES F, G)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def backtest_mtf_strategy_enhanced(self, strategy):
        """
        Enhanced backtesting with advanced price action strategy support (Modes F, G)
        FIXED: Passes 'pair_tf' to get_or_compute_states.
        FIXED: Dynamically gets 'pair' from strategy.
        """
        if not strategy['type'].startswith('mtf_'):
            return strategy
        
        # --- START OF FIX: DYNAMIC PAIR ---
        pair_tf = strategy.get('pair_tf', 'btc_usdt_1m')
        pair = "_".join(pair_tf.split('_')[:2])
        # --- END OF FIX ---

        # Get the aligned dataframes with advanced patterns
        htf_df, ttf_df, ltf_df = self.get_mtf_dataframes(
            pair,
            strategy['htf_timeframe'],
            strategy['ttf_timeframe'], 
            strategy['ltf_timeframe']
        )
        
        if htf_df is None:
            return strategy
        
        # Enhance dataframes with advanced patterns for modes F and G
        if strategy['type'] in ['mtf_mode_f', 'mtf_mode_g']:
            htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
            ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
            ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
        
        active_mask = pd.Series(True, index=ltf_df.index) # Default mask
        
        # --- START OF FIX: PASS pair_tf ---
        htf_pair_tf = f"{pair}_{strategy['htf_timeframe']}"
        ttf_pair_tf = f"{pair}_{strategy['ttf_timeframe']}"
        ltf_pair_tf = f"{pair}_{strategy['ltf_timeframe']}"
        # --- END OF FIX ---

        # Apply the specific MTF mode logic
        if strategy['type'] == 'mtf_mode_f':
            for tf_key, tf_signals in [('htf', strategy.get('htf_signals', [])), 
                                       ('ttf', strategy.get('ttf_signals', [])), 
                                       ('ltf', strategy.get('ltf_signals', []))]:
                df = htf_df if tf_key == 'htf' else (ttf_df if tf_key == 'ttf' else ltf_df)
                current_pair_tf = htf_pair_tf if tf_key == 'htf' else (ttf_pair_tf if tf_key == 'ttf' else ltf_pair_tf)
                
                for sig in tf_signals:
                    if sig in df.columns:
                        states = self.get_or_compute_states(df, sig, current_pair_tf)
                        if states is not None:
                            active_mask &= (states == 'bullish' if strategy['direction'] == 'bullish' else states == 'bearish')
        
        elif strategy['type'] == 'mtf_mode_g':
            for tf_key, tf_signals in [('htf', strategy.get('htf_signals', [])), 
                                       ('ttf', strategy.get('ttf_signals', [])), 
                                       ('ltf', strategy.get('ltf_signals', []))]:
                df = htf_df if tf_key == 'htf' else (ttf_df if tf_key == 'ttf' else ltf_df)
                current_pair_tf = htf_pair_tf if tf_key == 'htf' else (ttf_pair_tf if tf_key == 'ttf' else ltf_pair_tf)

                for sig in tf_signals:
                    if sig in df.columns:
                        states = self.get_or_compute_states(df, sig, current_pair_tf)
                        if states is not None:
                            if 'trend_structure' in sig:
                                if strategy['direction'] == 'bullish':
                                    active_mask &= (~states.isin(['strong_downtrend']))
                                else:
                                    active_mask &= (~states.isin(['strong_uptrend']))
                            else:
                                active_mask &= (states == 'bullish' if strategy['direction'] == 'bullish' else states == 'bearish')
        
        else:
            # Use existing logic for other modes
            return self.backtest_mtf_strategy(strategy)
        
        # Calculate performance metrics
        active_returns = ltf_df.loc[active_mask, 'future_return']
        
        if len(active_returns) == 0:
            return strategy
        
        if strategy['direction'] == 'bullish':
            wins = (active_returns > 0).sum()
            winning_returns = active_returns[active_returns > 0]
            losing_returns = active_returns[active_returns <= 0]
        else:
            wins = (active_returns < 0).sum()
            winning_returns = active_returns[active_returns < 0]
            losing_returns = active_returns[active_returns >= 0]
        
        total_signals = len(active_returns)
        win_rate = wins / total_signals if total_signals > 0 else 0
        avg_return = active_returns.mean()
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        profit_factor = abs(winning_returns.sum() / losing_returns.sum()) if losing_returns.sum() != 0 else np.inf
        
        strategy.update({
            'backtest_win_rate': win_rate,
            'backtest_total_signals': int(total_signals),
            'backtest_wins': int(wins),
            'backtest_losses': int(total_signals - wins),
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'performance_score': win_rate
        })
        
        return strategy

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 3: MTF Backtesting Functions (MODES D, E, H, I, J, K, L, M, N)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def backtest_mtf_strategy_generic(self, strategy):
        """
        A generic backtester for all modern MTF strategies.
        FIXED: Passes 'pair_tf' to get_or_compute_states.
        FIXED: Dynamically gets 'pair' from strategy.
        """
        try:
            # --- START OF FIX: DYNAMIC PAIR ---
            pair_tf = strategy.get('pair_tf', 'btc_usdt_1m')
            pair = "_".join(pair_tf.split('_')[:2])
            # --- END OF FIX ---

            # Get the aligned dataframes with advanced patterns
            htf_df, ttf_df, ltf_df = self.get_mtf_dataframes(
                pair,
                strategy['htf_timeframe'],
                strategy['ttf_timeframe'], 
                strategy['ltf_timeframe']
            )
            
            if htf_df is None:
                print(f"Warning: Could not get data for strategy {strategy.get('id')}")
                return strategy

            htf_df = self.optimized_detect_advanced_price_patterns(htf_df)
            ttf_df = self.optimized_detect_advanced_price_patterns(ttf_df)
            ltf_df = self.optimized_detect_advanced_price_patterns(ltf_df)
            
            # --- Build the Signal Mask ---
            active_mask = pd.Series(True, index=ltf_df.index)
            direction = strategy['direction']
            
            # This filters out counter-trend trades which are usually low quality
            if 'trend_structure' in htf_df.columns:
                # Map HTF rows to LTF (reindex/ffill done in get_mtf_dataframes, assuming aligned)
                if direction == 'bullish':
                    active_mask &= (htf_df['trend_structure'].isin(['Uptrend', 'Strong Uptrend']))
                else:
                    active_mask &= (htf_df['trend_structure'].isin(['Downtrend', 'Strong Downtrend']))
            
            # --- START OF FIX: PASS pair_tf ---
            htf_pair_tf = f"{pair}_{strategy['htf_timeframe']}"
            ttf_pair_tf = f"{pair}_{strategy['ttf_timeframe']}"
            ltf_pair_tf = f"{pair}_{strategy['ltf_timeframe']}"
            # --- END OF FIX ---

            signal_groups = [
                strategy.get('signals'),
                strategy.get('price_action_signals'),
                strategy.get('indicator_signals'),
                strategy.get('confirmation_signals'),
                strategy.get('adaptive_signal_set')
            ]
            
            signal_config = None
            for group in signal_groups:
                if isinstance(group, dict) and ('htf' in group or 'ttf' in group or 'ltf' in group):
                    signal_config = group
                    break
            
            if not signal_config:
                signal_config = {
                    'htf': strategy.get('htf_signals', []),
                    'ttf': strategy.get('ttf_signals', []),
                    'ltf': strategy.get('ltf_signals', [])
                }
            
            # Apply the mask
            for tf_key, signals in signal_config.items():
                if not signals: continue

                # Handle nested dicts (like Mode E)
                if tf_key in ['price_action_signals', 'indicator_signals'] and isinstance(signals, dict):
                    for nested_tf_key, nested_signals in signals.items():
                        df = htf_df if nested_tf_key == 'htf' else ttf_df if nested_tf_key == 'ttf' else ltf_df
                        current_pair_tf = htf_pair_tf if nested_tf_key == 'htf' else (ttf_pair_tf if nested_tf_key == 'ttf' else ltf_pair_tf)
                        for sig in nested_signals:
                            if sig in df.columns:
                                states = self.get_or_compute_states(df, sig, current_pair_tf)
                                if states is not None:
                                    active_mask &= (states == 'bullish' if direction == 'bullish' else states == 'bearish')
                # Handle flat list
                elif isinstance(signals, list):
                    df = htf_df if tf_key == 'htf' else ttf_df if tf_key == 'ttf' else ltf_df
                    current_pair_tf = htf_pair_tf if tf_key == 'htf' else (ttf_pair_tf if tf_key == 'ttf' else ltf_pair_tf)
                    for sig in signals:
                        if sig in df.columns:
                            states = self.get_or_compute_states(df, sig, current_pair_tf)
                            if states is not None:
                                active_mask &= (states == 'bullish' if direction == 'bullish' else states == 'bearish')

            # --- Calculate Performance ---
            active_returns = ltf_df.loc[active_mask, 'future_return']
            
            if len(active_returns) == 0:
                strategy['backtest_win_rate'] = 0
                strategy['backtest_total_signals'] = 0
                return strategy
            
            if direction == 'bullish':
                wins = (active_returns > 0).sum()
                winning_returns = active_returns[active_returns > 0]
                losing_returns = active_returns[active_returns <= 0]
            else:
                wins = (active_returns < 0).sum()
                winning_returns = active_returns[active_returns < 0]
                losing_returns = active_returns[active_returns >= 0]
            
            total_signals = len(active_returns)
            win_rate = wins / total_signals if total_signals > 0 else 0
            avg_return = active_returns.mean()
            
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
            profit_factor = abs(winning_returns.sum() / losing_returns.sum()) if losing_returns.sum() != 0 else np.inf
            
            strategy.update({
                'backtest_win_rate': win_rate,
                'backtest_total_signals': int(total_signals),
                'backtest_wins': int(wins),
                'backtest_losses': int(total_signals - wins),
                'avg_return': avg_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'performance_score': win_rate * (1 + abs(avg_return))
            })
            
            return strategy
            
        except Exception as e:
            import traceback
            print(f"❌ Error in generic MTF backtest for {strategy.get('id')}: {e}")
            print(traceback.format_exc())
            return strategy

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 4: Realistic & Regime Backtesting (for Single-TF)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def realistic_backtest(self, strategy, transaction_cost=0.001, max_position_size=0.1):
        """
        Vectorized realistic backtesting
        FIXED: Passes 'pair_tf' to get_or_compute_states.
        """
        pair_tf = strategy['pair_tf']
        df = self.all_dataframes[pair_tf].copy()
        
        df = self.enhanced_market_regime_detection(df)
        df = self.identify_price_states(df)
        
        # --- START OF FIX ---
        signal_states = self.get_or_compute_states(df, strategy['signal_name'], pair_tf)
        # --- END OF FIX ---
        
        if signal_states is None:
            return strategy
        
        capital = 10000
        position = 0
        trades = []
        equity_curve = [capital]
        
        close_values = df['close'].values
        signal_values = signal_states.values
        
        for i in range(len(df)):
            current_price = close_values[i]
            signal = signal_values[i]
            
            if signal == strategy['direction'] and position == 0:
                position_size = min(max_position_size * capital, capital * 0.1)
                shares = position_size / current_price
                
                position = shares
                entry_price = current_price
                entry_idx = i
                
                trades.append({
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'shares': shares,
                    'type': 'long' if strategy['direction'] == 'bullish' else 'short'
                })
            
            elif position != 0:
                if strategy['direction'] == 'bullish':
                    unrealized_pnl = (current_price - trades[-1]['entry_price']) * position
                else:
                    unrealized_pnl = (trades[-1]['entry_price'] - current_price) * position
                
                unrealized_pnl -= transaction_cost * capital
                
                stop_loss = -0.02 * capital
                take_profit = 0.04 * capital
                
                if unrealized_pnl <= stop_loss or unrealized_pnl >= take_profit:
                    capital += unrealized_pnl
                    position = 0
                    
                    trades[-1].update({
                        'exit_idx': i,
                        'exit_price': current_price,
                        'pnl': unrealized_pnl,
                        'return_pct': (unrealized_pnl / (trades[-1]['entry_price'] * trades[-1]['shares'])) * 100
                    })
            
            total_equity = capital + (position * current_price if position != 0 else 0)
            equity_curve.append(total_equity)
        
        if trades:
            strategy['realistic_trades'] = trades
            strategy['equity_curve'] = equity_curve
            strategy['final_capital'] = capital
            strategy['total_return_pct'] = (capital - 10000) / 10000 * 100
            strategy['max_drawdown_realistic'] = self._calculate_max_drawdown_fast(np.array(equity_curve))
        
        return strategy

    def regime_aware_backtesting(self, strategy_key, strategy):
        """
        Backtest strategy performance across different market regimes
        FIXED: Passes 'pair_tf' to get_or_compute_states.
        FIXED: Uses new 'historical_regime' column.
        """
        pair_tf = strategy['pair_tf']
        df = self.all_dataframes[pair_tf].copy()
        
        df = self.enhanced_market_regime_detection(df)
        df = self.identify_price_states(df)
        
        # --- START OF FIX ---
        signal_states = self.get_or_compute_states(df, strategy['signal_name'], pair_tf)
        # --- END OF FIX ---
        
        if signal_states is None:
            return strategy
        
        regime_performance = {}
        # FIX: Use the new 'historical_regime' column
        if 'historical_regime' not in df.columns:
            print(f"  ⚠️  Missing 'historical_regime' for {pair_tf}, skipping regime backtest.")
            return strategy
            
        regimes = df['historical_regime'].unique()
        
        for regime in regimes:
            regime_mask = (df['historical_regime'] == regime).values
            
            # --- START OF INDEXERROR FIX ---
            # This check is now redundant because the cache key is fixed in core.py,
            # but it provides good defense.
            if len(signal_states.values) != len(regime_mask):
                print(f"  ⚠️  CRITICAL MISMATCH in regime_aware_backtesting for {strategy_key}.")
                print(f"     Signal state length: {len(signal_states.values)}, DF/Regime length: {len(regime_mask)}")
                print(f"     This is a caching bug in core.py. Skipping this strategy's regime test.")
                break 
            # --- END OF INDEXERROR FIX ---
            
            if regime_mask.sum() < 10:
                continue
            
            # Apply the regime mask to signals and returns
            regime_signals = signal_states.values[regime_mask]
            regime_returns = df['future_return'].values[regime_mask]
            
            direction = strategy['direction']
            active_mask = regime_signals == direction
            
            if active_mask.sum() == 0:
                continue
            
            active_returns = regime_returns[active_mask]
            if direction == 'bullish':
                wins = (active_returns > 0).sum()
            else:
                wins = (active_returns < 0).sum()
            
            total = len(active_returns)
            win_rate = wins / total if total > 0 else 0
            
            regime_performance[regime] = {
                'win_rate': win_rate,
                'sample_size': total,
                'wins': int(wins),
                'losses': int(total - wins)
            }
        
        strategy['regime_performance'] = regime_performance
        strategy['regime_robustness_score'] = self._calculate_regime_robustness(regime_performance)
        
        return strategy

    def _calculate_regime_robustness(self, regime_performance):
        """Calculates a score based on how well a strategy performs across regimes."""
        if not regime_performance:
            return 0
        
        win_rates = [v['win_rate'] for v in regime_performance.values() if v['sample_size'] > 10]
        if not win_rates:
            return 0
            
        avg_win_rate = np.mean(win_rates)
        std_dev = np.std(win_rates)
        
        # Score favors consistency (low std_dev) and high avg_win_rate
        robustness_score = avg_win_rate * (1 - std_dev)
        return max(0, robustness_score) # Ensure score is not negative


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 5: Walk-Forward & Metric Helpers
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def walk_forward_detailed(self, df, signal_states, horizons=(3, 5, 10), 
                                window=500, step=100, cost_bps=5, 
                                stop_loss_pct=None, take_profit_pct=None):
        """Vectorized walk-forward analysis"""
        results = []
        
        close_values = df['close'].values
        signal_values = signal_states.values
        
        for start in range(0, len(df) - window - max(horizons), step):
            end = start + window
            test_start = end
            test_end = min(end + step, len(df) - max(horizons))
            
            test_idx = slice(test_start, test_end)
            
            for h in horizons:
                fut_ret = (np.roll(close_values, -h) - close_values) / close_values * 100
                fut_ret[-h:] = np.nan
                net_ret = fut_ret - (cost_bps / 100.0)
                
                test_signals = signal_values[test_idx]
                test_returns = net_ret[test_idx]
                
                bull_mask = test_signals == 'bullish'
                bear_mask = test_signals == 'bearish'
                
                if bull_mask.any():
                    bull_rets = test_returns[bull_mask]
                    bull_rets = bull_rets[~np.isnan(bull_rets)]
                    wr_bull = (bull_rets > 0).mean() if len(bull_rets) > 0 else np.nan
                    sharpe_bull = self._calculate_sharpe_fast(bull_rets)
                    sortino_bull = self._calculate_sortino_fast(bull_rets)
                    max_dd_bull = self._calculate_max_drawdown_fast(np.cumsum(bull_rets))
                else:
                    wr_bull = sharpe_bull = sortino_bull = max_dd_bull = np.nan
                
                if bear_mask.any():
                    bear_rets = -test_returns[bear_mask]
                    bear_rets = bear_rets[~np.isnan(bear_rets)]
                    wr_bear = (bear_rets > 0).mean() if len(bear_rets) > 0 else np.nan
                    sharpe_bear = self._calculate_sharpe_fast(bear_rets)
                    sortino_bear = self._calculate_sortino_fast(bear_rets)
                    max_dd_bear = self._calculate_max_drawdown_fast(np.cumsum(bear_rets))
                else:
                    wr_bear = sharpe_bear = sortino_bear = max_dd_bear = np.nan
                
                results.append({
                    'train_start': start,
                    'train_end': end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'horizon': h,
                    'wr_bull': wr_bull,
                    'sharpe_bull': sharpe_bull,
                    'sortino_bull': sortino_bull,
                    'max_dd_bull': max_dd_bull,
                    'wr_bear': wr_bear,
                    'sharpe_bear': sharpe_bear,
                    'sortino_bear': sortino_bear,
                    'max_dd_bear': max_dd_bear,
                    'n_bull_signals': int(bull_mask.sum()),
                    'n_bear_signals': int(bear_mask.sum())
                })
        
        return pd.DataFrame(results)

    def calculate_comprehensive_metrics(self, returns_series, benchmark_returns=None):
        """Vectorized comprehensive risk-adjusted performance metrics"""
        if len(returns_series) < 2:
            return {}
        
        returns = returns_series.dropna().values
        metrics = {}
        
        metrics['total_return'] = returns.sum()
        metrics['avg_return'] = returns.mean()
        metrics['volatility'] = returns.std()
        
        periods_per_year = 365 * 24 # Assuming 1-hour data, adjust if needed
        risk_free_rate = 0.02 / periods_per_year
        excess_returns = returns - risk_free_rate
        
        if excess_returns.std() > 0:
            metrics['sharpe_ratio'] = (excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year))
        else:
            metrics['sharpe_ratio'] = 0
        
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        if downside_volatility > 0:
            metrics['sortino_ratio'] = (excess_returns.mean() / downside_volatility * np.sqrt(periods_per_year))
        else:
            metrics['sortino_ratio'] = 0
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown.mean()
        
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns <= 0]
        
        metrics['win_rate'] = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
        metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
        metrics['profit_factor'] = (abs(winning_trades.sum()) / abs(losing_trades.sum()) 
                                if losing_trades.sum() != 0 else float('inf'))
        
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        if metrics['avg_loss'] != 0 and metrics['avg_win'] > 0:
            win_prob = metrics['win_rate']
            win_avg = metrics['avg_win']
            loss_avg = abs(metrics['avg_loss'])
            if loss_avg > 0:
                metrics['kelly_criterion'] = win_prob - (1 - win_prob) / (win_avg / loss_avg)
            else:
                metrics['kelly_criterion'] = 0
        else:
            metrics['kelly_criterion'] = 0
        
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = (metrics['avg_return'] * periods_per_year / 
                                    abs(metrics['max_drawdown']))
        else:
            metrics['calmar_ratio'] = 0
        
        metrics['max_winning_streak'] = self._calculate_max_streak(returns > 0)
        metrics['max_losing_streak'] = self._calculate_max_streak(returns <= 0)
        
        return metrics

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # SECTION 6: Vectorized Metric Helpers (Added)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _calculate_sharpe_fast(self, returns, periods_per_year=252*24):
        """Fast vectorized sharpe ratio calculation"""
        if len(returns) < 2:
            return 0.0
        
        std_dev = np.std(returns)
        if std_dev == 0:
            return 0.0
            
        return np.mean(returns) / std_dev * np.sqrt(periods_per_year)

    def _calculate_sortino_fast(self, returns, periods_per_year=252*24):
        """Fast vectorized sortino ratio calculation"""
        if len(returns) < 2:
            return 0.0
            
        downside_returns = returns[returns < 0]
        downside_std_dev = np.std(downside_returns)
        
        if downside_std_dev == 0:
            return 0.0
            
        return np.mean(returns) / downside_std_dev * np.sqrt(periods_per_year)

    def _calculate_max_drawdown_fast(self, cumulative_returns):
        """Fast vectorized max drawdown calculation"""
        if len(cumulative_returns) == 0:
            return 0.0
            
        # Ensure input is a numpy array
        cumulative_returns = np.asarray(cumulative_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max)
        
        # Handle cases where running_max is zero to avoid division by zero
        running_max_safe = np.where(running_max == 0, 1, running_max)
        drawdown_pct = drawdown / running_max_safe
        drawdown_pct[running_max == 0] = 0 # Set drawdown to 0 where max was 0
        
        return np.min(drawdown_pct)

    def _calculate_max_streak(self, bool_array):
        """Vectorized maximum consecutive True values"""
        if len(bool_array) == 0:
            return 0
        
        bool_array = np.asarray(bool_array, dtype=bool)
        diff = np.diff(np.concatenate(([False], bool_array, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
        
        return int((ends - starts).max())

