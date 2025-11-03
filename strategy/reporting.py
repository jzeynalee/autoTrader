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


class ReportsMixin:
    def format_win_rate_with_ci(self, win_rate, wins, total_trades, confidence=0.95):
        """
        Formats win rate with a Wilson score confidence interval.
        """
        if total_trades == 0:
            return f"{win_rate:.2%} (±0.0%) [0 trades]"
        
        try:            
            from statsmodels.stats.proportion import proportion_confint
            z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
            
            # Use statsmodels if available
            ci_low, ci_upp = proportion_confint(wins, total_trades, alpha=1-confidence, method='wilson')
            margin_of_error = (ci_upp - ci_low) / 2
        except ImportError:
            # Fallback to simple Normal approximation if statsmodels isn't installed
            if total_trades > 0:
                p = win_rate
                margin_of_error = z * np.sqrt((p * (1 - p)) / total_trades)
            else:
                margin_of_error = 0

        return f"{win_rate:.2%} (±{margin_of_error:.1%}) [{total_trades} trades]"
    
    def print_strategy_report(self, top_n=20):
        """Print detailed report of the top strategies"""
        print("\n" + "="*80)
        print(f"TOP {top_n} STRATEGIES BY PERFORMANCE SCORE")
        print("="*80)
        
        strategies_with_results = [
            (k, v) for k, v in self.strategy_pool.items() 
            if v.get('backtest_win_rate') is not None and v.get('backtest_total_signals', 0) >= 10 # Lowered threshold to see more
        ]
        
        # Sort by performance score
        strategies_with_results.sort(key=lambda x: x[1].get('performance_score', 0), reverse=True)

        
        for i, (key, strategy) in enumerate(strategies_with_results[:top_n], 1):
            print(f"\n{'─'*80}")
            print(f"#{i} | {key} | {strategy['pair_tf']}")
            print(f"{'─'*80}")
            
            # --- START FIX ---
            # Get a descriptive name for any strategy type
            strategy_type = strategy.get('type', 'single_signal')
            signal_name = strategy.get('signal_name') # For single-signal
            
            if not signal_name:
                signal_name = strategy.get('setup_name') # For Mode D, E, F, G, H, I
            if not signal_name:
                signal_name = strategy.get('pattern_name') # For Mode L, N, Confluence
            if not signal_name:
                # Fallback for Mode A, B, C
                signal_name = f"{strategy.get('htf_indicator', '?')} -> {strategy.get('ttf_indicator', '?')} -> {strategy.get('ltf_indicator', '?')}"
            
            print(f"Type: {strategy.get('signal_type', strategy_type).upper()}")
            print(f"Signal: {signal_name}")
            # --- END FIX ---
            
            print(f"Direction: {strategy['direction'].upper()} → {strategy['trade_direction'].upper()}")
            print(f"\nPerformance Metrics:")

            print(f"  Win Rate: {self.format_win_rate_with_ci(
                strategy['backtest_win_rate'],
                strategy['backtest_wins'],
                strategy['backtest_total_signals']
            )}")

            print(f"  Total Signals: {strategy['backtest_total_signals']}")
            print(f"  Wins: {strategy['backtest_wins']} | Losses: {strategy['backtest_losses']}")
            
            # Use .get() for optional keys to prevent future crashes
            print(f"  Avg Return: {strategy.get('avg_return', 0):.3f}%")
            print(f"  Avg Win: {strategy.get('avg_win', 0):.3f}% | Avg Loss: {strategy.get('avg_loss', 0):.3f}%")
            print(f"  Profit Factor: {strategy.get('profit_factor', 0):.3f}")
            print(f"  Performance Score: {strategy.get('performance_score', 0):.3f}")
            print(f"  Expectancy: {strategy.get('expectancy', 0):.3f}%")

    def export_summary_csv(self, filename='strategy_summary.csv'):
        """Export strategy summary to CSV"""
        summary_data = []
        
        for key, strategy in self.strategy_pool.items():
            if 'backtest_win_rate' in strategy:
                summary_data.append({
                    'strategy_id': key,
                    'pair_timeframe': strategy['pair_tf'],
                    'signal_type': strategy['signal_type'],
                    'signal_name': strategy['signal_name'],
                    'direction': strategy['direction'],
                    'trade_direction': strategy['trade_direction'],
                    'win_rate': strategy['backtest_win_rate'],
                    'total_signals': strategy['backtest_total_signals'],
                    'wins': strategy['backtest_wins'],
                    'losses': strategy['backtest_losses'],
                    'avg_return': strategy.get('avg_return', 0),
                    'profit_factor': strategy['profit_factor'],
                    'performance_score': strategy.get('performance_score', 0),
                    'expectancy': strategy.get('expectancy', 0)
                })

        if not summary_data:
            print("⚠️ No strategies with backtest results found. CSV not created.")
            return

        df = pd.DataFrame(summary_data)

        if 'performance_score' not in df.columns:
            print("⚠️ 'performance_score' column missing. Adding default = 0.")
            df['performance_score'] = 0

        try:
            df = df.sort_values('performance_score', ascending=False)
        except Exception as e:
            print(f"⚠️ Sort skipped due to: {e}")

        df.to_csv(filename, index=False)
        print(f"✓ Summary exported to {filename} ({len(df)} strategies)")

    def save_strategies_to_file(self, filename=None):
        """Save discovered strategies to JSON file"""
        if not self.strategy_pool:
            print("No strategies to save.")
            return
        
        if filename is None:
            filename = f"mtf_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        strategies_data = {
            'discovery_time': datetime.now().isoformat(),
            'total_strategies': len(self.strategy_pool),
            'strategies': {}
        }
        
        for strategy_id, strategy in self.strategy_pool.items():
            strategies_data['strategies'][strategy_id] = strategy
        
        try:
            with open(filename, 'w') as f:
                json.dump(strategies_data, f, indent=2, default=str)
            print(f"✅ Strategies saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving strategies: {e}")

    def generate_mtf_strategy_report(self):
        """Generate separate comprehensive MTF strategy report"""
        mtf_strategies = {
            k: v for k, v in self.strategy_pool.items() 
            if v.get('type', '').startswith('mtf_')
        }
        
        if not mtf_strategies:
            print("No MTF strategies found.")
            return
        
        print("\n" + "="*80)
        print("MULTI-TIMEFRAME STRATEGY REPORT")
        print("="*80)
        print(f"Total MTF Strategies: {len(mtf_strategies)}")
        
        # Group by mode
        by_mode = {}
        for strat_id, strat in mtf_strategies.items():
            mode = strat.get('type', 'unknown')
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append((strat_id, strat))
        
        # Report each mode
        for mode, strategies in sorted(by_mode.items()):
            print(f"\n{'─'*80}")
            print(f"MODE: {mode.upper()}")
            print(f"{'─'*80}")
            print(f"Count: {len(strategies)} strategies")
            
            # Sort by performance
            strategies.sort(key=lambda x: x[1].get('performance_score', 0), reverse=True)
            
            # Show top 5 per mode
            for i, (strat_id, strat) in enumerate(strategies[:5], 1):
                print(f"\n  #{i} {strat_id}")
                print(f"    Group: {strat.get('group', 'N/A')}")
                print(f"    Direction: {strat.get('direction', 'N/A').upper()}")
                print(f"    Timeframes: {strat.get('htf_timeframe')} → {strat.get('ttf_timeframe')} → {strat.get('ltf_timeframe')}")
                print(f"    Win Rate: {strat.get('discovered_accuracy', 0):.2%}")
                print(f"    Samples: {strat.get('sample_size', 0)}")
                print(f"    Performance Score: {strat.get('performance_score', 0):.3f}")
                
                # Show signals for advanced modes
                if 'htf_signals' in strat:
                    print(f"    HTF Signals: {', '.join(strat['htf_signals'][:3])}")
                    print(f"    TTF Signals: {', '.join(strat['ttf_signals'][:3])}")
                    print(f"    LTF Signals: {', '.join(strat['ltf_signals'][:3])}")                    


    def export_live_monitoring_config(self, filename='live_monitor_config.json'):
        """Export configuration for live monitoring based on discovered strategies"""
        print("\n" + "="*80)
        print("GENERATING LIVE MONITORING CONFIGURATION")
        print("="*80)
        
        top_strategies = [
            (k, v) for k, v in self.strategy_pool.items()
            if 'backtest_win_rate' in v and v['backtest_total_signals'] >= 20
            and v['backtest_win_rate'] >= 0.60
        ][:30]
        
        monitoring_config = {
            'generated_at': datetime.now().isoformat(),
            'lookforward_periods': self.lookforward_periods,
            'price_threshold': self.price_threshold,
            'strategies': []
        }
        
        for strategy_id, strategy in top_strategies:
            config_entry = {
                'strategy_id': strategy_id,
                'pair_timeframe': strategy['pair_tf'],
                'signal_type': strategy['signal_type'],
                'signal_name': strategy['signal_name'],
                'direction': strategy['direction'],
                'trade_direction': strategy['trade_direction'],
                'win_rate': strategy['backtest_win_rate'],
                'profit_factor': strategy['profit_factor'],
                'expectancy': strategy.get('expectancy', 0),
                'alert_condition': f"When {strategy['signal_name']} is {strategy['direction']}"
            }
            monitoring_config['strategies'].append(config_entry)
        
        with open(filename, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        print(f"✓ Live monitoring config exported to {filename}")
        print(f"  Included {len(top_strategies)} high-performance strategies")


    def print_mtf_strategy_report(self, top_n=15):
        """Print specialized report for MTF strategies"""
        mtf_strategies = [
            (k, v) for k, v in self.strategy_pool.items() 
            if v.get('type', '').startswith('mtf_')
        ]
        
        if not mtf_strategies:
            print("No MTF strategies found")
            return
        
        print("\n" + "="*80)
        print(f"MULTI-TIMEFRAME STRATEGY REPORT - TOP {top_n}")
        print("="*80)
        
        # Sort by performance score
        mtf_strategies.sort(key=lambda x: x[1].get('performance_score', 0), reverse=True)
        
        for i, (key, strategy) in enumerate(mtf_strategies[:top_n], 1):
            print(f"\n{'─'*80}")
            print(f"#{i} | {key} | {strategy['group']} | {strategy['type']}")
            print(f"{'─'*80}")
            
            print(f"Timeframes: {strategy['htf_timeframe']} → {strategy['ttf_timeframe']} → {strategy['ltf_timeframe']}")
            print(f"Direction: {strategy['direction'].upper()}")
            print(f"HTF: {strategy['htf_indicator']}")
            print(f"TTF: {strategy['ttf_indicator']}") 
            print(f"LTF: {strategy['ltf_indicator']}")
            
            if strategy['type'] == 'mtf_mode_b':
                print(f"Confluence: {strategy['confluence_type']}")
            elif strategy['type'] == 'mtf_mode_c':
                print(f"Score Threshold: {strategy['score_threshold']}")
            
            print(f"\nPerformance:")
            print(f"  Win Rate: {strategy.get('backtest_win_rate', strategy.get('discovered_accuracy', 0)):.2%}")
            print(f"  Samples: {strategy.get('backtest_total_signals', strategy.get('sample_size', 0))}")
            print(f"  Performance Score: {strategy.get('performance_score', 0):.3f}")


    def print_enhanced_strategy_report(self, top_n=15):
        """Print enhanced strategy report with new metrics"""
        print("\n" + "="*80)
        print(f"ENHANCED STRATEGY REPORT - TOP {top_n}")
        print("="*80)
        
        valid_strategies = [
            (k, v) for k, v in self.strategy_pool.items()
            if v.get('backtest_total_signals', 0) >= 20
            and v.get('regime_robustness_score', 0) > 0.3
        ]
        
        valid_strategies.sort(key=lambda x: (
            x[1].get('performance_score', 0) * 
            x[1].get('regime_robustness_score', 1)
        ), reverse=True)
        
        for i, (key, strategy) in enumerate(valid_strategies[:top_n], 1):
            print(f"\n{'─'*80}")
            print(f"#{i} | {key} | {strategy['pair_tf']}")
            print(f"{'─'*80}")
            
            print(f"Signal: {strategy['signal_name']} | Direction: {strategy['direction'].upper()}")
            print(f"Type: {strategy['signal_type'].upper()} | Samples: {strategy['backtest_total_signals']}")
            
            print(f"\nPerformance:")
            print(f"  Win Rate: {strategy['backtest_win_rate']:.2%}")
            print(f"  Profit Factor: {strategy['profit_factor']:.3f}")
            print(f"  Performance Score: {strategy['performance_score']:.3f}")
            print(f"  Regime Robustness: {strategy.get('regime_robustness_score', 0):.3f}")
            
            if 'realistic_trades' in strategy:
                print(f"  Realistic Return: {strategy.get('total_return_pct', 0):.2f}%")
                print(f"  Max Drawdown: {strategy.get('max_drawdown_realistic', 0):.2f}%")
            
            if 'regime_performance' in strategy:
                print(f"\nRegime Performance:")
                for regime, stats in list(strategy['regime_performance'].items())[:3]:
                    if stats['sample_size'] >= 10:
                        print(f"  {regime}: {stats['win_rate']:.2%} ({stats['sample_size']} samples)")

    def save_strategies_to_file(self, filename=None):
        """Save discovered strategies to JSON file"""
        if not self.strategy_pool:
            print("No strategies to save.")
            return
        
        if filename is None:
            filename = f"mtf_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        strategies_data = {
            'discovery_time': datetime.now().isoformat(),
            'total_strategies': len(self.strategy_pool),
            'strategies': {}
        }
        
        for strategy_id, strategy in self.strategy_pool.items():
            strategies_data['strategies'][strategy_id] = strategy
        
        try:
            with open(filename, 'w') as f:
                json.dump(strategies_data, f, indent=2, default=str)
            print(f"✅ Strategies saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving strategies: {e}")

    def generate_strategy_report(self):
        """Generate comprehensive strategy report"""
        if not self.strategy_pool:
            print("No strategies discovered yet.")
            return
        
        print("\n" + "="*80)
        print("MTF STRATEGY DISCOVERY REPORT")
        print("="*80)
        
        # Group by strategy type
        by_type = {}
        for strategy_id, strategy in self.strategy_pool.items():
            strategy_type = strategy.get('type', 'unknown')
            if strategy_type not in by_type:
                by_type[strategy_type] = []
            by_type[strategy_type].append(strategy)
        
        # Print summary by type
        for strategy_type, strategies in by_type.items():
            avg_accuracy = sum(s.get('discovered_accuracy', 0) for s in strategies) / len(strategies)
            total_samples = sum(s.get('sample_size', 0) for s in strategies)
            
            print(f"\n📊 {strategy_type.upper()}:")
            print(f"   Count: {len(strategies)} strategies")
            print(f"   Avg Accuracy: {avg_accuracy:.2%}")
            print(f"   Total Samples: {total_samples}")
            
            # Show top 3 strategies by accuracy
            top_strategies = sorted(strategies, key=lambda x: x.get('discovered_accuracy', 0), reverse=True)[:3]
            for i, strategy in enumerate(top_strategies, 1):
                print(f"   #{i}: {strategy.get('pair_tf', 'N/A')} - "
                    f"{strategy.get('discovered_accuracy', 0):.2%} accuracy "
                    f"(samples: {strategy.get('sample_size', 0)})")

    def export_strategies_to_csv(self, filename=None):
        """Export strategies to CSV for analysis"""
        if not self.strategy_pool:
            print("No strategies to export.")
            return
        
        if filename is None:
            filename = f"mtf_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            # Convert to DataFrame
            strategies_list = []
            for strategy_id, strategy in self.strategy_pool.items():
                row = strategy.copy()
                row['strategy_id'] = strategy_id
                strategies_list.append(row)
            
            df = pd.DataFrame(strategies_list)
            df.to_csv(filename, index=False)
            print(f"✅ Strategies exported to {filename}")
            return df
        except Exception as e:
            print(f"❌ Error exporting strategies: {e}")
            return None

    def generate_mtf_strategy_report(self):
        """Generate separate comprehensive MTF strategy report"""
        mtf_strategies = {
            k: v for k, v in self.strategy_pool.items() 
            if v.get('type', '').startswith('mtf_')
        }
        
        if not mtf_strategies:
            print("No MTF strategies found.")
            return
        
        print("\n" + "="*80)
        print("MULTI-TIMEFRAME STRATEGY REPORT")
        print("="*80)
        print(f"Total MTF Strategies: {len(mtf_strategies)}")
        
        # Group by mode
        by_mode = {}
        for strat_id, strat in mtf_strategies.items():
            mode = strat.get('type', 'unknown')
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append((strat_id, strat))
        
        # Report each mode
        for mode, strategies in sorted(by_mode.items()):
            print(f"\n{'─'*80}")
            print(f"MODE: {mode.upper()}")
            print(f"{'─'*80}")
            print(f"Count: {len(strategies)} strategies")
            
            # Sort by performance
            strategies.sort(key=lambda x: x[1].get('performance_score', 0), reverse=True)
            
            # Show top 5 per mode
            for i, (strat_id, strat) in enumerate(strategies[:5], 1):
                print(f"\n  #{i} {strat_id}")
                print(f"    Group: {strat.get('group', 'N/A')}")
                print(f"    Direction: {strat.get('direction', 'N/A').upper()}")
                print(f"    Timeframes: {strat.get('htf_timeframe')} → {strat.get('ttf_timeframe')} → {strat.get('ltf_timeframe')}")
                print(f"    Win Rate: {strat.get('discovered_accuracy', 0):.2%}")
                print(f"    Samples: {strat.get('sample_size', 0)}")
                print(f"    Performance Score: {strat.get('performance_score', 0):.3f}")
                
                # Show signals for advanced modes
                if 'htf_signals' in strat:
                    print(f"    HTF Signals: {', '.join(strat['htf_signals'][:3])}")
                    print(f"    TTF Signals: {', '.join(strat['ttf_signals'][:3])}")
                    print(f"    LTF Signals: {', '.join(strat['ltf_signals'][:3])}")                    

    def export_mtf_strategies_csv(self, filename='mtf_strategies.csv'):
        """Export MTF strategies to separate CSV"""
        mtf_strategies = []
        
        for strat_id, strat in self.strategy_pool.items():
            if not strat.get('type', '').startswith('mtf_'):
                continue
                
            row = {
                'strategy_id': strat_id,
                'mode': strat.get('type'),
                'group': strat.get('group'),
                'direction': strat.get('direction'),
                'htf_timeframe': strat.get('htf_timeframe'),
                'ttf_timeframe': strat.get('ttf_timeframe'),
                'ltf_timeframe': strat.get('ltf_timeframe'),
                'win_rate': strat.get('discovered_accuracy', 0),
                'sample_size': strat.get('sample_size', 0),
                'performance_score': strat.get('performance_score', 0),
            }
            
            # Add mode-specific fields
            if 'confluence_score' in strat:
                row['confluence_score'] = strat['confluence_score']
            if 'alignment_score' in strat:
                row['alignment_score'] = strat['alignment_score']
            if 'regime_context' in strat:
                row['regime_context'] = strat['regime_context']
                
            mtf_strategies.append(row)
        
        if mtf_strategies:
            df = pd.DataFrame(mtf_strategies)
            df = df.sort_values('performance_score', ascending=False)
            df.to_csv(filename, index=False)
            print(f"✅ MTF strategies exported to {filename}")
            return df
        else:
            print("⚠️ No MTF strategies to export")
            return None

    