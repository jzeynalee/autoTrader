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
except ImportError as e:
    # Use the error for debugging but still rely on the absolute/sys.path injection for discovery
    print(f"Warning: discovery_mapping.py import failed: {e}")
    # Fallback/alternative import attempts are too complex. We assume success after sys.path fix.
    print("FATAL: Cannot load required mapping file. System cannot proceed.")
    MAPPER_AVAILABLE = False
    exit(1)

from .analysis_trend import TrendAnalysisSystem
from .analysis_pullback import PullbackAnalysisSystem
from .analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .analysis_confluence_scoring import ConfluenceScoringSystem
from .core import StrategyDiscoverySystem

if __name__ == "__main__":
    start_time = time.time()
    
    # ============================================================================
    # PHASE 1: INITIALIZATION & DATA LOADING
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 1: SYSTEM INITIALIZATION")
    print("="*80)
    
    system = StrategyDiscoverySystem(
        data_dir='lbank_data',
        lookforward_periods=5,
        price_threshold=0.005,
        n_jobs=-1
    )
    
    system.trend_analyzer    = TrendAnalysisSystem()
    system.pullback_analyzer = PullbackAnalysisSystem()
    system.regime_detector   = AdvancedRegimeDetectionSystem()
    system.confluence_scorer = ConfluenceScoringSystem()
    
    if not system.load_data():
        print("❌ Failed to load data. Exiting.")
        exit(1)
    
    # Optimize memory usage once at start
    system.optimize_data_loading()
    
    # Detect market regimes (in proper HTF→LTF order)
    print("\n🔍 Detecting market regimes...")
    timeframe_priority = {'4h': 1, '1h': 2, '15m': 3, '5m': 4, '1m': 5}
    
    def _pair_tf_priority(pair_tf):
        try:
            tf = pair_tf.split('_')[-1].lower()
            return timeframe_priority.get(tf, 999)
        except:
            return 999
    
    for pair_tf in sorted(list(system.all_dataframes.keys()), key=_pair_tf_priority):
        df = system.all_dataframes[pair_tf]
        system.all_dataframes[pair_tf] = system.enhanced_market_regime_detection(df)
    
    print(f"✅ Loaded {len(system.all_dataframes)} datasets")
    
    # ============================================================================
    # PHASE 2: STRATEGY DISCOVERY (SINGLE PASS)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 2: STRATEGY DISCOVERY")
    print("="*80)
    
    # Single-timeframe strategies
    print("\n📊 Discovering single-timeframe strategies...")
    system.discover_multi_signal_strategies()
    single_tf_count = len(system.strategy_pool)
    print(f"✅ Found {single_tf_count} single-timeframe strategies")
    
    # Multi-timeframe strategies (all modes A-N + confluence)
    print("\n🔄 Discovering multi-timeframe strategies...")
    mtf_start_id = len(system.strategy_pool)
    system.discover_mtf_strategies()
    mtf_count = len(system.strategy_pool) - mtf_start_id
    print(f"✅ Found {mtf_count} MTF strategies")
    
    # ============================================================================
    # PHASE 3: BACKTESTING (SINGLE PASS - ENHANCED METHOD ONLY)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 3: STRATEGY BACKTESTING")
    print("="*80)
    
    for strategy_key in list(system.strategy_pool.keys()):
        strategy = system.strategy_pool[strategy_key]
        strategy_type = strategy.get('type', 'single_signal')

        try:
            # --- Path for all MTF Strategies (A, B, C... N, Confluence) ---
            if strategy_type.startswith('mtf_'):
                
                # Use legacy backtester for A, B, C (as it has specific logic)
                if strategy_type in ['mtf_mode_a', 'mtf_mode_b', 'mtf_mode_c']:
                    strategy = system.backtest_mtf_strategy(strategy)
                
                # Use legacy backtester for F, G (as it has specific logic)
                elif strategy_type in ['mtf_mode_f', 'mtf_mode_g']:
                     strategy = system.backtest_mtf_strategy_enhanced(strategy)

                # Use NEW generic backtester for all other MTF modes
                # (D, E, H, I, J, K, L, M, N, Confluence)
                else:
                    strategy = system.backtest_mtf_strategy_generic(strategy)
            
            # --- Path for Single-Signal Strategies ---
            elif strategy_type == 'single_signal':
                strategy = system.regime_aware_backtesting(strategy_key, strategy)
                strategy = system.realistic_backtest(strategy)
            
            # --- Path for Combination Strategies ---
            elif strategy_type == 'combination':
                # Combination strategies have no backtester, so we skip them
                pass
            
            # Update the strategy in the pool
            system.strategy_pool[strategy_key] = strategy

        except Exception as e:
            import traceback
            print(f"⚠️ Error backtesting {strategy_key} (Type: {strategy_type}): {e}")
            print(traceback.format_exc())

    print(f"✅ Backtested {len(system.strategy_pool)} strategies")

    # ============================================================================
    # PHASE 4: ADVANCED ANALYTICS
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 4: ADVANCED ANALYTICS")
    print("="*80)
    
    # Strategy correlation analysis (for portfolio construction)
    print("\n📈 Building strategy correlation matrix...")
    correlation_results = system.build_strategy_correlation_matrix()
    if correlation_results:
        print(f"✅ Correlation analysis complete")
        if 'diversified_portfolios' in correlation_results:
            print(f"   Found {len(correlation_results['diversified_portfolios'])} diversified portfolios")
    
    # Pattern effectiveness analysis
    print("\n🎯 Analyzing pattern effectiveness...")
    system.generate_pattern_effectiveness_report()
    
    # Sideways market analysis
    print("\n📉 Analyzing sideways conditions...")
    sideways_summary = {}
    for pair_tf, df in system.all_dataframes.items():
        try:
            sideways_mask = system.identify_sideways_conditions(df)
            sideways_pct = sideways_mask.mean()
            sideways_summary[pair_tf] = sideways_pct
        except Exception as e:
            print(f"⚠️ {pair_tf}: Error analyzing sideways conditions")
    
    print(f"✅ Analyzed sideways conditions for {len(sideways_summary)} datasets")
    
    # ============================================================================
    # PHASE 5: SL/TP CALCULATION (FOR LIVE TRADING ONLY)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 5: CALCULATING SL/TP FOR LIVE TRADING")
    print("="*80)
    print("⚠️  Note: SL/TP calculated using LAST available bar (for live signals)")
    
    # Only calculate SL/TP for top-performing MTF strategies
    mtf_strategies = {
        k: v for k, v in system.strategy_pool.items() 
        if v.get('type', '').startswith('mtf_')
        and v.get('backtest_win_rate', v.get('discovered_accuracy', 0)) >= 0.60  # Use backtest or discovery accuracy
    }
    
    # Sort by performance
    sorted_mtf = sorted(
        mtf_strategies.items(), 
        key=lambda x: x[1].get('performance_score', 0), 
        reverse=True
    )[:30]  # Top 30 only
    
    enhanced_count = 0
    for strat_id, strategy in sorted_mtf:
        enhanced_strategy = system.calculate_mtf_sl_tp(strategy)
        
        if 'stop_loss' in enhanced_strategy and 'take_profit' in enhanced_strategy:
            system.strategy_pool[strat_id] = enhanced_strategy
            enhanced_count += 1
    
    print(f"✅ Enhanced {enhanced_count}/{len(sorted_mtf)} MTF strategies with SL/TP")
    
    # ============================================================================
    # PHASE 6: REPORTING & EXPORT
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 6: GENERATING REPORTS")
    print("="*80)
    
    # Console reports
    system.print_strategy_report(top_n=20)
    system.generate_mtf_strategy_report()
    
    # File exports
    print("\n📁 Exporting results...")
    export_files = []
    
    try:
        # Strategy pool (all strategies)
        system.save_strategies_to_file('strategy_pool.json')
        export_files.append('strategy_pool.json')
        
        # Summary CSV
        system.export_summary_csv('strategy_summary.csv')
        export_files.append('strategy_summary.csv')
        
        # MTF strategies with SL/TP (live trading ready)
        system.export_mtf_strategies_csv('mtf_strategies_with_sltp.csv')
        export_files.append('mtf_strategies_with_sltp.csv')
        
        # Live monitoring config
        system.export_live_monitoring_config('live_monitor_config.json')
        export_files.append('live_monitor_config.json')
        
        print(f"✅ Exported {len(export_files)} files")
        
    except Exception as e:
        print(f"⚠️ Export error: {e}")
    
    # ============================================================================
    # PHASE 7: FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("STRATEGY DISCOVERY COMPLETE")
    print("="*80)
    
    # Strategy breakdown
    single_signal_count = len([s for s in system.strategy_pool.values() 
                              if s.get('type', 'single_signal') == 'single_signal'])
    combination_count = len([s for s in system.strategy_pool.values() 
                            if s.get('type') == 'combination'])
    mtf_count = len([s for s in system.strategy_pool.values() 
                    if s.get('type', '').startswith('mtf_')])
    
    print(f"\n📊 Strategy Breakdown:")
    print(f"   • Single-Timeframe: {single_signal_count}")
    print(f"   • Combination: {combination_count}")
    print(f"   • Multi-Timeframe: {mtf_count}")
    print(f"   • Total: {len(system.strategy_pool)}")
    
    # High-quality strategies
    high_quality = len([s for s in system.strategy_pool.values() 
                       if s.get('backtest_win_rate', s.get('discovered_accuracy', 0)) >= 0.60])
    live_ready = len([s for s in system.strategy_pool.values() 
                     if 'stop_loss' in s and 'take_profit' in s])
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   • High Win Rate (≥60%): {high_quality}")
    print(f"   • Live Trading Ready (with SL/TP): {live_ready}")
    
    # Performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n⏱️  Execution Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"   Datasets Processed: {len(system.all_dataframes)}")
    
    try:
        perf_stats = system.get_performance_stats()
        cache_hit_rate = perf_stats.get('cache_hit_rate', 0)
        print(f"   Cache Efficiency: {cache_hit_rate:.1%}")
    except:
        pass
    
    print(f"\n📁 Exported Files:")
    for file in export_files:
        print(f"   • {file}")
    
    print("\n✅ Strategy Discovery Complete!")
