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

# Note: We keep the relative imports here, assuming execution via autoTrader.main
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
    print(f"Warning: discovery_mapping.py import failed: {e}")
    print("FATAL: Cannot load required mapping file. System cannot proceed.")
    MAPPER_AVAILABLE = False
    # Do not exit(1) here; the main entry point will handle the shutdown.
    # We proceed by raising an error inside the function if called.

from .analysis_trend import TrendAnalysisSystem
from .analysis_pullback import PullbackAnalysisSystem
from .analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .analysis_confluence_scoring import ConfluenceScoringSystem
from .regime_based_discovery_system import RegimeStrategyDiscovery
from .core import StrategyDiscoverySystem
from .discovery_modes import DiscoveryModesMixin
from .regime_data_export_integration import (
    RegimeDataExportIntegration,
    add_export_to_strategy_pipeline
)
from .regime_instance_engine import RegimeInstanceEngine
from .regime_data_access_sqlite import RegimeDataAccess
from .regime_statistical_analysis_sqlite import RegimeStatisticalAnalyzer

# ============================================================================
# NEW ENTRY FUNCTION: RUN STRATEGY DISCOVERY
# ============================================================================

def run_strategy_discovery(db_connector):
    """
    Orchestrates the entire strategy discovery, backtesting, and reporting pipeline.
    
    Args:
        db_connector: An instantiated DatabaseConnector object.
    """
    if not MAPPER_AVAILABLE:
        print("❌ ERROR: Discovery mapping failed to load. Aborting strategy pipeline.")
        return

    start_time = time.time()
    export_files = [] # Moved from Phase 6 to be available for all phases
    
    # ============================================================================
    # PHASE 1: INITIALIZATION & DATA LOADING (FROM DB)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 1: STRATEGY PIPELINE INITIALIZATION (Loading from DB)")
    print("="*80)
    
    # 1. Instantiate Core System (Passing DB Connector and initial config)
    system = StrategyDiscoverySystem(
        # Note: data_dir is now just a config parameter, not a file path
        data_dir='./data/ingestion',
        lookforward_periods=5,
        price_threshold=0.005,
        n_jobs=-1,
        db_connector=db_connector # <-- Database dependency injection
    )
    
    # 2. Inject Analysis Systems (The modules are imported, but must be attached)
    #    We rely on the top-level main.py passing instantiated systems to run_strategy_discovery,
    #    or, more simply, we instantiate them here as they are lightweight classes.
    try:
        system.trend_analyzer    = TrendAnalysisSystem()
        system.pullback_analyzer = PullbackAnalysisSystem()
        system.regime_detector   = AdvancedRegimeDetectionSystem()
        system.confluence_scorer = ConfluenceScoringSystem()
    except Exception as e:
        print(f"❌ ERROR: Failed to instantiate Analysis Systems: {e}")
        return
    

    # 3. Load Data from DB (Pivoting from file I/O)
    # Note: We must change the call from system.load_data() to system.load_data_from_db()
    #       (assuming the latter exists in core.py and uses db_connector).
    #       However, for simplicity in this refactor, we retain the original call 
    #       and assume core.py's load_data is now database-aware. 
    #       (The correct approach is load_data_from_db, but we use the existing method name.)

    if not system.load_data_from_db(): 
        print("❌ Failed to load data from DB. Aborting.")
        return
    
    # === EXPORT CONFIGURATION ===
    export_config = {
        'enabled': True,  # Set to False to disable exports
        'output_dir': './data/exports',  # Where to save CSV files
        'pairs': ['BTCUSDT'],  # Which pairs to export
        'timeframes': ['4h'],  # Which timeframes to export
        'export_after_phase': 2,  # Export after Phase 2 (feature engineering)
        'run_regime_detection': False  # False = faster (detection already done)
    }
    
    print(f"\n📊 Export Configuration:")
    print(f"   Enabled: {export_config['enabled']}")
    print(f"   Output: {export_config['output_dir']}")
    print(f"   Pairs: {', '.join(export_config['pairs'])}")
    print(f"   Timeframes: {', '.join(export_config['timeframes'])}")
    # === END EXPORT CONFIGURATION ===
    
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
        # Assuming system.enhanced_market_regime_detection still works on the loaded dataframe
        system.all_dataframes[pair_tf] = system.enhanced_market_regime_detection(df)
    
    print(f"✅ Loaded {len(system.all_dataframes)} datasets")

    # ============================================================================
    # OPTIONAL: EXPORT DATA FOR STATISTICAL ANALYSIS
    # ============================================================================
    if export_config.get('enabled', False):
        print("\n" + "="*80)
        print("EXPORTING DATA FOR STATISTICAL ANALYSIS")
        print("="*80)
        
        try:
            export_results = add_export_to_strategy_pipeline(system, export_config)
            
            if export_results:
                print(f"\n✅ Exported {len(export_results)} dataset(s)")
                print(f"   Location: {export_config['output_dir']}")
                print(f"   Files: 10 CSV files per dataset")
                
                # Show what was exported
                for key, result in export_results.items():
                    if 'summary' in result:
                        print(f"\n   {key}:")
                        print(f"      - Bars: {result['summary']['total_bars']}")
                        print(f"      - Indicators: {result['summary']['total_indicators']}")
                        print(f"      - Regime instances: {result['summary'].get('regime_instances', 0)}")
            else:
                print("⚠️  No datasets exported (check configuration)")
                
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
            print("   Continuing with strategy discovery...")
            import traceback
            traceback.print_exc()
    else:
        print("\n📊 Data export disabled in configuration")
    # === END EXPORT CALL ===
    

    # ============================================================================
    # --- NEW PHASE 1.5: REGIME STRATEGY PLAYBOOK DISCOVERY ---
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 1.5: REGIME INSTANCE DISCOVERY & STATISTICAL ANALYSIS")
    print("="*80)

    # Initialize components
    instance_engine = RegimeInstanceEngine(
        min_bars_per_instance=24,
        max_bars_per_instance=168,
        volatility_threshold=0.25,
    )

    regime_dao = RegimeDataAccess(db_connector)
    analyzer = RegimeStatisticalAnalyzer(regime_dao)

    # Discover and store instances
    all_instances = {}
    total_instances = 0

    for pair_tf, df in system.all_dataframes.items():
        pair, timeframe = pair_tf.rsplit('_', 1)
        
        instances = instance_engine.discover_instances(df, pair, timeframe)
        
        for instance in instances:
            regime_dao.store_regime_instance(instance)
        
        all_instances[pair_tf] = instances
        total_instances += len(instances)

    print(f"\n✅ Discovered & Stored: {total_instances} regime instances")

    # Run statistical analysis on top indicators
    print("\n🔬 Running Statistical Analysis...")

    top_indicators = [
        'rsi', 'macd_hist', 'ppo', 'adx', 'bb_width', 
        'obv', 'volume_zscore', 'atr_percent'
    ]

    indicator_analysis = {}

    for indicator in top_indicators:
            try:
                result = analyzer.analyze_indicator_causality(indicator)
                if 'error' not in result:
                    indicator_analysis[indicator] = result
                    # UPDATED: V2 keys handling + V1 compatibility keys added in drop-in
                    power = result.get('predictive_power', 0) # Added in drop-in V2
                    rec = result.get('recommendation', 'N/A')
                    mi = result.get('mutual_info', 0)
                    print(f"  {indicator:20s} | Power: {power:5.1f} | MI: {mi:.3f} | {rec}")
            except Exception as e:
                print(f"  ⚠️ Error analyzing {indicator}: {e}")

    # Find optimal combinations
    print("\n🎯 Finding Optimal Indicator Combinations...")
    # UPDATED: Handle return type (Dict instead of List)
    combo_result = analyzer.find_optimal_indicator_combinations(max_indicators=5)
    combinations = combo_result.get('combinations_top', [])

    if combinations:
        print(f"\nTop 5 Indicator Combinations:")
        for i, combo in enumerate(combinations[:5], 1):
            indicators_str = ' + '.join(combo['indicators'])
            print(f"  {i}. {indicators_str}")
            print(f"     Win Rate: {combo['win_rate']:.1f}% | Avg Return: {combo['avg_return']:.2f}% | Samples: {combo['sample_size']}")

    # Store analysis results
    if db_connector:
        try:
            db_connector.execute("""
                INSERT INTO statistical_experiments (
                    experiment_name,
                    experiment_type,
                    parameters_json,
                    results_json,
                    sample_size,
                    run_time
                ) VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                'Indicator Causality Analysis',
                'causality',
                json.dumps({'indicators': top_indicators}),
                json.dumps(indicator_analysis),
                total_instances
            ))
        except Exception as e:
            print(f"  ⚠️ Failed to store analysis results: {e}")

    print("\n✅ Statistical Analysis Complete")
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
    # PHASE 6: REPORTING & EXPORT (Now exports via DB and files)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 6: GENERATING REPORTS AND PERSISTING TO DB")
    print("="*80)
    
    # Console reports
    system.print_strategy_report(top_n=20)
    system.generate_mtf_strategy_report()
    
    # File exports (Retained for quick inspection)
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


# ----------------------------------------------------------------------------
# The contents of the original if __name__ == "__main__": block is retained 
# here for review, but will be replaced by the run_strategy_discovery function 
# that is called by the top-level autoTrader/main.py.
# ----------------------------------------------------------------------------