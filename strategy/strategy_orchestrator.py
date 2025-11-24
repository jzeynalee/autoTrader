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

from ..logger import setup_logging

# Initialize global logger (can be reconfigured)
logger = setup_logging()

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
    export_files = [] 
    
    # ============================================================================
    # PHASE 1: INITIALIZATION & DATA LOADING (FROM DB)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 1: STRATEGY PIPELINE INITIALIZATION (Loading from DB)")
    print("="*80)
    
    # 1. Instantiate Core System
    system = StrategyDiscoverySystem(
        data_dir='./data/ingestion',
        lookforward_periods=5,
        price_threshold=0.005,
        n_jobs=-1,
        db_connector=db_connector 
    )
    
    # 2. Inject Analysis Systems
    try:
        system.trend_analyzer    = TrendAnalysisSystem()
        system.pullback_analyzer = PullbackAnalysisSystem()
        system.regime_detector   = AdvancedRegimeDetectionSystem()
        system.confluence_scorer = ConfluenceScoringSystem()
    except Exception as e:
        print(f"❌ ERROR: Failed to instantiate Analysis Systems: {e}")
        return
    

    # 3. Load Data from DB
    if not system.load_data_from_db(): 
        print("❌ Failed to load data from DB. Aborting.")
        return
    
    # Optimize memory usage once at start
    system.optimize_data_loading()
    
    # Detect market regimes
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
    # PHASE 1.5: REGIME STRATEGY PLAYBOOK DISCOVERY (THE BRAIN)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 1.5: REGIME INSTANCE DISCOVERY & STATISTICAL ANALYSIS")
    print("="*80)

    # 1. Initialize Components
    instance_engine = RegimeInstanceEngine(
        min_bars_per_instance=24,
        max_bars_per_instance=168,
        volatility_threshold=0.25,
    )
    regime_dao = RegimeDataAccess(db_connector)
    analyzer = RegimeStatisticalAnalyzer(regime_dao)
    
    # 2. Initialize The Strategy Discovery Brain
    # This component unifies detection (System) and validation (Analyzer)
    strategy_discoverer = RegimeStrategyDiscovery(system.regime_detector)
    strategy_discoverer.stats_analyzer = analyzer  # Inject analyzer for statistical validation

    # 3. Discover and store instances
    total_instances = 0
    for pair_tf, df in system.all_dataframes.items():
        pair, timeframe = pair_tf.rsplit('_', 1)
        instances = instance_engine.discover_instances(df, pair, timeframe)
        for instance in instances:
            regime_dao.store_regime_instance(instance)
        total_instances += len(instances)

    print(f"\n✅ Discovered & Stored: {total_instances} regime instances")

    # 4. Run Statistical Analysis (The Math)
    print("\n🔬 Running Statistical Analysis on Indicators...")
    top_indicators = ['rsi', 'macd_hist', 'ppo', 'adx', 'bb_width', 'obv', 'volume_zscore', 'atr_percent']
    
    indicator_analysis = {}
    for indicator in top_indicators:
        try:
            # This calculates P-values and correlations
            result = analyzer.analyze_indicator_causality(indicator)
            if 'error' not in result:
                indicator_analysis[indicator] = result
                print(f"  Analyzed {indicator}")
        except Exception as e:
            print(f"  ⚠️ Error analyzing {indicator}: {e}")

    # 5. Generate Strategy Playbook (The Rules)
    print("\n📘 Generating Regime Strategy Playbook...")
    try:
        # This uses the analyzed stats to filter valid indicators for each regime
        playbook = strategy_discoverer.discover_strategies()
        strategy_discoverer.print_repository_summary()
        
        # Convert Playbook to Strategies for the Pool
        playbook_strategies_count = 0
        for regime_id, rules in playbook.items():
            # Create a strategy object compatible with our backtester
            if not rules.get('confirming_indicators') and not rules.get('strategy_patterns'):
                continue
                
            strat_entry = {
                'id': f"PLAYBOOK_{regime_id}",
                'type': 'regime_playbook',
                'name': f"Regime {rules.get('regime_label', regime_id)} Strategy",
                'regime_id': regime_id,
                'regime_context': rules.get('regime_label', 'Unknown'),
                'trend_direction': rules.get('trend_direction', 'neutral'),
                'entry_conditions': list(rules['confirming_indicators']),
                'patterns': list(rules['strategy_patterns']),
                'status': 'candidate'
            }
            
            # Add to system pool
            system.strategy_pool[strat_entry['id']] = strat_entry
            playbook_strategies_count += 1
            
        print(f"✅ Generated {playbook_strategies_count} strategies from Regime Playbook")
        
    except Exception as e:
        print(f"⚠️ Failed to generate playbook: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # PHASE 2: STRATEGY DISCOVERY (EXECUTION MODES)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 2: STRATEGY DISCOVERY (MODES)")
    print("="*80)
    
    # Single-timeframe strategies
    print("\n📊 Discovering single-timeframe strategies...")
    system.discover_multi_signal_strategies()
    
    # Multi-timeframe strategies
    # Note: Modes A-H have been deprecated. This calls surviving modes (J, K, L, M, N, Confluence)
    print("\n🔄 Discovering multi-timeframe strategies...")
    mtf_start_id = len(system.strategy_pool)
    system.discover_mtf_strategies()
    mtf_count = len(system.strategy_pool) - mtf_start_id
    print(f"✅ Found {mtf_count} MTF strategies")
    
    # ============================================================================
    # PHASE 3: BACKTESTING (CONSOLIDATED & CLEANED)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 3: STRATEGY BACKTESTING")
    print("="*80)
    
    for strategy_key in list(system.strategy_pool.keys()):
        strategy = system.strategy_pool[strategy_key]
        strategy_type = strategy.get('type', 'single_signal')

        try:
            # --- Path 1: MTF Strategies (Surviving Modes) ---
            if strategy_type.startswith('mtf_') or strategy_type == 'mtf_confluence':
                # All remaining MTF modes use the modern generic backtester
                # Modes A-H are gone, so no need for specific handlers
                strategy = system.backtest_mtf_strategy_generic(strategy)
            
            # --- Path 2: Regime Playbook Strategies (New) ---
            elif strategy_type == 'regime_playbook':
                # These behave like single-signal strategies but filtered by regime
                strategy = system.regime_aware_backtesting(strategy_key, strategy)
            
            # --- Path 3: Single-Signal Strategies ---
            elif strategy_type == 'single_signal':
                strategy = system.regime_aware_backtesting(strategy_key, strategy)
                strategy = system.realistic_backtest(strategy)
            
            # --- Path 4: Combination Strategies ---
            elif strategy_type == 'combination':
                pass
            
            # Update the strategy in the pool
            system.strategy_pool[strategy_key] = strategy

        except Exception as e:
            # Don't spam console if it's just a data missing warning
            if "Insufficient data" not in str(e):
                print(f"⚠️ Error backtesting {strategy_key} ({strategy_type}): {e}")

    print(f"✅ Backtested {len(system.strategy_pool)} strategies")

    # ============================================================================
    # PHASE 4: ADVANCED ANALYTICS
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 4: ADVANCED ANALYTICS")
    print("="*80)
    
    # Strategy correlation analysis
    print("\n📈 Building strategy correlation matrix...")
    correlation_results = system.build_strategy_correlation_matrix()
    if correlation_results and 'diversified_portfolios' in correlation_results:
        print(f"   Found {len(correlation_results['diversified_portfolios'])} diversified portfolios")
    
    # Pattern effectiveness analysis
    print("\n🎯 Analyzing pattern effectiveness...")
    system.generate_pattern_effectiveness_report()
    
    # ============================================================================
    # PHASE 5: SL/TP CALCULATION (FOR LIVE TRADING)
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 5: CALCULATING SL/TP FOR LIVE TRADING")
    print("="*80)
    
    # Only calculate SL/TP for top-performing strategies
    # Look at both backtest win rate and discovery accuracy
    top_strategies = {
        k: v for k, v in system.strategy_pool.items() 
        if v.get('backtest_win_rate', v.get('discovered_accuracy', 0)) >= 0.55
    }
    
    # Sort by performance and take top 30
    sorted_strats = sorted(
        top_strategies.items(), 
        key=lambda x: x[1].get('performance_score', 0), 
        reverse=True
    )[:30]
    
    enhanced_count = 0
    for strat_id, strategy in sorted_strats:
        # Use MTF calculation for MTF strategies, standard for others
        if strategy.get('type', '').startswith('mtf_'):
            enhanced_strategy = system.calculate_mtf_sl_tp(strategy)
        else:
            # Fallback to standard logic if needed, or implement calculate_standard_sl_tp
            enhanced_strategy = strategy # Placeholder if standard calc not available
            
        if 'stop_loss' in enhanced_strategy and 'take_profit' in enhanced_strategy:
            system.strategy_pool[strat_id] = enhanced_strategy
            enhanced_count += 1
    
    print(f"✅ Enhanced {enhanced_count}/{len(sorted_strats)} top strategies with SL/TP")
    
    # ============================================================================
    # PHASE 6: REPORTING & EXPORT
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 6: GENERATING REPORTS")
    print("="*80)
    
    # Console reports
    system.print_strategy_report(top_n=20)
    
    # File exports
    print("\n📁 Exporting results...")
    try:
        system.save_strategies_to_file('strategy_pool.json')
        export_files.append('strategy_pool.json')
        
        system.export_summary_csv('strategy_summary.csv')
        export_files.append('strategy_summary.csv')
        
        system.export_mtf_strategies_csv('mtf_strategies_with_sltp.csv')
        export_files.append('mtf_strategies_with_sltp.csv')
        
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
    playbook_count = len([s for s in system.strategy_pool.values() if s.get('type') == 'regime_playbook'])
    mtf_count = len([s for s in system.strategy_pool.values() if s.get('type', '').startswith('mtf_')])
    
    print(f"\n📊 Strategy Breakdown:")
    print(f"   • Regime Playbook: {playbook_count}")
    print(f"   • Multi-Timeframe: {mtf_count}")
    print(f"   • Total: {len(system.strategy_pool)}")
    
    # Quality Metrics
    high_quality = len([s for s in system.strategy_pool.values() 
                       if s.get('backtest_win_rate', 0) >= 0.60])
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   • High Win Rate (≥60%): {high_quality}")
    
    # Performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n⏱️  Execution Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("\n✅ Strategy Discovery Complete!")