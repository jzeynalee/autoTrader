"""
export_regime_data_for_stats.py
Export BTC regime data to CSV files for statistical analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Import your existing modules
from analysis_advanced_regime import AdvancedRegimeDetectionSystem
from regime_instance_engine import RegimeInstanceEngine
from regime_data_access_sqlite import RegimeDataAccess

def export_btc_data_for_statistics(
    df: pd.DataFrame,
    output_dir: str = './data_exports',
    pair: str = 'BTCUSDT',
    timeframe: str = '1h'
):
    """
    Export all relevant data to CSV files for statistical testing.
    
    Args:
        df: Raw OHLCV DataFrame
        output_dir: Directory to save CSV files
        pair: Trading pair name
        timeframe: Timeframe string
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"Exporting data to {output_dir}...")
    
    # ===== 1. Export Raw OHLCV Data =====
    raw_filename = f'{output_dir}/01_raw_ohlcv_{pair}_{timeframe}_{timestamp}.csv'
    df.to_csv(raw_filename, index=True)
    print(f"✓ Exported raw OHLCV: {raw_filename} ({len(df)} rows)")
    
    # ===== 2. Run Regime Detection =====
    detector = AdvancedRegimeDetectionSystem(atr_multiplier=1.0)
    df_with_regimes = detector.detect_advanced_market_regimes(df.copy())
    
    # Export data with regime labels
    regimes_filename = f'{output_dir}/02_data_with_regimes_{pair}_{timeframe}_{timestamp}.csv'
    df_with_regimes.to_csv(regimes_filename, index=True)
    print(f"✓ Exported data with regimes: {regimes_filename} ({len(df_with_regimes)} rows)")
    
    # ===== 3. Export Regime Instance Summary =====
    # Get unique regime instances
    regime_instances = []
    for instance_id in df_with_regimes['regime_instance_id'].unique():
        if pd.isna(instance_id):
            continue
            
        instance_data = df_with_regimes[df_with_regimes['regime_instance_id'] == instance_id]
        
        regime_instances.append({
            'instance_id': instance_id,
            'regime_type': instance_data['regime_type'].iloc[0],
            'start_time': instance_data.index[0],
            'end_time': instance_data.index[-1],
            'duration_hours': len(instance_data),
            'start_price': instance_data['close'].iloc[0],
            'end_price': instance_data['close'].iloc[-1],
            'price_change_pct': ((instance_data['close'].iloc[-1] / instance_data['close'].iloc[0]) - 1) * 100,
            'max_price': instance_data['close'].max(),
            'min_price': instance_data['close'].min(),
            'volatility': instance_data['close'].pct_change().std() * 100,
            'volume_mean': instance_data['volume'].mean(),
            'bar_count': len(instance_data)
        })
    
    instances_df = pd.DataFrame(regime_instances)
    instances_filename = f'{output_dir}/03_regime_instances_summary_{pair}_{timeframe}_{timestamp}.csv'
    instances_df.to_csv(instances_filename, index=False)
    print(f"✓ Exported regime instances: {instances_filename} ({len(instances_df)} instances)")
    
    # ===== 4. Export Full Regime Instances (with all features) =====
    # Build complete regime instances using the engine
    engine = RegimeInstanceEngine()
    full_instances = []
    
    for instance_id in df_with_regimes['regime_instance_id'].unique():
        if pd.isna(instance_id):
            continue
        
        instance_data = df_with_regimes[df_with_regimes['regime_instance_id'] == instance_id]
        bar_indices = instance_data.index.tolist()
        
        try:
            regime_instance = engine.build_regime_instance(
                df=df_with_regimes,
                bar_indices=bar_indices,
                pair=pair,
                timeframe=timeframe
            )
            
            # Convert to dictionary for CSV export
            instance_dict = {
                'instance_id': regime_instance.instance_id,
                'pair': regime_instance.pair,
                'timeframe': regime_instance.timeframe,
                'start_time': regime_instance.start_time,
                'end_time': regime_instance.end_time,
                'duration_hours': regime_instance.duration_hours,
                'swing_count': regime_instance.swing_count,
                'avg_swing_magnitude_pct': regime_instance.avg_swing_magnitude_pct,
                'dominant_structure': regime_instance.dominant_structure,
                'price_change_pct': regime_instance.price_change_pct,
                'max_drawdown_pct': regime_instance.max_drawdown_pct,
                'max_runup_pct': regime_instance.max_runup_pct,
                'volatility_mean': regime_instance.volatility_mean,
                'volatility_std': regime_instance.volatility_std,
                'volatility_trend': regime_instance.volatility_trend,
                'rsi_mean': regime_instance.rsi_mean,
                'rsi_std': regime_instance.rsi_std,
                'rsi_trend': regime_instance.rsi_trend,
                'macd_hist_mean': regime_instance.macd_hist_mean,
                'macd_crossovers': regime_instance.macd_crossovers,
                'adx_mean': regime_instance.adx_mean,
                'adx_trend': regime_instance.adx_trend,
                'volume_mean': regime_instance.volume_mean,
                'volume_trend': regime_instance.volume_trend,
                'volume_spikes': regime_instance.volume_spikes,
                'higher_highs': regime_instance.higher_highs,
                'higher_lows': regime_instance.higher_lows,
                'lower_highs': regime_instance.lower_highs,
                'lower_lows': regime_instance.lower_lows,
                'structure_breaks_bullish': regime_instance.structure_breaks_bullish,
                'structure_breaks_bearish': regime_instance.structure_breaks_bearish,
                'pullback_count': regime_instance.pullback_count,
                'avg_pullback_depth_pct': regime_instance.avg_pullback_depth_pct,
                'failed_pullbacks': regime_instance.failed_pullbacks,
                'next_1d_return_pct': regime_instance.next_1d_return_pct,
                'next_3d_return_pct': regime_instance.next_3d_return_pct,
                'next_7d_return_pct': regime_instance.next_7d_return_pct,
                'max_favorable_excursion_1d': regime_instance.max_favorable_excursion_1d,
                'max_adverse_excursion_1d': regime_instance.max_adverse_excursion_1d,
                'consistency_score': regime_instance.consistency_score,
                'predictability_score': regime_instance.predictability_score,
                'bar_count': len(regime_instance.bar_indices)
            }
            
            full_instances.append(instance_dict)
            
        except Exception as e:
            print(f"  Warning: Could not build instance {instance_id}: {e}")
            continue
    
    if full_instances:
        full_instances_df = pd.DataFrame(full_instances)
        full_filename = f'{output_dir}/04_full_regime_instances_{pair}_{timeframe}_{timestamp}.csv'
        full_instances_df.to_csv(full_filename, index=False)
        print(f"✓ Exported full regime instances: {full_filename} ({len(full_instances_df)} instances)")
    
    # ===== 5. Export Regime Type Statistics =====
    regime_stats = df_with_regimes.groupby('regime_type').agg({
        'close': ['count', 'mean', 'std'],
        'volume': ['mean', 'std'],
        'regime_instance_id': 'nunique'
    }).round(4)
    
    regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns.values]
    regime_stats_filename = f'{output_dir}/05_regime_type_stats_{pair}_{timeframe}_{timestamp}.csv'
    regime_stats.to_csv(regime_stats_filename)
    print(f"✓ Exported regime type stats: {regime_stats_filename}")
    
    # ===== 6. Export Returns Analysis =====
    if full_instances:
        returns_analysis = {
            'regime_type': [],
            'count': [],
            'avg_1d_return': [],
            'avg_3d_return': [],
            'avg_7d_return': [],
            'win_rate_1d': [],
            'median_1d_return': [],
            'std_1d_return': []
        }
        
        for regime_type in full_instances_df['dominant_structure'].unique():
            subset = full_instances_df[full_instances_df['dominant_structure'] == regime_type]
            
            returns_analysis['regime_type'].append(regime_type)
            returns_analysis['count'].append(len(subset))
            returns_analysis['avg_1d_return'].append(subset['next_1d_return_pct'].mean())
            returns_analysis['avg_3d_return'].append(subset['next_3d_return_pct'].mean())
            returns_analysis['avg_7d_return'].append(subset['next_7d_return_pct'].mean())
            returns_analysis['win_rate_1d'].append((subset['next_1d_return_pct'] > 0).mean() * 100)
            returns_analysis['median_1d_return'].append(subset['next_1d_return_pct'].median())
            returns_analysis['std_1d_return'].append(subset['next_1d_return_pct'].std())
        
        returns_df = pd.DataFrame(returns_analysis)
        returns_filename = f'{output_dir}/06_returns_by_regime_{pair}_{timeframe}_{timestamp}.csv'
        returns_df.to_csv(returns_filename, index=False)
        print(f"✓ Exported returns analysis: {returns_filename}")
    
    # ===== 7. Export Summary Metadata =====
    summary = {
        'export_timestamp': timestamp,
        'pair': pair,
        'timeframe': timeframe,
        'total_bars': len(df),
        'date_range_start': str(df.index[0]),
        'date_range_end': str(df.index[-1]),
        'total_regime_instances': len(instances_df),
        'unique_regime_types': df_with_regimes['regime_type'].nunique(),
        'files_exported': 6
    }
    
    summary_df = pd.DataFrame([summary])
    summary_filename = f'{output_dir}/00_export_summary_{timestamp}.csv'
    summary_df.to_csv(summary_filename, index=False)
    print(f"✓ Exported summary: {summary_filename}")
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Total files exported: 7")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles ready for statistical analysis:")
    print(f"  - Raw OHLCV data")
    print(f"  - Data with regime labels")
    print(f"  - Regime instance summaries")
    print(f"  - Full regime instances with features")
    print(f"  - Regime type statistics")
    print(f"  - Returns analysis by regime")
    print(f"  - Export metadata")
    
    return {
        'output_dir': output_dir,
        'timestamp': timestamp,
        'summary': summary
    }


# ===== Example Usage =====
if __name__ == "__main__":
    # Example: Load your BTC data
    # df = load_test_data('BTCUSDT_1h', days=365)  # Your actual data loading
    
    # For demonstration, create sample data structure
    print("Load your actual BTC data, then call:")
    print("export_btc_data_for_statistics(df, pair='BTCUSDT', timeframe='1h')")
    print("\nThis will create 7 CSV files ready for statistical testing:")
    print("  1. Raw OHLCV data")
    print("  2. Data with regime labels")
    print("  3. Regime instance summaries")
    print("  4. Full regime instances (all features)")
    print("  5. Regime type statistics")
    print("  6. Returns analysis by regime type")
    print("  7. Export metadata summary")
