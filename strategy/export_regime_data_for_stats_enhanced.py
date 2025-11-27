"""
export_regime_data_for_stats_enhanced.py
ENHANCED: Export BTC regime data with ALL features for statistical analysis.

Includes:
- Enriched DF with all indicators
- Chart patterns
- Pullback states  
- Price action status
- HMM states
- XGB predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

# Import project modules
from .analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .regime_instance_engine import RegimeInstanceEngine
from .regime_data_access_sqlite import RegimeDataAccess


class EnhancedRegimeDataExporter:
    """
    Comprehensive data exporter for statistical analysis.
    Integrates with the regime detection system to export ALL features.
    """
    
    def __init__(self, output_dir: str = './data_exports'):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
    def export_all_regime_data(
        self,
        df: pd.DataFrame,
        pair: str = 'BTCUSDT',
        timeframe: str = '1h',
        run_regime_detection: bool = True
    ) -> dict:
        """
        Export comprehensive regime data to multiple CSV files.
        
        Args:
            df: Raw OHLCV DataFrame
            pair: Trading pair
            timeframe: Timeframe string
            run_regime_detection: Whether to run full regime detection (slow but complete)
            
        Returns:
            Dictionary with export summary and file paths
        """
        print("="*80)
        print("ENHANCED REGIME DATA EXPORT FOR STATISTICAL ANALYSIS")
        print("="*80)
        print(f"Pair: {pair} | Timeframe: {timeframe}")
        print(f"Data range: {df.index[0]} to {df.index[-1]} ({len(df)} bars)")
        print()
        
        export_files = {}
        
        # =====================================================================
        # FILE 1: Raw OHLCV Data
        # =====================================================================
        print("📊 [1/10] Exporting raw OHLCV data...")
        raw_file = f'{self.output_dir}/01_raw_ohlcv_{pair}_{timeframe}_{self.timestamp}.csv'
        df.to_csv(raw_file, index=True)
        export_files['raw_ohlcv'] = raw_file
        print(f"   ✓ Saved: {raw_file} ({len(df)} bars)")
        
        # =====================================================================
        # FILE 2: Enriched DF with ALL Indicators
        # =====================================================================
        print("\n🔧 [2/10] Creating enriched DataFrame with all indicators...")
        
        if run_regime_detection:
            detector = AdvancedRegimeDetectionSystem(atr_multiplier=1.0)
            df_enriched = detector.detect_advanced_market_regimes(df.copy())
        else:
            df_enriched = df.copy()
        
        # List all available columns for reference
        indicator_cols = [col for col in df_enriched.columns if col not in 
                         ['open', 'high', 'low', 'close', 'volume']]
        
        enriched_file = f'{self.output_dir}/02_enriched_with_indicators_{pair}_{timeframe}_{self.timestamp}.csv'
        df_enriched.to_csv(enriched_file, index=True)
        export_files['enriched_df'] = enriched_file
        print(f"   ✓ Saved: {enriched_file}")
        print(f"   ✓ Added {len(indicator_cols)} indicator columns")
        
        # =====================================================================
        # FILE 3: Indicator Columns Reference
        # =====================================================================
        print("\n📋 [3/10] Creating indicator reference...")
        
        indicator_ref = pd.DataFrame({
            'column_name': indicator_cols,
            'data_type': [str(df_enriched[col].dtype) for col in indicator_cols],
            'non_null_count': [df_enriched[col].notna().sum() for col in indicator_cols],
            'null_count': [df_enriched[col].isna().sum() for col in indicator_cols],
            'mean': [df_enriched[col].mean() if df_enriched[col].dtype in ['float64', 'int64'] else np.nan 
                    for col in indicator_cols],
            'std': [df_enriched[col].std() if df_enriched[col].dtype in ['float64', 'int64'] else np.nan 
                   for col in indicator_cols]
        })
        
        ref_file = f'{self.output_dir}/03_indicator_reference_{self.timestamp}.csv'
        indicator_ref.to_csv(ref_file, index=False)
        export_files['indicator_reference'] = ref_file
        print(f"   ✓ Saved: {ref_file}")
        
        # =====================================================================
        # FILE 4: Chart Patterns Per Bar
        # =====================================================================
        print("\n📈 [4/10] Extracting chart patterns...")
        
        chart_pattern_cols = [col for col in df_enriched.columns if 
                             'pattern' in col.lower() or 'engulfing' in col.lower() or 
                             'doji' in col.lower() or 'hammer' in col.lower()]
        
        if chart_pattern_cols:
            chart_patterns_df = df_enriched[['close'] + chart_pattern_cols].copy()
            patterns_file = f'{self.output_dir}/04_chart_patterns_{pair}_{timeframe}_{self.timestamp}.csv'
            chart_patterns_df.to_csv(patterns_file, index=True)
            export_files['chart_patterns'] = patterns_file
            print(f"   ✓ Saved: {patterns_file}")
            print(f"   ✓ Found {len(chart_pattern_cols)} pattern columns")
        else:
            print(f"   ⚠ No chart pattern columns found in dataframe")
            export_files['chart_patterns'] = None
        
        # =====================================================================
        # FILE 5: Pullback States Per Bar
        # =====================================================================
        print("\n🔄 [5/10] Extracting pullback states...")
        
        pullback_cols = [col for col in df_enriched.columns if 
                        'pullback' in col.lower() or 'retracement' in col.lower()]
        
        if pullback_cols:
            pullback_df = df_enriched[['close', 'high', 'low'] + pullback_cols].copy()
            pullback_file = f'{self.output_dir}/05_pullback_states_{pair}_{timeframe}_{self.timestamp}.csv'
            pullback_df.to_csv(pullback_file, index=True)
            export_files['pullback_states'] = pullback_file
            print(f"   ✓ Saved: {pullback_file}")
            print(f"   ✓ Found {len(pullback_cols)} pullback columns")
        else:
            print(f"   ⚠ No pullback columns found in dataframe")
            export_files['pullback_states'] = None
        
        # =====================================================================
        # FILE 6: Price Action Status Per Bar
        # =====================================================================
        print("\n💹 [6/10] Extracting price action status...")
        
        price_action_cols = [col for col in df_enriched.columns if any(x in col.lower() for x in
                            ['swing', 'structure', 'trend', 'slope', 'higher_high', 'lower_low'])]
        
        if price_action_cols:
            price_action_df = df_enriched[['close'] + price_action_cols].copy()
            pa_file = f'{self.output_dir}/06_price_action_status_{pair}_{timeframe}_{self.timestamp}.csv'
            price_action_df.to_csv(pa_file, index=True)
            export_files['price_action'] = pa_file
            print(f"   ✓ Saved: {pa_file}")
            print(f"   ✓ Found {len(price_action_cols)} price action columns")
        else:
            print(f"   ⚠ No price action columns found")
            export_files['price_action'] = None
        
        # =====================================================================
        # FILE 7: HMM States Per Bar
        # =====================================================================
        print("\n🎯 [7/10] Extracting HMM regime states...")
        
        hmm_cols = [col for col in df_enriched.columns if 'hmm' in col.lower()]
        
        if hmm_cols:
            hmm_df = df_enriched[['close'] + hmm_cols].copy()
            
            # Add regime transition detection
            if 'hmm_regime' in hmm_cols:
                hmm_df['regime_changed'] = (hmm_df['hmm_regime'] != hmm_df['hmm_regime'].shift(1)).astype(int)
                hmm_df['bars_in_regime'] = hmm_df.groupby(
                    (hmm_df['hmm_regime'] != hmm_df['hmm_regime'].shift(1)).cumsum()
                ).cumcount() + 1
            
            hmm_file = f'{self.output_dir}/07_hmm_states_{pair}_{timeframe}_{self.timestamp}.csv'
            hmm_df.to_csv(hmm_file, index=True)
            export_files['hmm_states'] = hmm_file
            print(f"   ✓ Saved: {hmm_file}")
            print(f"   ✓ Found {len(hmm_cols)} HMM columns")
            
            if 'hmm_regime' in df_enriched.columns:
                regime_counts = df_enriched['hmm_regime'].value_counts()
                print(f"   ✓ Regime distribution:")
                for regime, count in regime_counts.items():
                    print(f"      - Regime {regime}: {count} bars ({count/len(df_enriched)*100:.1f}%)")
        else:
            print(f"   ⚠ No HMM columns found (regime detection may not have run)")
            export_files['hmm_states'] = None
        
        # =====================================================================
        # FILE 8: XGB Predictions Per Bar
        # =====================================================================
        print("\n🤖 [8/10] Extracting XGBoost predictions...")
        
        xgb_cols = [col for col in df_enriched.columns if 'xgb' in col.lower() or 'prediction' in col.lower()]
        
        if xgb_cols:
            xgb_df = df_enriched[['close'] + xgb_cols].copy()
            
            # Add prediction accuracy if we have actual regime labels
            if 'xgb_regime' in xgb_cols and 'hmm_regime' in df_enriched.columns:
                xgb_df['prediction_match'] = (
                    xgb_df['xgb_regime'] == df_enriched['hmm_regime']
                ).astype(int)
                accuracy = xgb_df['prediction_match'].mean() * 100
                print(f"   ✓ XGBoost accuracy: {accuracy:.2f}%")
            
            xgb_file = f'{self.output_dir}/08_xgb_predictions_{pair}_{timeframe}_{self.timestamp}.csv'
            xgb_df.to_csv(xgb_file, index=True)
            export_files['xgb_predictions'] = xgb_file
            print(f"   ✓ Saved: {xgb_file}")
            print(f"   ✓ Found {len(xgb_cols)} XGB columns")
        else:
            print(f"   ⚠ No XGBoost columns found")
            export_files['xgb_predictions'] = None
        
        # =====================================================================
        # FILE 9: Regime Instance Summary (from previous version)
        # =====================================================================
        print("\n📦 [9/10] Creating regime instance summary...")
        
        if 'regime_instance_id' in df_enriched.columns:
            instances = []
            for instance_id in df_enriched['regime_instance_id'].unique():
                if pd.isna(instance_id):
                    continue
                
                instance_data = df_enriched[df_enriched['regime_instance_id'] == instance_id]
                
                instances.append({
                    'instance_id': instance_id,
                    'regime_type': instance_data['regime_type'].iloc[0] if 'regime_type' in instance_data.columns else 'unknown',
                    'start_time': instance_data.index[0],
                    'end_time': instance_data.index[-1],
                    'duration_hours': len(instance_data),
                    'start_price': instance_data['close'].iloc[0],
                    'end_price': instance_data['close'].iloc[-1],
                    'price_change_pct': ((instance_data['close'].iloc[-1] / instance_data['close'].iloc[0]) - 1) * 100,
                    'max_price': instance_data['high'].max(),
                    'min_price': instance_data['low'].min(),
                    'volatility': instance_data['close'].pct_change().std() * 100,
                    'volume_mean': instance_data['volume'].mean(),
                    'bar_count': len(instance_data)
                })
            
            instances_df = pd.DataFrame(instances)
            instances_file = f'{self.output_dir}/09_regime_instances_{pair}_{timeframe}_{self.timestamp}.csv'
            instances_df.to_csv(instances_file, index=False)
            export_files['regime_instances'] = instances_file
            print(f"   ✓ Saved: {instances_file} ({len(instances_df)} instances)")
        else:
            print(f"   ⚠ No regime instances found (segmentation may not have run)")
            export_files['regime_instances'] = None
        
        # =====================================================================
        # FILE 10: Export Summary & Metadata
        # =====================================================================
        print("\n📄 [10/10] Creating export summary...")
        
        summary = {
            'export_timestamp': self.timestamp,
            'pair': pair,
            'timeframe': timeframe,
            'total_bars': len(df),
            'date_range_start': str(df.index[0]),
            'date_range_end': str(df.index[-1]),
            'total_indicators': len(indicator_cols),
            'chart_pattern_columns': len(chart_pattern_cols) if chart_pattern_cols else 0,
            'pullback_columns': len(pullback_cols) if pullback_cols else 0,
            'price_action_columns': len(price_action_cols) if price_action_cols else 0,
            'hmm_columns': len(hmm_cols) if hmm_cols else 0,
            'xgb_columns': len(xgb_cols) if xgb_cols else 0,
            'regime_instances': len(instances) if 'regime_instance_id' in df_enriched.columns else 0,
            'files_exported': sum(1 for v in export_files.values() if v is not None)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_file = f'{self.output_dir}/00_export_summary_{self.timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        export_files['summary'] = summary_file
        print(f"   ✓ Saved: {summary_file}")
        
        # Also save as JSON for easy programmatic access
        json_file = f'{self.output_dir}/00_export_summary_{self.timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump({
                'summary': summary,
                'files': {k: v for k, v in export_files.items() if v is not None}
            }, f, indent=2)
        print(f"   ✓ Saved: {json_file}")
        
        # =====================================================================
        # COMPLETION REPORT
        # =====================================================================
        print("\n" + "="*80)
        print("EXPORT COMPLETE")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Files exported: {summary['files_exported']}")
        print()
        print("📊 Files ready for statistical analysis:")
        for key, filepath in export_files.items():
            if filepath:
                status = "✓"
            else:
                status = "✗"
            print(f"   {status} {key}: {filepath if filepath else 'Not available'}")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Load enriched_with_indicators CSV for full feature analysis")
        print("2. Use chart_patterns CSV for pattern effectiveness studies")
        print("3. Use pullback_states CSV for pullback strategy optimization")
        print("4. Use price_action_status CSV for structure-based strategies")
        print("5. Use hmm_states CSV for regime-based backtesting")
        print("6. Use xgb_predictions CSV for ML model validation")
        print("7. Use regime_instances CSV for instance-level analysis")
        print("8. See STATISTICAL_TESTING_GUIDE.md for analysis examples")
        print("="*80)
        
        # Prepare dataframes dictionary
        dataframes = {
            'df_raw': df,
            'df_enriched': df_enriched
        }

        # Add optional dataframes if they exist
        if chart_pattern_cols:
            dataframes['chart_patterns'] = df_enriched[['close'] + chart_pattern_cols].copy()
            
        if pullback_cols:
            dataframes['pullback_states'] = df_enriched[['close', 'high', 'low'] + pullback_cols].copy()
            
        if price_action_cols:
            dataframes['price_action'] = df_enriched[['close'] + price_action_cols].copy()
            
        if hmm_cols:
            dataframes['hmm_states'] = df_enriched[['close'] + hmm_cols].copy()
            
        if xgb_cols:
            dataframes['xgb_predictions'] = df_enriched[['close'] + xgb_cols].copy()
            
        if 'regime_instance_id' in df_enriched.columns:
            dataframes['regime_instances'] = instances_df

        return {
            'summary': summary,
            'files': export_files,
            'timestamp': self.timestamp,
            'dataframes': dataframes  # ← NEW: Return actual DataFrames
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def export_btc_data_for_statistics(
    df: pd.DataFrame,
    output_dir: str = './data_exports',
    pair: str = 'BTCUSDT',
    timeframe: str = '1h',
    run_regime_detection: bool = True
):
    """
    Convenience function to export all BTC regime data.
    
    Args:
        df: Raw OHLCV DataFrame
        output_dir: Directory to save exports
        pair: Trading pair
        timeframe: Timeframe string
        run_regime_detection: Whether to run full regime detection
        
    Returns:
        Dictionary with export summary and file paths
    """
    exporter = EnhancedRegimeDataExporter(output_dir=output_dir)
    return exporter.export_all_regime_data(
        df=df,
        pair=pair,
        timeframe=timeframe,
        run_regime_detection=run_regime_detection
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    Enhanced Regime Data Exporter
    ==============================
    
    Usage:
    ------
    from export_regime_data_for_stats_enhanced import export_btc_data_for_statistics
    
    # Load your BTC data
    df = load_test_data('BTCUSDT_1h', days=365)
    
    # Export everything
    result = export_btc_data_for_statistics(
        df=df,
        output_dir='./btc_exports',
        pair='BTCUSDT',
        timeframe='1h',
        run_regime_detection=True  # Set False to skip regime detection (faster)
    )
    
    # Access exported files
    print(result['files']['enriched_df'])
    print(result['files']['hmm_states'])
    print(result['files']['xgb_predictions'])
    """)
