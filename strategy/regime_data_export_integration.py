"""
regime_data_export_integration.py
Integration module to wire enhanced data export into the strategy discovery system.
"""

import pandas as pd
from typing import Optional, Dict
from .export_regime_data_for_stats_enhanced import EnhancedRegimeDataExporter


class RegimeDataExportIntegration:
    """
    Integrates enhanced data export with the existing regime-based discovery system.
    Can be called from strategy_orchestrator or run independently.
    """
    
    def __init__(self, db_connector=None):
        """
        Args:
            db_connector: Optional database connector for loading data
        """
        self.db_connector = db_connector
        self.exporter = None
        
    def export_from_database(
        self,
        pair: str = 'BTCUSDT',
        timeframe: str = '1h',
        output_dir: str = './data_exports',
        days_lookback: Optional[int] = 365,
        run_regime_detection: bool = True
    ) -> Dict:
        """
        Load data from database and export for statistical analysis.
        
        Args:
            pair: Trading pair to export
            timeframe: Timeframe to export
            output_dir: Directory for exports
            days_lookback: Number of days to load (None = all available)
            run_regime_detection: Whether to run full regime detection
            
        Returns:
            Export result dictionary
        """
        if not self.db_connector:
            raise ValueError("Database connector not provided. Initialize with db_connector.")
        
        print(f"Loading {pair} {timeframe} data from database...")
        
        # Load data from database
        # Adjust this based on your actual database schema/methods
        df = self.db_connector.load_ohlcv_data(
            pair=pair,
            timeframe=timeframe,
            days_lookback=days_lookback
        )
        
        if df is None or len(df) == 0:
            raise ValueError(f"No data loaded for {pair} {timeframe}")
        
        print(f"✓ Loaded {len(df)} bars")
        
        # Export using enhanced exporter
        self.exporter = EnhancedRegimeDataExporter(output_dir=output_dir)
        result = self.exporter.export_all_regime_data(
            df=df,
            pair=pair,
            timeframe=timeframe,
            run_regime_detection=run_regime_detection
        )
        
        # Store export metadata in database (optional)
        if self.db_connector and hasattr(self.db_connector, 'store_export_metadata'):
            self._store_export_metadata(result)
        
        return result
    
    def export_from_dataframe(
        self,
        df: pd.DataFrame,
        pair: str = 'BTCUSDT',
        timeframe: str = '1h',
        output_dir: str = './data_exports',
        run_regime_detection: bool = True
    ) -> Dict:
        """
        Export data from existing DataFrame.
        
        Args:
            df: OHLCV DataFrame
            pair: Trading pair name
            timeframe: Timeframe string
            output_dir: Directory for exports
            run_regime_detection: Whether to run full regime detection
            
        Returns:
            Export result dictionary
        """
        self.exporter = EnhancedRegimeDataExporter(output_dir=output_dir)
        return self.exporter.export_all_regime_data(
            df=df,
            pair=pair,
            timeframe=timeframe,
            run_regime_detection=run_regime_detection
        )
    
    def export_multiple_timeframes(
        self,
        pair: str = 'BTCUSDT',
        timeframes: list = ['1h', '4h', '1d'],
        output_dir: str = './data_exports',
        run_regime_detection: bool = True
    ) -> Dict[str, Dict]:
        """
        Export data for multiple timeframes.
        
        Args:
            pair: Trading pair
            timeframes: List of timeframes to export
            output_dir: Base output directory
            run_regime_detection: Whether to run regime detection
            
        Returns:
            Dictionary mapping timeframe to export result
        """
        results = {}
        
        for tf in timeframes:
            print(f"\n{'='*80}")
            print(f"EXPORTING {pair} {tf}")
            print(f"{'='*80}")
            
            try:
                # Create timeframe-specific subdirectory
                tf_output_dir = f"{output_dir}/{pair}_{tf}"
                
                if self.db_connector:
                    result = self.export_from_database(
                        pair=pair,
                        timeframe=tf,
                        output_dir=tf_output_dir,
                        run_regime_detection=run_regime_detection
                    )
                else:
                    print(f"⚠ No database connector - skip {tf}")
                    continue
                
                results[tf] = result
                print(f"✓ {tf} export complete")
                
            except Exception as e:
                print(f"✗ Failed to export {tf}: {e}")
                results[tf] = {'error': str(e)}
        
        return results
    
    def _store_export_metadata(self, result: Dict):
        """Store export metadata in database for tracking."""
        try:
            metadata = {
                'timestamp': result['timestamp'],
                'pair': result['summary']['pair'],
                'timeframe': result['summary']['timeframe'],
                'total_bars': result['summary']['total_bars'],
                'files_exported': result['summary']['files_exported'],
                'export_path': result['files'].get('summary', '')
            }
            
            self.db_connector.store_export_metadata(metadata)
            print("✓ Export metadata stored in database")
            
        except Exception as e:
            print(f"⚠ Could not store export metadata: {e}")


# ============================================================================
# INTEGRATION WITH STRATEGY ORCHESTRATOR
# ============================================================================

def add_export_to_strategy_pipeline(system, export_config: Optional[Dict] = None):
    """
    Hook to add data export functionality to the strategy orchestrator.
    
    Call this from strategy_orchestrator.py after Phase 2 (Feature Engineering).
    
    Args:
        system: StrategyDiscoverySystem instance
        export_config: Configuration dict with:
            - enabled: bool (default True)
            - output_dir: str
            - export_after_phase: int (1=raw, 2=enriched, 3=after regimes)
            - pairs: list of pairs to export
            - timeframes: list of timeframes to export
            
    Example usage in strategy_orchestrator.py:
        # After Phase 2: Feature Engineering
        if config.get('export_enabled', False):
            from regime_data_export_integration import add_export_to_strategy_pipeline
            add_export_to_strategy_pipeline(system, export_config)
    """
    if not export_config:
        export_config = {'enabled': True, 'output_dir': './data_exports'}
    
    if not export_config.get('enabled', True):
        print("Data export disabled in config")
        return
    
    print("\n" + "="*80)
    print("DATA EXPORT FOR STATISTICAL ANALYSIS")
    print("="*80)
    
    integrator = RegimeDataExportIntegration(db_connector=system.db_connector)
    
    pairs = export_config.get('pairs', ['BTCUSDT'])
    timeframes = export_config.get('timeframes', ['1h'])
    output_dir = export_config.get('output_dir', './data_exports')
    
    results = {}
    
    for pair in pairs:
        for timeframe in timeframes:
            key = f"{pair}_{timeframe}"
            
            # Get data from system if available
            if hasattr(system, 'dataframes') and key in system.dataframes:
                df = system.dataframes[key]
                
                result = integrator.export_from_dataframe(
                    df=df,
                    pair=pair,
                    timeframe=timeframe,
                    output_dir=output_dir,
                    run_regime_detection=False  # Already done in system
                )
                
                results[key] = result
                print(f"✓ Exported {key}")
            else:
                print(f"⚠ Data not found for {key}")
    
    print(f"\n✓ Exported {len(results)} dataset(s) for statistical analysis")
    
    return results


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    """
    Standalone execution for data export.
    Can be run independently of the main strategy system.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Export regime data for statistical analysis')
    parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--output-dir', default='./data_exports', help='Output directory')
    parser.add_argument('--no-regime-detection', action='store_true', 
                       help='Skip regime detection (faster)')
    parser.add_argument('--from-csv', type=str, help='Load from CSV file instead of database')
    
    args = parser.parse_args()
    
    integrator = RegimeDataExportIntegration()
    
    if args.from_csv:
        print(f"Loading data from {args.from_csv}...")
        df = pd.read_csv(args.from_csv, index_col=0, parse_dates=True)
        
        result = integrator.export_from_dataframe(
            df=df,
            pair=args.pair,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
            run_regime_detection=not args.no_regime_detection
        )
    else:
        print("ERROR: Database connector not implemented in standalone mode")
        print("Use --from-csv to load data from a CSV file")
        return
    
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"Summary: {result['files']['summary']}")


if __name__ == "__main__":
    main()
