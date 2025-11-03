# autoTrader/main.py

import sys
import os
import argparse

# Inject the project root to handle absolute imports (if needed) or simplify package discovery
# This line ensures Python finds 'db_connector' and 'strategy' modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .db_connector import DatabaseConnector
from .ingestion import DataIngestionSystem
from .features_engineering import FeatureEngineer
from .strategy.strategy_orchestrator import run_strategy_discovery # New entry function

def parse_args():
    parser = argparse.ArgumentParser(description="AutoTrader Algorithmic Strategy Platform")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'ingest', 'feature', 'discover'],
                        help="Execution mode: ingest data, engineer features, or run discovery.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("AUTO TRADER APPLICATION STARTUP")
    print("="*80)

    # 1. Initialize Database
    db_connector = DatabaseConnector(db_path='./data/auto_trader_db.sqlite')
    '''
    # 2. Initialize Core Modules
    ingestion_system = DataIngestionSystem(db_connector=db_connector)
    feature_engineer = FeatureEngineer(db_connector=db_connector)

    if args.mode in ['all', 'ingest']:
        # A. Ingestion: Fetch historical and start WebSocket (simulated)
        print("\n--- Starting Data Ingestion Service ---")
        ingestion_system.start_historical_ingestion()
        # ingestion_system.start_websocket_listener() # Future: Must be threaded/async

    if args.mode in ['all', 'feature']:
        # B. Feature Engineering: Calculate indicators on raw data
        print("\n--- Starting Feature Engineering Service ---")
        feature_engineer.calculate_and_save_all_features()'''

    if args.mode in ['all', 'discover']:
        # C. Strategy Discovery: Run the main orchestration logic
        print("\n--- Starting Strategy Discovery Pipeline ---")
        # The Orchestrator manages Phases 1-7 using the already populated database
        run_strategy_discovery(db_connector=db_connector)

    print("\n" + "="*80)
    print("APPLICATION SHUTDOWN COMPLETE")
    print("="*80)
    db_connector.close()


if __name__ == "__main__":
    main()