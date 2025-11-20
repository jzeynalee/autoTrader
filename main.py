# autoTrader/main.py

import sys
import os
import time
import argparse
import multiprocessing
import logging
import schedule

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from .db_connector import DatabaseConnector
from .ingestion import DataIngestionSystem
from .features_engineering import FeatureEngineer
from .strategy.strategy_orchestrator import run_strategy_discovery # New entry function
from .strategy.analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .config import get_db_path # Import the config helper

# --- PILLAR 1: STRATEGY FINDER (The Researcher) ---
def run_discovery_pillar(interval_hours=4):
    """
    Pillar 1: Runs periodically on heavy historical data.
    Detects regimes and updates the Strategy Playbook.
    """
    print(f"🔍 [Discovery] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    
    def job():
        print(f"\n🔍 [Discovery] Starting scheduled analysis cycle...")
        try:
            # Create a fresh DB connection for this process
            db = DatabaseConnector(db_path)
            # Run the heavy discovery pipeline
            run_strategy_discovery(db_connector=db)
            db.close()
            print(f"✅ [Discovery] Cycle complete. Sleeping...")
        except Exception as e:
            print(f"❌ [Discovery] Error: {e}")

    # Run immediately on startup
    job()
    
    # Schedule future runs
    schedule.every(interval_hours).hours.do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# --- PILLAR 2: TRADER (The Sniper) ---
def run_execution_pillar():
    """
    Pillar 2: Runs in a tight loop. Ingests live data, checks Playbook, enters trades.
    """
    print(f"🔫 [Execution] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    db = DatabaseConnector(db_path)
    
    # In a real scenario, you would initialize your WebSocket client here
    # from .ingestion import DataIngestionSystem
    # ingestion = DataIngestionSystem(db)
    # ingestion.start_websocket_listener() (Async)

    print("🔫 [Execution] Entering High-Frequency Loop...")
    while True:
        try:
            # 1. Get latest live price (Simulated here)
            # latest_bar = ingestion.get_latest_bar()
            
            # 2. Fetch active strategies from DB Playbook
            # strategies = db.fetch("SELECT * FROM strategy_playbook WHERE status='ACTIVE'")
            
            # 3. Check for entry signals
            # if signal_found: execute_order()
            
            # Sleep to match candle timeframe or tick rate
            time.sleep(1) 
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ [Execution] Error: {e}")
            time.sleep(5)

# --- PILLAR 3: TRADE MANAGER (The Risk Officer) ---
def run_risk_pillar():
    """
    Pillar 3: Monitors open positions, updates trailing stops, handles exits.
    """
    print(f"🛡️ [Risk Manager] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    db = DatabaseConnector(db_path)
    
    # from .trailing import TrailingStopEngine
    
    print("🛡️ [Risk Manager] Monitoring Positions...")
    while True:
        try:
            # 1. Fetch Open Positions
            # positions = db.fetch("SELECT * FROM positions WHERE status='OPEN'")
            
            # 2. For each position:
            #    - Check if Stop Loss hit
            #    - Check if Take Profit hit
            #    - Update Trailing Stop (using trailing.py logic)
            
            # 3. Close positions if needed
            
            time.sleep(1) # Fast loop for risk checks
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ [Risk] Error: {e}")
            time.sleep(5)

# --- MAIN ORCHESTRATOR ---
def start_system(args):
    processes = []

    # Define the pillars based on arguments
    if args.mode in ['all', 'discover']:
        p1 = multiprocessing.Process(target=run_discovery_pillar, name="DiscoveryEngine")
        processes.append(p1)

    if args.mode in ['all', 'trade']:
        p2 = multiprocessing.Process(target=run_execution_pillar, name="ExecutionEngine")
        processes.append(p2)
        
    if args.mode in ['all', 'trade']: # Risk usually runs with Trade
        p3 = multiprocessing.Process(target=run_risk_pillar, name="RiskEngine")
        processes.append(p3)

    # Start all processes
    print(f"\n🚀 STARTING AUTO TRADER [Mode: {args.mode.upper()}]")
    print("="*60)
    
    for p in processes:
        p.start()
        
    # Monitor loop
    try:
        while True:
            time.sleep(1)
            # Check if processes are alive, restart if crashed (advanced logic can go here)
            for p in processes:
                if not p.is_alive():
                    print(f"⚠️ Process {p.name} died! (Real system should restart this)")
                    break
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all pillars...")
        for p in processes:
            p.terminate()
            p.join()
        print("✅ System Shutdown Complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="AutoTrader Multi-Process System")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'discover', 'trade', 'ingest'],
                        help="Run specific pillars or all.")
    return parser.parse_args()

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    args = parse_args()
    start_system(args)


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

    # 1. Initialize Database from Config
    db_path = get_db_path()
    print(f"--- Connecting to database: {db_path} ---")
    db_connector = DatabaseConnector(db_path=db_path)
    
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
        feature_engineer.calculate_and_save_all_features()

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