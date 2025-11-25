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
from .strategy.strategy_orchestrator import run_strategy_discovery
from .strategy.analysis_advanced_regime import AdvancedRegimeDetectionSystem
from .config import get_db_path

# --- PILLAR 1: STRATEGY FINDER (The Researcher) ---
def run_discovery_pillar(interval_hours=4):
    """
    Pillar 1: Runs periodically on heavy historical data.
    1. Updates Features (Calculates indicators)
    2. Detects regimes and updates the Strategy Playbook.
    """
    print(f"🔍 [Discovery] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    
    def job():
        print(f"\n🔍 [Discovery] Starting scheduled analysis cycle...")
        try:
            # Create a fresh DB connection for this process
            db = DatabaseConnector(db_path)
            
            # --- STEP 1: Feature Engineering (The Fix) ---
            # We must calculate indicators before we can discover strategies based on them.
            print("⚙️ [Discovery] Updating Features (Engineering)...")
            fe = FeatureEngineer(db)
            fe.calculate_and_save_all_features()
            
            # --- STEP 2: Strategy Discovery ---
            print("🧠 [Discovery] Running Strategy Orchestrator...")
            run_strategy_discovery(db_connector=db)
            
            db.close()
            print(f"✅ [Discovery] Cycle complete. Sleeping...")
            
        except Exception as e:
            print(f"❌ [Discovery] Error: {e}")
            import traceback
            traceback.print_exc()

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
    
    # Initialize components
    ingestion = DataIngestionSystem(db_connector=db)
    
    # In a real scenario, you would start the socket here
    # ingestion.start_websocket_listener() 

    print("🔫 [Execution] Entering High-Frequency Loop...")
    while True:
        try:
            # Placeholder for live trading loop
            # 1. Get latest live price
            # 2. Update incremental features
            # 3. Check Playbook
            
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
    
    print("🛡️ [Risk Manager] Monitoring Positions...")
    while True:
        try:
            # Placeholder for risk management
            # 1. Fetch positions
            # 2. Check stops/targets
            
            time.sleep(1) 
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ [Risk] Error: {e}")
            time.sleep(5)

# --- MAIN ORCHESTRATOR ---
def start_system(args):
    processes = []
    
    print(f"\n🚀 STARTING AUTO TRADER [Mode: {args.mode.upper()}]")
    print("="*60)

    # 1. Initial Ingestion Check (Optional but recommended before branching)
    if args.mode in ['all', 'discover']:
        # Run a quick check or initial historical load in the main process
        # so workers have something to work with immediately.
        try:
            db_path = get_db_path()
            print(f"--- Pre-flight: Connecting to {db_path} ---")
            # You could run initial ingestion here if needed
        except Exception as e:
            print(f"⚠️ Pre-flight check warning: {e}")

    # 2. Define the pillars based on arguments
    if args.mode in ['all', 'discover']:
        p1 = multiprocessing.Process(target=run_discovery_pillar, name="DiscoveryEngine")
        processes.append(p1)

    if args.mode in ['all', 'trade']:
        p2 = multiprocessing.Process(target=run_execution_pillar, name="ExecutionEngine")
        processes.append(p2)
        
    if args.mode in ['all', 'trade']: # Risk usually runs with Trade
        p3 = multiprocessing.Process(target=run_risk_pillar, name="RiskEngine")
        processes.append(p3)

    # 3. Start all processes
    for p in processes:
        p.start()
        
    # 4. Monitor loop (Main Process)
    try:
        while True:
            time.sleep(1)
            # Simple health check
            for p in processes:
                if not p.is_alive():
                    print(f"⚠️ Process {p.name} died! (PID: {p.pid})")
                    # Logic to restart process could go here
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