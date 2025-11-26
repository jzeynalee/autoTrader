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
from .config import get_db_path

# --- PROCESS WRAPPERS FOR PARALLEL EXECUTION ---

def run_scheduler_process(interval_hours=4):
    """
    Runs the periodic re-discovery and maintenance tasks.
    """
    print(f"🕰️ [Scheduler] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    
    def scheduled_job():
        print(f"\n🔄 [Scheduler] Starting periodic analysis...")
        try:
            db = DatabaseConnector(db_path)
            
            # 1. Update Features (Incremental)
            print("⚙️ [Scheduler] Updating Features...")
            fe = FeatureEngineer(db)
            fe.calculate_and_save_all_features()
            
            # 2. Re-run Discovery
            print("🧠 [Scheduler] Refreshing Strategies...")
            run_strategy_discovery(db_connector=db)
            
            db.close()
            print(f"✅ [Scheduler] Cycle complete.")
        except Exception as e:
            print(f"❌ [Scheduler] Error: {e}")

    # Schedule future runs
    schedule.every(interval_hours).hours.do(scheduled_job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def run_execution_pillar():
    """
    Pillar 2: THE SNIPER
    Runs in a tight loop. Ingests live data, checks Playbook, enters trades.
    Only starts after strategies are generated.
    """
    print(f"🔫 [Execution] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    
    # Initialize
    try:
        db = DatabaseConnector(db_path)
        ingestion = DataIngestionSystem(db_connector=db)
        
        # Start WebSocket (Blocking/Async in real app, simulated loop here)
        # ingestion.start_websocket_listener() 
        
        print("🔫 [Execution] Ready. Waiting for signals...")
        while True:
            # Placeholder for Real-Time Logic:
            # 1. msg = websocket_queue.get()
            # 2. db.upsert_candle(msg)
            # 3. signals = signal_finder.check_strategies(strategies)
            # 4. if signal: order_manager.execute(signal)
            time.sleep(1) 
            
    except Exception as e:
        print(f"❌ [Execution] Crash: {e}")

def run_risk_pillar():
    """
    Pillar 3: THE RISK OFFICER
    Monitors open positions.
    """
    print(f"🛡️ [Risk Manager] Process started (PID: {os.getpid()})")
    db_path = get_db_path()
    db = DatabaseConnector(db_path)
    
    print("🛡️ [Risk Manager] Monitoring Positions...")
    while True:
        # Placeholder:
        # 1. positions = db.get_open_positions()
        # 2. if not positions: sleep(5)
        # 3. else: check_trailing_stops(positions)
        time.sleep(5)

# --- SEQUENTIAL INITIALIZATION PIPELINE ---

def run_initialization_sequence(db):
    """
    Performs the mandatory 'Cold Start' sequence.
    System cannot trade until this completes successfully.
    """
    print("\n" + "="*60)
    print("🏗️  SYSTEM INITIALIZATION PIPELINE (COLD START)")
    print("="*60)

    # STEP 1: INGESTION (Get the Raw Data)
    print("\n[Step 1/3] 📥 Historical Data Ingestion...")
    ingestion = DataIngestionSystem(db_connector=db)
    ingestion.start_historical_ingestion()
    print("✅ Ingestion Complete.")

    # STEP 2: ENGINEERING (Calculate the Indicators)
    print("\n[Step 2/3] ⚙️  Feature Engineering...")
    fe = FeatureEngineer(db_connector=db)
    fe.calculate_and_save_all_features()
    print("✅ Feature Engineering Complete.")

    # STEP 3: DISCOVERY (Build the Brain)
    print("\n[Step 3/3] 🧠 Strategy Discovery & Backtesting...")
    # This generates strategy_pool.json
    run_strategy_discovery(db_connector=db) 
    print("✅ Strategy Discovery Complete.")
    
    print("\n✨ SYSTEM READY FOR LIVE OPERATIONS ✨")
    print("="*60 + "\n")

# --- MAIN ENTRY POINT ---

def start_system(args):
    print(f"\n🚀 STARTING AUTO TRADER [Mode: {args.mode.upper()}]")
    
    db_path = get_db_path()
    print(f"--- Database: {db_path} ---")
    
    # Create main process DB connection
    db = DatabaseConnector(db_path)

    # --- PHASE A: SEQUENTIAL INIT ---
    if args.mode in ['all', 'discover']:
        try:
            run_initialization_sequence(db)
        except Exception as e:
            print(f"❌ FATAL: Initialization failed. Aborting startup.")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # --- PHASE B: PARALLEL EXECUTION ---
    if args.mode in ['all', 'trade']:
        print("🔥 Launching Parallel Pillars...")
        processes = []

        # 1. Execution Engine (Trades)
        p_exec = multiprocessing.Process(target=run_execution_pillar, name="ExecutionEngine")
        processes.append(p_exec)

        # 2. Risk Manager (Protects)
        p_risk = multiprocessing.Process(target=run_risk_pillar, name="RiskEngine")
        processes.append(p_risk)
        
        # 3. Scheduler (Updates Strategies Periodically)
        # Only needed if we want to re-discover while trading
        if args.mode == 'all':
            p_sched = multiprocessing.Process(target=run_scheduler_process, name="Scheduler")
            processes.append(p_sched)

        # Start All
        for p in processes:
            p.start()

        # Monitor Loop
        try:
            while True:
                time.sleep(1)
                for p in processes:
                    if not p.is_alive():
                        print(f"⚠️ Process {p.name} died! Restarting is recommended.")
                        # Logic to restart could go here
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            for p in processes:
                p.terminate()
                p.join()

def parse_args():
    parser = argparse.ArgumentParser(description="AutoTrader System")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'discover', 'trade', 'ingest'],
                        help="Run mode")
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    start_system(args)