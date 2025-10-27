#!/usr/bin/env python3
"""
Enhanced Main Module for MTF Trading System
Manages parallel processing for Strategy Discovery and Signal Builder
"""

import threading
import time
import queue
import logging
from datetime import datetime
from typing import Dict, Any, List
import signal
import sys

# Import your existing modules
from mtf_discovery import StrategyDiscoverySystem
from calculator import FeatureCalculator
from discovery_mapping import SignalStateMapper
from sl_tp_planner import RiskManagementPlanner

# Import new data ingestion module
from data_ingestion import DataIngestionEngine

class ParallelTradingSystem:
    """
    Main controller that manages parallel processing for:
    A. Strategy Discovery
    B. Signal Builder
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        
        # Shared data structures
        self.data_queue = queue.Queue(maxsize=1000)
        self.signal_queue = queue.Queue(maxsize=500)
        self.control_queue = queue.Queue()
        
        # Initialize components
        self.data_engine = DataIngestionEngine(config)
        self.feature_calculator = FeatureCalculator(config)
        self.signal_mapper = SignalStateMapper(config)
        self.mtf_discovery = MTFDiscoveryEngine(config)
        self.risk_planner = RiskManagementPlanner(config)
        
        # Thread management
        self.threads = []
        self.thread_status = {}
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'trading_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("ParallelTradingSystem")
        
    def start_strategy_discovery(self):
        """Thread A: Strategy Discovery Pipeline"""
        self.logger.info("Starting Strategy Discovery Thread")
        
        while self.running:
            try:
                # Get fresh data from queue
                market_data = self.data_queue.get(timeout=1.0)
                
                # Feature engineering
                features = self.feature_calculator.calculate_all_indicators(market_data)
                
                # Signal state mapping
                mapped_states = self.signal_mapper.map_to_directional_states(features)
                
                # MTF strategy discovery
                strategies = self.mtf_discovery.discover_strategies(mapped_states)
                
                # Update discovery results
                self.mtf_discovery.update_strategy_library(strategies)
                
                self.logger.debug("Strategy discovery cycle completed")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Strategy discovery error: {e}")
                time.sleep(5)
                
    def start_signal_builder(self):
        """Thread B: Signal Builder Pipeline"""
        self.logger.info("Starting Signal Builder Thread")
        
        while self.running:
            try:
                # Get latest market data
                market_data = self.data_queue.get(timeout=1.0)
                
                # Calculate real-time features
                features = self.feature_calculator.calculate_all_indicators(market_data)
                
                # Generate signals using discovered strategies
                signals = self.mtf_discovery.generate_signals(features)
                
                # Risk management planning
                risk_plans = self.risk_planner.calculate_risk_parameters(signals, features)
                
                # Put signals in output queue
                if signals:
                    self.signal_queue.put({
                        'timestamp': datetime.now(),
                        'signals': signals,
                        'risk_plans': risk_plans
                    })
                
                self.logger.debug("Signal builder cycle completed")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Signal builder error: {e}")
                time.sleep(5)
                
    def start_data_ingestion(self):
        """Data ingestion thread - feeds both pipelines"""
        self.logger.info("Starting Data Ingestion Thread")
        
        # Start data ingestion engine
        self.data_engine.start()
        
        while self.running:
            try:
                # Get latest data from ingestion engine
                data_batch = self.data_engine.get_latest_data()
                
                if data_batch:
                    # Put data in queue for both threads
                    self.data_queue.put(data_batch)
                    
                time.sleep(0.1)  # Small delay to prevent CPU overload
                
            except Exception as e:
                self.logger.error(f"Data ingestion error: {e}")
                time.sleep(5)
                
    def start_monitoring(self):
        """Monitoring and health check thread"""
        while self.running:
            try:
                # Check thread status
                for thread_name, status in self.thread_status.items():
                    if not status['alive']:
                        self.logger.warning(f"Thread {thread_name} is not alive")
                        
                # Log system status
                self.logger.info(
                    f"System Status - Data Queue: {self.data_queue.qsize()}, "
                    f"Signal Queue: {self.signal_queue.qsize()}"
                )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
                
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received shutdown signal")
        self.stop()
        
    def start(self):
        """Start the parallel trading system"""
        self.logger.info("Starting Parallel Trading System")
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Define threads
        threads_config = [
            ('data_ingestion', self.start_data_ingestion),
            ('strategy_discovery', self.start_strategy_discovery),
            ('signal_builder', self.start_signal_builder),
            ('monitoring', self.start_monitoring)
        ]
        
        # Start all threads
        for name, target in threads_config:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)
            self.thread_status[name] = {'alive': True, 'thread': thread}
            
        self.logger.info("All system threads started successfully")
        
        # Main loop
        try:
            while self.running:
                # Process control messages
                try:
                    control_msg = self.control_queue.get(timeout=1.0)
                    self._handle_control_message(control_msg)
                except queue.Empty:
                    pass
                    
                # Process output signals
                try:
                    signal_data = self.signal_queue.get_nowait()
                    self._process_final_signals(signal_data)
                except queue.Empty:
                    pass
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()
            
    def _handle_control_message(self, message: Dict[str, Any]):
        """Handle system control messages"""
        msg_type = message.get('type')
        
        if msg_type == 'shutdown':
            self.running = False
        elif msg_type == 'reload_config':
            self.logger.info("Reloading configuration")
            # Implement configuration reload logic
        elif msg_type == 'emergency_stop':
            self.logger.warning("EMERGENCY STOP triggered")
            self.running = False
            
    def _process_final_signals(self, signal_data: Dict[str, Any]):
        """Process final trading signals"""
        try:
            # Here you would integrate with your execution system
            signals = signal_data['signals']
            risk_plans = signal_data['risk_plans']
            
            self.logger.info(f"Generated {len(signals)} signals with risk plans")
            
            # Log signals for now - replace with actual execution
            for signal in signals:
                self.logger.info(
                    f"Signal: {signal.get('pair')} | "
                    f"Direction: {signal.get('direction')} | "
                    f"Confidence: {signal.get('confidence', 0):.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Error processing final signals: {e}")
            
    def stop(self):
        """Stop the trading system gracefully"""
        self.logger.info("Stopping Parallel Trading System")
        self.running = False
        
        # Stop data engine
        if hasattr(self, 'data_engine'):
            self.data_engine.stop()
            
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
            
        self.logger.info("Trading system stopped successfully")

def main():
    """Main entry point"""
    # Configuration - you can load this from a file
    config = {
        'pairs': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 
                 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT', 'EOSUSDT',
                 'TRXUSDT', 'ETCUSDT', 'XTZUSDT', 'ATOMUSDT', 'NEOUSDT',
                 'IOTAUSDT', 'VETUSDT', 'THETAUSDT', 'ALGOUSDT', 'FILUSDT'],
        'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
        'database': {
            'url': 'sqlite:///trading_data.db',
            'ohlcv_table': 'ohlcv_data',
            'features_table': 'technical_indicators'
        },
        'websocket': {
            'reconnect_delay': 5,
            'max_reconnect_attempts': 10
        },
        'processing': {
            'max_queue_size': 1000,
            'processing_batch_size': 100
        }
    }
    
    # Create and start the system
    trading_system = ParallelTradingSystem(config)
    
    try:
        trading_system.start()
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()