import requests
import pandas as pd
import numpy as np
import time
import asyncio
import websockets
import json
import os
import concurrent.futures

from typing import Optional, List, Dict
from datetime import datetime, timedelta

# Assuming db_connector is available via relative import from the parent package
# and config.env constants are handled elsewhere or loaded here (using placeholders)
from .db_connector import DatabaseConnector # Assumes DBConnector is available

# --- Configuration Placeholders (In a real app, these come from config.env) ---
API_BASE_URL = "https://api.lbkex.com/v2/kline.do"
WS_BASE_URL = "wss://ws.lbkex.com/ws/V2/"
WS_MAX_RETRIES = 5
WS_RECONNECT_DELAY_S = 10
# Assuming 20 pairs total, mapping to LBANK format
PAIRS_TO_FETCH = ['btc_usdt']#, 'eth_usdt', 'sol_usdt', 'trx_usdt', 'doge_usdt']
TIME_FRAMES = {'1m': 'minute1', '5m': 'minute5', '15m': 'minute15', '1h': 'hour1', '4h': 'hour4'}
MAX_LIMIT = 2000
DAYS_HISTORICAL = 365 # Default for initial setup
# ----------------------------------------------------------------------------


class DataIngestionSystem:
    """
    Manages historical REST API ingestion, WebSocket streaming, and fallback logic 
    to ensure the features_data table has fresh, accurate OHLCV data.
    """
    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
        self.pairs = PAIRS_TO_FETCH
        self.timeframes = TIME_FRAMES
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10) # For REST calls

    def get_last_timestamp(self, pair_tf: str) -> Optional[int]:
        """
        Retrieves the last known timestamp (in seconds) for a pair_tf from the database.
        
        Note: This relies on the db_connector having a load_last_timestamp(pair_tf) method.
        """
        try:
            # We assume a load_last_timestamp method exists in DatabaseConnector
            last_ts = self.db.load_last_timestamp(pair_tf)
            return last_ts # Should return timestamp in seconds (int)
        except AttributeError:
            print("CRITICAL: DatabaseConnector is missing load_last_timestamp method.")
            return None
        except Exception as e:
            print(f"Error checking last timestamp for {pair_tf}: {e}")
            return None

    # =========================================================================
    # A. HISTORICAL DATA INGESTION (REST API)
    # =========================================================================

    def _fetch_klines_rest(self, symbol, timeframe, time_type, start_time_s: int):
        """
        Fetches K-line data via REST API starting from a specific timestamp.
        """
        
        # Ensure we don't request data from the future
        end_time = int(time.time())
        if start_time_s >= end_time:
             print(f"    Skipping {symbol} {timeframe}: Database is up-to-date.")
             return []

        params = {
            'symbol': symbol,
            'type': time_type,
            'size': MAX_LIMIT,
            'time': start_time_s # Use provided start time
        }
        
        try:
            print(f"  Requesting {symbol}_{timeframe} from: {datetime.fromtimestamp(start_time_s)}")
            
            response = requests.get(API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('result') == 'true' and 'data' in data and data['data']:
                return data['data']
            else:
                print(f"    ✗ API Error for {symbol} {timeframe}: {data.get('msg', 'Unknown')}")
                return []
                
        except Exception as e:
            print(f"    ✗ REST Exception for {symbol} {timeframe}: {str(e)}")
            return []

    def _process_and_save_data(self, raw_data, symbol, timeframe):
        """Converts raw API data to DataFrame and saves raw OHLCV to the database."""
        if not raw_data:
            return 0

        df = pd.DataFrame(raw_data)
        
        if df.empty:
            return 0
        
        # LBANK API returns timestamp in seconds (column 0)
        df = df.rename(columns={
            0: 'timestamp_s', 
            1: 'open',
            2: 'high',
            3: 'low',
            4: 'close',
            5: 'volume'
        })
        
        # Clean and prepare for DB
        df = df.drop_duplicates(subset=['timestamp_s'])
        df = df[df['timestamp_s'] > 0]

        pair_tf_key = f"{symbol}_{timeframe}"
        df['pair_tf'] = pair_tf_key
        
        # Prepare for bulk UPSERT (We only save OHLCV + required columns)
        df_to_save = df[['timestamp_s', 'pair_tf', 'open', 'high', 'low', 'close', 'volume']]
        
        last_ts_in_db = self.get_last_timestamp(pair_tf_key) 

        if last_ts_in_db is not None:
            # We filter to keep only data that is STRICTLY newer than the last known bar.
            df_to_save = df_to_save[df_to_save['timestamp_s'] > last_ts_in_db]

        if df_to_save.empty:
            # The API successfully returned data, but it was all old/redundant data.
            # This is a successful incremental check, but zero new bars were saved.
            return 0
        
        # We use timestamp_s as the PRIMARY KEY (PK) integer value
        self.db.upsert_raw_ohlcv(df_to_save, f"{symbol}_{timeframe}")

        return len(df_to_save)

    def start_historical_ingestion(self):
        """
        Main entry point for historical data ingestion. Determines start time 
        based on last database timestamp for incremental loading.
        """
        print("\n--- A1. Starting Incremental REST Ingestion ---")
        
        tasks = []
        total_saved_rows = 0
        current_time_s = int(time.time())

        for pair in self.pairs:
            for tf, time_type in self.timeframes.items():
                pair_tf = f"{pair}_{tf}"
                
                # 1. Check Database for last timestamp
                last_ts = self.db.load_last_timestamp(pair_tf)
                
                if last_ts is not None:
                    # Increment last timestamp by 1 second to fetch the next bar onwards
                    start_time_s = last_ts + 1
                    time_source = f"DB (Last TS: {datetime.fromtimestamp(last_ts)})"
                else:
                    # Database empty: fetch a full year
                    start_time_s = current_time_s - (DAYS_HISTORICAL * 24 * 60 * 60)
                    time_source = f"FULL YEAR (Start: {datetime.fromtimestamp(start_time_s)})"
                
                print(f"  {pair_tf}: Source: {time_source}")
                
                # 2. Add task for concurrent REST API call
                tasks.append((pair, tf, time_type, start_time_s))

        # 3. Execute tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_task = {
                executor.submit(
                    self._fetch_klines_rest, pair, tf, time_type, start_time_s
                ): (pair, tf)
                for pair, tf, time_type, start_time_s in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                pair, tf = future_to_task[future]
                try:
                    raw_data = future.result()
                    
                    if raw_data:
                        saved_count = self._process_and_save_data(raw_data, pair, tf)
                        total_saved_rows += saved_count
                        if saved_count > 0:
                            print(f"  ✅ Saved {saved_count} bars for {pair}_{tf}")
                        # If saved_count is 0, print logic is handled inside fetcher/process methods
                    
                except Exception as exc:
                    print(f"  ❌ Task failed for {pair}_{tf}: {exc}")

        print(f"\n--- A1. Historical Ingestion Complete. Total Bars Saved: {total_saved_rows} ---")

    # =========================================================================
    # B. REAL-TIME AND FALLBACK (WEBSOCKET/REST)
    # =========================================================================
    
    async def _handle_websocket_data(self, message):
        """Processes real-time data and saves to DB."""
        try:
            data = json.loads(message)
            if 'channel' in data and data['channel'].startswith('addChannel.'):
                # This is a K-line subscription update
                kline_data = data['data']
                
                # Extract relevant fields (timestamp, OHLCV)
                # NOTE: WebSocket data format may differ from REST; careful parsing needed.
                # Assuming kline_data is a list of kline dictionaries
                
                # Placeholder: In a real implementation, aggregate or save immediately.
                if kline_data:
                    # Example: Get last candle
                    last_kline = kline_data[-1]
                    # last_kline = { 'timestamp': 1678886400, 'open': '20000', ... }
                    
                    # We need a dedicated function in db_connector to handle single candle upsert
                    # self.db.upsert_single_candle(last_kline, pair_tf)
                    pass 
                
        except json.JSONDecodeError:
            pass # Ignore non-JSON messages
        except Exception as e:
            print(f"Websocket data handler error: {e}")

    async def _websocket_subscribe(self, websocket):
        """Sends subscription messages for all pairs and 1m timeframe."""
        for pair in self.pairs:
            # LBANK WebSocket uses different symbol convention (e.g., btcusdt)
            ws_symbol = pair.replace('_', '')
            # Subscribe to 1-minute K-line
            subscribe_msg = {
                "action": "subscribe",
                "subscribe": f"addChannel.kline_{ws_symbol}_1min"
            }
            await websocket.send(json.dumps(subscribe_msg))
            await asyncio.sleep(0.1) # Respect message limit

    async def _websocket_listener(self):
        """Manages the WebSocket connection and reconnection logic."""
        uri = WS_BASE_URL
        retries = 0

        while retries < WS_MAX_RETRIES:
            try:
                print(f"\n--- B2. Attempting WebSocket connection (Attempt {retries + 1}) ---")
                async with websockets.connect(uri) as websocket:
                    await self._websocket_subscribe(websocket)
                    print("✅ WebSocket Subscribed. Listening for real-time data...")

                    while True:
                        try:
                            message = await websocket.recv()
                            await self._handle_websocket_data(message)
                        except websockets.ConnectionClosed:
                            print("⚠️ WebSocket Connection closed unexpectedly.")
                            break
                        except Exception as e:
                            print(f"WebSocket receive error: {e}")
                            break # Break inner loop to trigger reconnection

            except ConnectionRefusedError:
                print(f"❌ Connection Refused. Retrying in {WS_RECONNECT_DELAY_S}s...")
            except Exception as e:
                print(f"❌ WebSocket generic error: {e}")

            retries += 1
            await asyncio.sleep(WS_RECONNECT_DELAY_S)
        
        print("❌ CRITICAL: Max WebSocket retries reached. Initiating REST Fallback...")
        # Fallback must be managed by the main application loop or a dedicated thread/process.
        # self._start_rest_fallback_monitor()

    def start_websocket_listener(self):
        """Starts the WebSocket listener in an asyncio event loop."""
        print("\n--- B2. Starting Real-Time WebSocket Listener (Future Feature) ---")
        # NOTE: This requires integration into the main application's event loop
        # or running in a separate process/thread managed by autoTrader/main.py.
        # asyncio.run(self._websocket_listener())
        
        # For simplicity in this step, we confirm the method structure.
        pass

    def _start_rest_fallback_monitor(self):
        """Simulates the REST API polling monitor for real-time updates."""
        print("--- B3. Starting REST API Polling Fallback (1m Polling) ---")
        # NOTE: In a real system, this would be a high-frequency threaded loop
        # that fetches the latest minute bar via REST and UPSERTs to the DB.
        pass

# --- Execution for Feature Testing ---
if __name__ == "__main__":
    # Create the necessary directory structure
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # Mock DBConnector for standalone testing
    class MockDBConnector:
        def __init__(self):
            print("Mock DB: Initializing...")
        def connect(self):
            pass
        def close(self):
            pass
        def create_tables(self):
            pass
        def get_all_raw_pair_tfs(self):
            return [f'{p}_{t}' for p in PAIRS_TO_FETCH for t in TIME_FRAMES.keys()]
        def load_raw_ohlcv(self, pair_tf):
            return None # Not implemented for mock
        def upsert_raw_ohlcv(self, df, pair_tf):
            print(f"Mock DB: UPSERT {len(df)} rows for {pair_tf}")
            pass

    # This part relies on the actual DatabaseConnector implementation:
    # connector = DatabaseConnector(db_path='./data/test_ingestion.sqlite')
    connector = MockDBConnector() 

    ingestion_system = DataIngestionSystem(db_connector=connector)
    ingestion_system.start_historical_ingestion()
    # ingestion_system.start_websocket_listener()