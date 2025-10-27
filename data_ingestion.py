#!/usr/bin/env python3
"""
Advanced Data Ingestion Engine with WebSocket and REST API support
Handles real-time OHLCV data for multiple pairs and timeframes
"""

import asyncio
import websockets
import json
import logging
import pandas as pd
import sqlalchemy as sa
from datetime import datetime, timedelta
import aiohttp
import time
import threading
from typing import Dict, List, Any, Optional
import redis
import prometheus_client as prom
from collections import defaultdict

class WebSocketManager:
    """Manages WebSocket connections with reconnection and fallback logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}
        self.reconnect_attempts = defaultdict(int)
        self.max_reconnect_attempts = config.get('websocket', {}).get('max_reconnect_attempts', 10)
        self.reconnect_delay = config.get('websocket', {}).get('reconnect_delay', 5)
        self.logger = logging.getLogger("WebSocketManager")
        
        # Metrics
        self.connection_metrics = prom.Counter('websocket_connections', 'WebSocket connection events', ['status'])
        self.message_metrics = prom.Counter('websocket_messages', 'WebSocket messages received', ['type'])
        
    async def connect(self, url: str, streams: List[str]) -> bool:
        """Establish WebSocket connection"""
        try:
            connection = await websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Subscribe to streams
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await connection.send(json.dumps(subscribe_msg))
            
            self.connections[url] = connection
            self.reconnect_attempts[url] = 0
            self.connection_metrics.labels(status='connected').inc()
            
            self.logger.info(f"WebSocket connected to {url} for {len(streams)} streams")
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            self.connection_metrics.labels(status='failed').inc()
            return False
            
    async def handle_reconnection(self, url: str, streams: List[str]):
        """Handle reconnection with exponential backoff"""
        while self.reconnect_attempts[url] < self.max_reconnect_attempts:
            delay = self.reconnect_delay * (2 ** self.reconnect_attempts[url])
            self.logger.info(f"Reconnecting in {delay} seconds...")
            await asyncio.sleep(delay)
            
            if await self.connect(url, streams):
                return
                
            self.reconnect_attempts[url] += 1
            
        self.logger.error(f"Max reconnection attempts reached for {url}")
        
    async def listen(self, url: str, callback):
        """Listen to WebSocket messages"""
        while True:
            try:
                connection = self.connections.get(url)
                if not connection:
                    break
                    
                message = await connection.recv()
                data = json.loads(message)
                
                self.message_metrics.labels(type='received').inc()
                await callback(data)
                
            except websockets.ConnectionClosed:
                self.logger.warning(f"WebSocket connection closed for {url}")
                self.connection_metrics.labels(status='closed').inc()
                await self.handle_reconnection(url, self.get_streams_for_url(url))
            except Exception as e:
                self.logger.error(f"WebSocket listen error: {e}")
                await asyncio.sleep(1)

class RESTDataFetcher:
    """Handles REST API data fetching as fallback"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = "https://api.binance.com/api/v3"
        self.session = None
        self.logger = logging.getLogger("RESTDataFetcher")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch_historical_ohlcv(self, pair: str, interval: str, limit: int = 2000) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': pair,
                'interval': interval,
                'limit': limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    df = self._parse_ohlcv_data(data, pair, interval)
                    self.logger.info(f"Fetched {len(df)} rows for {pair} {interval}")
                    return df
                else:
                    self.logger.error(f"REST API error: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {pair}: {e}")
            return None
            
    def _parse_ohlcv_data(self, data: List, pair: str, interval: str) -> pd.DataFrame:
        """Parse OHLCV data from API response"""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['pair'] = pair
        df['timeframe'] = interval
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        return df[['timestamp', 'pair', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]

class DataStorage:
    """Handles data storage in database"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = sa.create_engine(config['database']['url'])
        self.logger = logging.getLogger("DataStorage")
        self._setup_database()
        
    def _setup_database(self):
        """Setup database tables"""
        # Define table schemas
        ohlcv_metadata = sa.MetaData()
        
        self.ohlcv_table = sa.Table(
            self.config['database']['ohlcv_table'],
            ohlcv_metadata,
            sa.Column('id', sa.Integer, primary_key=True),
            sa.Column('timestamp', sa.DateTime, nullable=False),
            sa.Column('pair', sa.String(20), nullable=False),
            sa.Column('timeframe', sa.String(10), nullable=False),
            sa.Column('open', sa.Float, nullable=False),
            sa.Column('high', sa.Float, nullable=False),
            sa.Column('low', sa.Float, nullable=False),
            sa.Column('close', sa.Float, nullable=False),
            sa.Column('volume', sa.Float, nullable=False),
            sa.Column('created_at', sa.DateTime, default=datetime.now),
            sa.UniqueConstraint('timestamp', 'pair', 'timeframe', name='unique_ohlcv')
        )
        
        # Create tables
        ohlcv_metadata.create_all(self.engine)
        
    def store_ohlcv_data(self, df: pd.DataFrame):
        """Store OHLCV data in database"""
        try:
            with self.engine.begin() as connection:
                df.to_sql(
                    self.config['database']['ohlcv_table'],
                    connection,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
            self.logger.debug(f"Stored {len(df)} OHLCV records")
        except Exception as e:
            self.logger.error(f"Error storing OHLCV data: {e}")

class DataIngestionEngine:
    """Main data ingestion engine coordinating WebSocket and REST operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.loop = None
        self.thread = None
        
        # Initialize components
        self.ws_manager = WebSocketManager(config)
        self.data_storage = DataStorage(config)
        
        # Data buffers
        self.ohlcv_buffer = []
        self.latest_data = {}
        
        self.logger = logging.getLogger("DataIngestionEngine")
        
    def start(self):
        """Start the data ingestion engine in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_async, daemon=True)
        self.thread.start()
        self.logger.info("Data ingestion engine started")
        
    def _run_async(self):
        """Run async operations in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            self.logger.error(f"Async loop error: {e}")
        finally:
            self.loop.close()
            
    async def _main_loop(self):
        """Main async loop for data ingestion"""
        # First, fetch historical data
        await self._fetch_initial_data()
        
        # Then start WebSocket connections
        await self._start_websockets()
        
    async def _fetch_initial_data(self):
        """Fetch initial historical data for all pairs and timeframes"""
        self.logger.info("Fetching initial historical data...")
        
        async with RESTDataFetcher(self.config) as fetcher:
            for pair in self.config['pairs']:
                for timeframe in self.config['timeframes']:
                    df = await fetcher.fetch_historical_ohlcv(pair, timeframe, 2000)
                    if df is not None:
                        self.data_storage.store_ohlcv_data(df)
                        self._update_latest_data(df)
                        
        self.logger.info("Initial data fetching completed")
        
    async def _start_websockets(self):
        """Start WebSocket connections for real-time data"""
        # Generate WebSocket streams for Binance
        streams = []
        for pair in self.config['pairs']:
            for timeframe in self.config['timeframes']:
                # Convert timeframe to WebSocket format (e.g., 1m -> 1m, 1h -> 1h)
                stream = f"{pair.lower()}@kline_{timeframe}"
                streams.append(stream)
                
        # Connect to WebSocket
        url = "wss://stream.binance.com:9443/ws"
        success = await self.ws_manager.connect(url, streams)
        
        if success:
            await self.ws_manager.listen(url, self._handle_websocket_message)
        else:
            self.logger.error("WebSocket connection failed, switching to REST polling")
            await self._start_rest_polling()
            
    async def _handle_websocket_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            if 'e' in message and message['e'] == 'kline':
                kline_data = message['k']
                
                ohlcv_record = {
                    'timestamp': pd.to_datetime(kline_data['t'], unit='ms'),
                    'pair': message['s'],
                    'timeframe': kline_data['i'],
                    'open': float(kline_data['o']),
                    'high': float(kline_data['h']),
                    'low': float(kline_data['l']),
                    'close': float(kline_data['c']),
                    'volume': float(kline_data['v'])
                }
                
                # Store in buffer
                self.ohlcv_buffer.append(ohlcv_record)
                
                # Update latest data
                key = f"{ohlcv_record['pair']}_{ohlcv_record['timeframe']}"
                self.latest_data[key] = ohlcv_record
                
                # Batch insert every 100 records or 5 seconds
                if len(self.ohlcv_buffer) >= 100:
                    self._flush_buffer()
                    
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
            
    def _flush_buffer(self):
        """Flush buffer to database"""
        if self.ohlcv_buffer:
            df = pd.DataFrame(self.ohlcv_buffer)
            self.data_storage.store_ohlcv_data(df)
            self.ohlcv_buffer.clear()
            
    def _update_latest_data(self, df: pd.DataFrame):
        """Update latest data from dataframe"""
        for _, row in df.iterrows():
            key = f"{row['pair']}_{row['timeframe']}"
            self.latest_data[key] = row.to_dict()
            
    async def _start_rest_polling(self):
        """Fallback to REST API polling if WebSocket fails"""
        self.logger.info("Starting REST API polling fallback")
        
        while self.running:
            try:
                async with RESTDataFetcher(self.config) as fetcher:
                    for pair in self.config['pairs']:
                        for timeframe in self.config['timeframes']:
                            df = await fetcher.fetch_historical_ohlcv(pair, timeframe, 10)  # Last 10 candles
                            if df is not None:
                                self.data_storage.store_ohlcv_data(df.tail(1))  # Store only latest
                                self._update_latest_data(df.tail(1))
                                
                await asyncio.sleep(10)  # Poll every 10 seconds
                
            except Exception as e:
                self.logger.error(f"REST polling error: {e}")
                await asyncio.sleep(30)
                
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest data for processing"""
        if self.latest_data:
            return {
                'timestamp': datetime.now(),
                'data': self.latest_data.copy(),
                'type': 'market_data'
            }
        return None
        
    def stop(self):
        """Stop the data ingestion engine"""
        self.running = False
        self._flush_buffer()
        
        if self.loop and self.loop.is_running():
            self.loop.stop()
            
        self.logger.info("Data ingestion engine stopped")