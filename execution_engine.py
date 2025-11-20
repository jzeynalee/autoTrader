import hashlib
import hmac
import base64
import json
import time
import requests
import uuid
import logging
import urllib.parse
import random
import string
from typing import Dict, Optional, Any, Tuple
from decimal import Decimal

# Local imports
from .db_connector import DatabaseConnector
from .models import Signal, TradeDirection, OrderSide


# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class LBankExecutionEngine:
    """
    Robust Execution Engine for LBank Exchange V2 API.
    Handles authentication, order management, and database synchronization.
    
    API Documentation Reference: https://www.lbank.com/docs/index.html
    """

    BASE_URL = "https://api.lbkex.com/v2"
    
    # Endpoints
    ENDPOINTS = {
        "create_order": "/supplement/create_order.do",
        "cancel_order": "/supplement/cancel_order.do",
        "orders_info": "/supplement/orders_info.do",
        "user_info": "/supplement/user_info.do",
        "currency_pairs": "/currencyPairs.do"
    }

    def __init__(self, api_key: str, api_secret: str, db_connector: Optional[DatabaseConnector] = None):
        """
        Initialize the Execution Engine.
        
        Args:
            api_key: LBank API Key
            api_secret: LBank Secret Key
            db_connector: Instance of DatabaseConnector for logging trades/positions
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.db = db_connector
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/x-www-form-urlencoded"
        })

    def _get_timestamp(self) -> str:
        """Current timestamp in milliseconds."""
        return str(int(time.time() * 1000))

    def _get_echostr(self, length=30) -> str:
        """Generate a random echo string for signature uniqueness."""
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate LBank V2 Signature.
        
        Algorithm:
        1. Filter out 'sign' parameter.
        2. Sort parameters alphabetically by key.
        3. Create query string (key=value&...).
        4. Compute MD5 digest of the query string (uppercase hex).
        5. Compute HmacSHA256 signature of the MD5 digest using API Secret.
        6. Return Base64 encoded signature.
        """
        # 1. Filter & Sort
        filtered_params = {k: v for k, v in params.items() if k != 'sign'}
        sorted_keys = sorted(filtered_params.keys())
        
        # 2. Build Query String
        query_string_parts = []
        for key in sorted_keys:
            query_string_parts.append(f"{key}={filtered_params[key]}")
        query_string = '&'.join(query_string_parts)
        
        # 3. MD5 Digest (Uppercase)
        md5_hash = hashlib.md5(query_string.encode('utf-8')).hexdigest().upper()
        
        # 4. HmacSHA256 Signature
        # Note: Secret key must be bytes
        secret_bytes = self.api_secret.encode('utf-8')
        message_bytes = md5_hash.encode('utf-8')
        
        hmac_hash = hmac.new(secret_bytes, message_bytes, hashlib.sha256).digest()
        
        # 5. Base64 Encode
        signature = base64.b64encode(hmac_hash).decode('utf-8')
        return signature

    def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = True) -> Dict[str, Any]:
        """
        Execute a robust HTTP request with retries and error handling.
        """
        if params is None:
            params = {}

        url = f"{self.BASE_URL}{endpoint}"

        if signed:
            # Add required auth params
            params['api_key'] = self.api_key
            params['signature_method'] = 'HmacSHA256'
            params['timestamp'] = self._get_timestamp()
            params['echostr'] = self._get_echostr()
            
            # Generate signature
            params['sign'] = self._generate_signature(params)

        # LBank V2 uses POST with x-www-form-urlencoded for most authenticated actions
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if method.upper() == "POST":
                    response = self.session.post(url, data=params, timeout=10)
                else:
                    response = self.session.get(url, params=params, timeout=10)

                response.raise_for_status()
                data = response.json()

                # Check LBank logic-level success
                if data.get('result') == 'true':
                    return data
                else:
                    error_code = data.get('error_code')
                    msg = data.get('msg', 'Unknown Error')
                    logger.error(f"LBank API Error {error_code}: {msg} | Endpoint: {endpoint}")
                    
                    # Handle specific recoverable errors (e.g., rate limit)
                    if error_code == 10012: # Request too frequent
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    
                    return {"error": True, "code": error_code, "msg": msg}

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error on attempt {attempt+1}: {e}")
                time.sleep(1)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from {url}")
                return {"error": True, "msg": "Invalid JSON response"}

        return {"error": True, "msg": "Max retries exceeded"}

    # =========================================================================
    # Public Interface
    # =========================================================================

    def get_account_balance(self) -> Dict[str, float]:
        """
        Fetch current asset balances.
        Returns a dict: {'BTC': {'free': 0.1, 'frozen': 0.0}, ...}
        """
        endpoint = self.ENDPOINTS["user_info"]
        response = self._request("POST", endpoint, signed=True)
        
        balances = {}
        if response.get("result") == "true":
            info = response.get("data", {})
            
            # Map 'free' and 'freeze' assets
            free = info.get("free", {})
            frozen = info.get("freeze", {})
            
            all_assets = set(free.keys()) | set(frozen.keys())
            
            for asset in all_assets:
                free_amt = float(free.get(asset, 0))
                frozen_amt = float(frozen.get(asset, 0))
                if free_amt > 0 or frozen_amt > 0:
                    balances[asset.upper()] = {
                        "free": free_amt,
                        "frozen": frozen_amt,
                        "total": free_amt + frozen_amt
                    }
        return balances

    def create_order(self, symbol: str, side: str, amount: float, price: float, order_type: str = 'buy') -> Dict[str, Any]:
        """
        Place a Limit Order.
        
        Args:
            symbol: Pair name (e.g., 'btc_usdt')
            side: 'buy' or 'sell' (LBank uses 'type' parameter for this)
            amount: Quantity to trade
            price: Limit price
            order_type: 'buy' | 'sell' | 'buy_market' | 'sell_market'
                        (Defaults to limit orders as they are safer)
        
        Returns:
            Dict containing order_id and status.
        """
        # Normalize symbol (LBank uses lower case with underscore, e.g., btc_usdt)
        normalized_symbol = symbol.lower().replace('/', '_')
        
        # LBank 'type' parameter defines side AND order type
        # For limit orders: 'buy', 'sell'
        # For market orders: 'buy_market', 'sell_market'
        
        # Determine internal LBank type string
        lbank_type = side.lower() 
        if 'market' in order_type:
            lbank_type += "_market"

        params = {
            "symbol": normalized_symbol,
            "type": lbank_type,
            "price": str(price),
            "amount": str(amount),
            "custom_id": str(uuid.uuid4())[:20] # Traceable ID
        }
        
        logger.info(f"Placing Order: {lbank_type} {amount} {normalized_symbol} @ {price}")
        
        result = self._request("POST", self.ENDPOINTS["create_order"], params, signed=True)
        
        if result.get("result") == "true":
            data = result.get("data", {})
            order_id = data.get("order_id")
            logger.info(f"Order Placed Successfully: ID {order_id}")
            return {
                "success": True,
                "order_id": order_id,
                "symbol": normalized_symbol,
                "side": side,
                "price": price,
                "amount": amount,
                "raw": data
            }
        else:
            logger.error(f"Order Placement Failed: {result.get('msg')}")
            return {"success": False, "error": result.get("msg")}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel a specific order.
        """
        normalized_symbol = symbol.lower().replace('/', '_')
        params = {
            "symbol": normalized_symbol,
            "order_id": order_id
        }
        
        result = self._request("POST", self.ENDPOINTS["cancel_order"], params, signed=True)
        
        if result.get("result") == "true":
            logger.info(f"Order {order_id} cancelled successfully.")
            return True
        
        logger.warning(f"Failed to cancel order {order_id}: {result.get('msg')}")
        return False

    def fetch_open_orders(self, symbol: str, page_no: int = 1, page_length: int = 100) -> list:
        """
        Fetch current open orders for a symbol.
        """
        normalized_symbol = symbol.lower().replace('/', '_')
        params = {
            "symbol": normalized_symbol,
            "current_page": page_no,
            "page_length": page_length
        }
        
        result = self._request("POST", self.ENDPOINTS["orders_info"], params, signed=True)
        
        if result.get("result") == "true":
            return result.get("data", {}).get("orders", [])
        return []

    def execute_signal(self, signal: Signal) -> bool:
        """
        High-level method to execute a strategy signal.
        
        1. Validates account balance.
        2. Calculates safe position size (if not provided).
        3. Places the order.
        4. Logs the position to the database.
        
        Args:
            signal: Dict with keys ['pair', 'direction', 'entry_price', 'stop_loss', 'take_profit', 'strategy_id']
        """
        symbol = signal.get('pair', 'btc_usdt')
        direction = signal.get('direction', 'bullish')
        price = float(signal.get('entry_price', 0))
        
        # Map 'bullish'/'bearish' or 'long'/'short' to 'buy'/'sell'
        if direction.lower() in ['bullish', 'long']:
            side = 'buy'
        else:
            side = 'sell'

        # 1. Balance Check
        # Extract base/quote currency (e.g., btc_usdt -> BTC, USDT)
        try:
            base, quote = symbol.split('_')
        except ValueError:
            base, quote = symbol[:3], symbol[3:] # Fallback assumption
            
        balances = self.get_account_balance()
        
        # Determine required funds
        if side == 'buy':
            required_asset = quote.upper()
            available = balances.get(required_asset, {}).get('free', 0.0)
            trade_cost = available * 0.10 
            amount = trade_cost / price
        else:
            required_asset = base.upper()
            available = balances.get(required_asset, {}).get('free', 0.0)
            amount = available * 0.10 

        # Minimum order size check 
        if (amount * price) < 10:
            logger.warning(f"Signal ignored: Calculated trade value ({amount*price:.2f}) too small.")
            return False

        # 2. Place Order
        amount = round(amount, 4) 
        price = round(price, 2)
        
        order_result = self.create_order(symbol, side, amount, price)
        
        if order_result['success']:
            # 3. Log to DB (Positions Table)
            if self.db:
                try:
                    # Use Unix timestamp for entry_time
                    position_data = {
                        "position_id": order_result['order_id'],
                        "strategy_id": signal.get('strategy_id', 'MANUAL'),
                        "pair": symbol,
                        "direction": direction,
                        "entry_price": price,
                        "quantity": amount,
                        "stop_loss": signal.get('stop_loss'),
                        "take_profit": signal.get('take_profit'),
                        "entry_time": time.time(),  # Corrected to use time.time()
                        "status": "open"
                    }
                    
                    query = """
                    INSERT INTO positions (
                        position_id, strategy_id, pair, direction,
                        entry_price, quantity, stop_loss, take_profit,
                        entry_time, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    self.db.execute(query, tuple(position_data.values()))
                    logger.info(f"Position {order_result['order_id']} logged to DB.")
                except Exception as e:
                    logger.error(f"Failed to log position to DB: {e}")
            
            return True
            
        return False

# --- Testing Block ---
if __name__ == "__main__":
    # Import the configuration object
    try:
        from .config import CONFIG
    except ImportError:
        import sys
        import os
        import configparser
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), '../config.ini'))
        CONFIG = config

    # Fetch keys from config.ini
    try:
        TEST_KEY = CONFIG['lbank']['api_key']
        TEST_SECRET = CONFIG['lbank']['api_secret']
        print(f"Loaded keys from config.ini for test.")
    except KeyError:
        print("⚠️  Keys not found in [lbank] section of config.ini. Using dummies.")
        TEST_KEY = "test_key"
        TEST_SECRET = "test_secret"
    
    # Initialize engine with keys from config
    engine = LBankExecutionEngine(TEST_KEY, TEST_SECRET)
    
    # 1. Test Signature Generation
    test_params = {
        "symbol": "eth_btc",
        "type": "buy",
        "price": "100",
        "amount": "10"
    }
    
    print("Test Signature Generation:")
    try:
        # Manually injecting timestamp/echo for visibility
        test_params['signature_method'] = 'HmacSHA256'
        test_params['timestamp'] = str(int(time.time() * 1000))
        test_params['echostr'] = 'test_echo_string'
        
        sig = engine._generate_signature(test_params)
        print(f"Params: {test_params}")
        print(f"Signature: {sig}")
        print("Signature generation logic executed successfully.")
    except Exception as e:
        print(f"Signature generation failed: {e}")