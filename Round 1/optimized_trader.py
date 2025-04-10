"""
Optimized version of trader.py with performance improvements:
1. More efficient data structures (use deques instead of lists for historical data)
2. Minimize object creation in hot loops
3. Reduce unnecessary calculations and logging
4. Use vectorized operations where possible
"""
import json
from typing import Any, List, Dict
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import _pickle as cPickle
import io
import base64
import os
from collections import deque

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
        self.debug_enabled = os.environ.get('DEBUG', '0') == '1'

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        if self.debug_enabled:
            self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        # print(
        #     self.to_json(
        #         [
        #             self.compress_state(state, self.truncate(state.traderData, max_item_length)),
        #             self.compress_orders(orders),
        #             conversions,
        #             self.truncate(trader_data, max_item_length),
        #             self.truncate(self.logs, max_item_length),
        #         ]
        #     )
        # )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        self.positions = {"RAINFOREST_RESIN": 0, "KELP": 0, "SQUID_INK": 0}
        
        # Use deques with max length for more efficient historical data storage
        max_history = 200  # Limit history to control memory usage
        self.historical_data = {
            "RAINFOREST_RESIN": {"mid_price": deque(maxlen=max_history), "vwap": deque(maxlen=max_history)}, 
            "KELP": {"mid_price": deque(maxlen=max_history), "vwap": deque(maxlen=max_history)}, 
            "SQUID_INK": {"mid_price": deque(maxlen=max_history), "vwap": deque(maxlen=max_history)}
        }
        
        # Precompute RSI lookback periods based on env vars or defaults
        self.rsi_period = int(os.environ.get('PERIOD', '25'))
        self.rsi_oversold = int(os.environ.get('OVERSOLD', '32'))
        self.rsi_overbought = int(os.environ.get('OVERBOUGHT', '68'))
        self.aggression_factor = float(os.environ.get('AGGRESSION_FACTOR', '0.5'))
        self.low_buy = int(os.environ.get('LOW_BUY', '10'))
        self.high_sell = int(os.environ.get('HIGH_SELL', '85'))
        self.buy_amount = int(os.environ.get('BUY_AMOUNT', '10'))

        # Pre-allocate arrays for calculations to avoid recreating them
        self.gains_array = np.zeros(max_history)
        self.losses_array = np.zeros(max_history)
        
        # Cache for previously calculated values
        self.price_cache = {}

    def calculate_kelp_price(self, order_depth: OrderDepth):
        """Optimized version of calculate_kelp_price"""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Use list comprehension instead of filter for better performance
        filtered_ask = [price for price, qty in order_depth.sell_orders.items() if abs(qty) >= 15]
        filtered_bid = [price for price, qty in order_depth.buy_orders.items() if abs(qty) >= 15]
        
        # Quick checks before more expensive operations
        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None
        
        if mm_ask is None or mm_bid is None:
            return (best_ask + best_bid) / 2
        return (mm_ask + mm_bid) / 2
    
    def calculate_rsi(self, prices, period=None):
        """Optimized RSI calculation using pre-allocated arrays"""
        if period is None:
            period = self.rsi_period
            
        # Convert deque to list only if needed
        if isinstance(prices, deque):
            prices = list(prices)
            
        # Check if we have enough data
        if len(prices) < period + 1:
            return 50  # Return neutral RSI when insufficient data
        
        # Use numpy operations for better performance
        price_array = np.array(prices[-period-1:])
        deltas = np.diff(price_array)
        
        # Directly use pre-allocated arrays
        np.maximum(deltas, 0, out=self.gains_array[:len(deltas)])
        np.maximum(-deltas, 0, out=self.losses_array[:len(deltas)])
        
        # Use only the relevant portions of the arrays
        gains = self.gains_array[:len(deltas)]
        losses = self.losses_array[:len(deltas)]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        # Check for zero division
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def rsi_strategy(self, prices, order_depth: OrderDepth):
        """Optimized RSI strategy with cached values and fewer calculations"""
        # Only calculate RSI if we have sufficient data
        if len(prices) < self.rsi_period + 1:
            return "HOLD", 0
        
        # Cache key for this specific calculation
        cache_key = (len(prices), self.rsi_period)
        if cache_key in self.price_cache:
            rsi = self.price_cache[cache_key]
        else:
            rsi = self.calculate_rsi(prices)
            self.price_cache[cache_key] = rsi
        
        # Get current position
        position = self.positions.get("SQUID_INK", 0)
        position_limit = self.position_limits["SQUID_INK"]
        available_to_buy = position_limit - position
        available_to_sell = position_limit + position

        print(f"RSI: {rsi}, Position: {position}, Available to Buy: {available_to_buy}, Available to Sell: {available_to_sell}")
        
        # Generate trading signals
        if rsi < self.rsi_oversold:
            # Buy signal - RSI below oversold threshold
            rsi_scale_factor = max(self.aggression_factor, 
                                   min(1.0, (self.rsi_oversold - rsi) / (self.rsi_oversold - self.low_buy)))
            
            if order_depth.sell_orders and available_to_buy > 0:
                best_ask_price = min(order_depth.sell_orders.keys())
                best_ask_amount = -order_depth.sell_orders[best_ask_price]
                
                # Calculate amount to trade
                amount_to_trade = min(int(available_to_buy * rsi_scale_factor) + self.buy_amount, best_ask_amount)
                return "BUY", amount_to_trade
            
        elif rsi > self.rsi_overbought:
            # Sell signal - RSI above overbought threshold
            rsi_scale_factor = max(self.aggression_factor, 
                                   min(1.0, (rsi - self.rsi_overbought) / (self.high_sell - self.rsi_overbought)))
            
            if order_depth.buy_orders and available_to_sell > 0:
                best_bid_price = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid_price]
                
                # Calculate amount to trade
                amount_to_trade = min(int(available_to_sell * rsi_scale_factor) + self.buy_amount, best_bid_amount)
                return "SELL", amount_to_trade
                
        return "HOLD", 0
    
    def bollinger_band(self, prices, order_depth: OrderDepth, window=50, num_std=2):
        """Calculate Bollinger Bands and generate trading signals"""
        # Check if we have enough data
        if len(prices) < 1:
            return "HOLD", 0
        
        # Get the most recent price (last element)
        current_price = prices[-1]
        
        # If we don't have enough data for the window, use all available data
        actual_window = min(window, len(prices))
        if actual_window < 2:  # Need at least 2 points for a meaningful calculation
            return "HOLD", 0
        
        # Calculate moving average and standard deviation
        # Convert deque to list before slicing
        prices_list = list(prices)
        recent_prices = prices_list[-actual_window:]
        ma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        # Calculate upper and lower bands
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        # Generate trading signals with safety checks on order_depth
        if current_price < lower_band:
            # Buy signal - price below lower band
            if order_depth.sell_orders:  # Check that sell_orders exists and is not empty
                best_ask_price = min(order_depth.sell_orders.keys())
                best_ask_amount = abs(order_depth.sell_orders[best_ask_price])
                return "BUY", best_ask_amount
            else:
                return "HOLD", 0
        elif current_price > upper_band:
            # Sell signal - price above upper band
            if order_depth.buy_orders:  # Check that buy_orders exists and is not empty
                best_bid_price = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid_price]
                return "SELL", best_bid_amount
            else:
                return "HOLD", 0
        else:
            return "HOLD", 0
        
    def calculate_mid_price(self, order_depth: OrderDepth):
        """Calculate mid price based on order depth"""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    
    def calculate_vwap_price(self, order_depth: OrderDepth):
        """Calculate VWAP price based on order depth"""
        if not order_depth.sell_orders and not order_depth.buy_orders:
            return None
        
        total_volume = 0
        total_value = 0
        
        for price, qty in order_depth.sell_orders.items():
            total_volume += qty
            total_value += price * qty
                
        for price, qty in order_depth.buy_orders.items():
            total_volume += abs(qty)
            total_value += price * abs(qty)
                
        if total_volume == 0:
            return None
            
        return total_value / total_volume
    
    def update_historical_data(self, product: str, order_depth: OrderDepth):
        """Update historical data using deques for better performance"""
        mid_price = self.calculate_mid_price(order_depth)
        price_value = mid_price if mid_price is not None else 0

        vwap = self.calculate_vwap_price(order_depth)
        
        # Direct append to deque (no need to check if exists, already initialized in __init__)
        self.historical_data[product]['mid_price'].append(price_value)
        self.historical_data[product]['vwap'].append(vwap if vwap is not None else 0)
    
    def run(self, state: TradingState):
        # Extract historical data from traderData - only if necessary
        if state.traderData and len(state.traderData):
            try:
                binary_data = base64.b64decode(state.traderData)
                buffer = io.BytesIO(binary_data)
                loaded_data = cPickle.load(buffer)
                
                # Convert loaded data to deque if needed
                for product, data in loaded_data.items():
                    if product in self.historical_data:
                        for key, values in data.items():
                            if key in self.historical_data[product]:
                                if isinstance(values, list):
                                    # Convert list to deque (only copy what we need)
                                    max_len = self.historical_data[product][key].maxlen
                                    self.historical_data[product][key] = deque(values[-max_len:], maxlen=max_len)
                                else:
                                    # Already a deque or other data structure
                                    self.historical_data[product][key] = values
            except Exception:
                # Continue with existing data on error
                pass
                
        # Clear the price cache at the start of each run
        self.price_cache = {}

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            # Extract order depth for the product
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Update historical data
            self.update_historical_data(product, order_depth)
            
            # Update position from state
            self.positions[product] = state.position.get(product, 0)
            
            # Process each product type
            if product == "RAINFOREST_RESIN":
                # RAINFOREST_RESIN strategy implementation
                # ...existing code...
                pass
                
            elif product == "KELP":
                # KELP strategy implementation
                # ...existing code...
                pass
                
            elif product == "SQUID_INK":
                price_data = self.historical_data.get(product, None)
                # # Initialize tracking variables
                # position = state.position.get(product,0)
                # position_limit = self.position_limits[product]
                # buy_limit = position_limit - position
                # sell_limit = -position_limit - position  # negative limit
                # bought_amount, sold_amount = 0, 0
                # left_to_buy = buy_limit - bought_amount
                # left_to_sell = sold_amount - sell_limit

                # indicator, amount_to_trade = self.bollinger_band(price_data['mid_price'], order_depth,
                #                                                      window=int(os.environ.get('WINDOW')),
                #                                                      num_std=float(os.environ.get('NUM_STD')))
                # if indicator == "BUY":
                #     if position < self.position_limits[product]:
                #         best_ask_price, best_ask_amount = list(order_depth.sell_orders.items())[0] 
                #         orders.append(Order(product, best_ask_price, best_ask_amount))
                #         position += amount_to_trade
                #         bought_amount += amount_to_trade
                # elif indicator == "SELL":
                #     if position > -self.position_limits[product]:
                #         best_bid_price, best_bid_amount = list(order_depth.buy_orders.items())[0]
                #         orders.append(Order(product, best_bid_price, -best_bid_amount))
                #         position -= amount_to_trade
                #         sold_amount -= amount_to_trade
                
                # # Debug to check what data we have
                # logger.print(f"SQUID_INK price_data: {bool(price_data)}")
                # if price_data:
                #     logger.print(f"VWAP available: {bool('vwap' in price_data)}")
                #     logger.print(f"VWAP length: {len(price_data['vwap']) if 'vwap' in price_data else 0}")
                
                # Changed this condition to check if 'vwap' exists and has data
                if price_data and 'mid_price' in price_data and len(price_data['mid_price']) > 0:
                    # Get RSI trading signal
                    indicator, amount_to_trade = self.rsi_strategy(
                        price_data['mid_price'], order_depth)
                    
                    if indicator == "BUY" and amount_to_trade > 0:
                        if order_depth.sell_orders:
                            best_ask_price = min(order_depth.sell_orders.keys())
                            orders.append(Order(product, best_ask_price, amount_to_trade))
                            logger.print(f"Placing BUY order for {amount_to_trade} at {best_ask_price}")
                    
                    elif indicator == "SELL" and amount_to_trade > 0:
                        if order_depth.buy_orders:
                            best_bid_price = max(order_depth.buy_orders.keys())
                            orders.append(Order(product, best_bid_price, -amount_to_trade))
                            logger.print(f"Placing SELL order for {amount_to_trade} at {best_bid_price}")
                else:
                    logger.print("Not enough SQUID_INK data for trading")
            
            # Add orders to result if we have any
            if orders:
                result[product] = orders

        # Serialize historical data efficiently
        buffer = io.BytesIO()
        # Convert deques to lists for serialization
        serializable_data = {}
        for product, data in self.historical_data.items():
            serializable_data[product] = {k: list(v) if isinstance(v, deque) else v for k, v in data.items()}
            
        cPickle.dump(serializable_data, buffer)
        binary_data = buffer.getvalue()
        traderData = base64.b64encode(binary_data).decode('ascii')
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData