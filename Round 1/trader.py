import json
from typing import Any, List

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
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

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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
        
        self.params = {
            "KELP": {
                "take_width": 1,         # Threshold for taking market orders
                "clear_width": 0,         # Threshold for clearing positions
                "prevent_adverse": True,  # Avoid trading against large orders
                "adverse_volume": 15,     # Volume threshold for large orders
                "disregard_edge": 1,      # Ignore orders near fair value
                "join_edge": 0,           # Join existing orders within this range
                "default_edge": 1,        # Default spread from fair value
                # GLFT parameters
                "gamma": 0.2,             # Risk aversion parameter
                "sigma": 0.75,            # Volatility estimate from historical data
                "order_amount": 50,       # Optimal order size from backtesting
                "mu": 0                   # Assume no drift
            },
            "SQUID_INK": {
                "ma_window": 10,       # Moving average window
                "threshold_factor": 2, # Multiply std dev to determine entry threshold
            }
        }
        # Price history for volatility calculation
        self.price_history = {"KELP": []}
        self.mid_prices = {"KELP": None}
        
    
    def calculate_kelp_price(self, order_depth: OrderDepth):
        """Calculate fair price of kelp based on adverse volume market making"""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Store mid price for reference
            self.mid_prices["KELP"] = (best_ask + best_bid) / 2
            
            # Find large volume orders
            filtered_ask = [ #adverse volume
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params["KELP"]["adverse_volume"]
            ]
            filtered_bid = [ #adverse volume
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params["KELP"]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            # Calculate fair price
            if mm_ask == None or mm_bid == None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            
            # Update price history for volatility calculation
            self.price_history["KELP"].append(mmmid_price)
            if len(self.price_history["KELP"]) > 50:  # Keep a rolling window
                self.price_history["KELP"].pop(0)
                
            return mmmid_price
        return None
    
    def get_max_volume_prices(self, order_depth: OrderDepth):
        """Find prices with maximum volume on bid and ask sides"""
        max_bid_vol = 0
        max_bid_price = None
        for price, volume in order_depth.buy_orders.items():
            if volume > max_bid_vol:
                max_bid_vol = volume
                max_bid_price = price
                
        max_ask_vol = 0
        max_ask_price = None
        for price, volume in order_depth.sell_orders.items():
            if abs(volume) > max_ask_vol:
                max_ask_vol = abs(volume)
                max_ask_price = price
                
        if max_bid_price is None and len(order_depth.buy_orders) > 0:
            max_bid_price = max(order_depth.buy_orders.keys())
            
        if max_ask_price is None and len(order_depth.sell_orders) > 0:
            max_ask_price = min(order_depth.sell_orders.keys())
            
        return max_bid_price, max_ask_price
    
    def calculate_order_book_imbalance(self, order_depth: OrderDepth):
        """Calculate the order book imbalance"""
        total_bid_vol = sum(vol for vol in order_depth.buy_orders.values())
        total_ask_vol = sum(abs(vol) for vol in order_depth.sell_orders.values())
        total_vol = total_bid_vol + total_ask_vol
        
        if total_vol > 0:
            return (total_bid_vol - total_ask_vol) / total_vol
        return 0

    def trade_squid_ink(self, state, orders: List['Order']):
        """
        Trade SQUID_INK using a simplified mean reversion strategy.
        """
        product = "SQUID_INK"
        
        # Make sure we have market data
        if product not in state.order_depths:
            return orders
        
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        self.positions[product] = position
        
        # Check if we have bids and asks to form a mid-price
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        
        # Initialize price history if needed
        if product not in self.price_history:
            self.price_history[product] = []
        
        # Update rolling price history
        self.price_history[product].append(mid_price)
        
        # Extract parameters
        ma_window = self.params[product]["ma_window"]
        threshold_factor = self.params[product]["threshold_factor"]
        position_limit = self.position_limits[product]
        
        # Ensure enough data points to compute a moving average
        if len(self.price_history[product]) < ma_window:
            return orders
        
        # Keep price history from growing infinitely
        # (Keep about 2 x ma_window for safety)
        max_history = ma_window * 2
        if len(self.price_history[product]) > max_history:
            self.price_history[product] = self.price_history[product][-max_history:]
        
        # Compute the simple moving average (SMA) and volatility (std dev)
        recent_prices = self.price_history[product][-ma_window:]
        sma = sum(recent_prices) / len(recent_prices)
        
        # Calculate standard deviation for the same window
        mean = sma
        variance = sum((p - mean)**2 for p in recent_prices) / len(recent_prices)
        stdev = math.sqrt(variance)
        
        # Define an adaptive threshold = threshold_factor * stdev
        threshold_up = sma + threshold_factor * stdev
        threshold_down = sma - threshold_factor * stdev
        
        # Trading signal logic:
        # - If mid_price is above threshold_up -> short
        # - If mid_price is below threshold_down -> long
        # - Exit when mid_price crosses back near SMA
        signal = "none"
        
        if mid_price > threshold_up:
            signal = "short"
        elif mid_price < threshold_down:
            signal = "long"
        
        # Decide how to update positions
        if signal == "long":
            # Target a full long position
            target_position = position_limit
            if position < target_position:
                buy_size = target_position - position
                # Execute buy near the ask
                orders.append(Order(product, best_ask, buy_size))
        
        elif signal == "short":
            # Target a full short position
            target_position = -position_limit
            if position > target_position:
                sell_size = target_position - position  # This will be negative
                # Execute sell near the bid
                orders.append(Order(product, best_bid, sell_size))
        
        else:
            # Check if we have an existing position that we want to unwind
            # (exit positions when price goes back near the SMA)
            if position != 0:
                # If we're long and the price has dropped below SMA, or
                # we’re short and price is above SMA, we can exit.
                # For simplicity, exit any time mid_price crosses the SMA.
                # You could also add a small "band" around SMA.
                
                # If currently long but mid_price < sma => exit
                # If currently short but mid_price > sma => exit
                should_exit = False
                if position > 0 and mid_price < sma + threshold_factor * stdev:
                    should_exit = True
                elif position < 0 and mid_price > sma - threshold_factor * stdev:
                    should_exit = True
                
                if should_exit:
                    # Sell if position > 0, buy if position < 0
                    if position > 0:
                        # Sell the entire long position at the bid
                        orders.append(Order(product, best_bid, -position))
                    else:
                        # Buy back the entire short position at the ask
                        orders.append(Order(product, best_ask, -position))
        
        return orders
    
    #def trade_kelp_glft(self, state: TradingState, left_to_buy, left_to_sell, orders):
        """
        Implement GLFT market making strategy for KELP with adverse selection protection
        """
        product = "KELP"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        
        # Update position tracking
        self.positions[product] = position
        
        # Strategy parameters from config
        params = self.params[product]
        gamma = params["gamma"]
        sigma = params["sigma"]
        order_amount = params["order_amount"]
        mu = params["mu"]
        
        # Calculate fair price
        fair_price = self.calculate_kelp_price(order_depth)
        if fair_price is None:
            return []  # No fair price available
            
        # Get market variables
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fair_price - 5
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fair_price + 5
        max_bid_price, max_ask_price = self.get_max_volume_prices(order_depth)
        
        # Detect adverse selection risk
        #adverse_risk = self.detect_adverse_selection(order_depth)
        
        # Calculate order book imbalance
        imbalance = self.calculate_order_book_imbalance(order_depth)
        
        # Apply GLFT model
        q = position / order_amount  # Normalized position
        
        # Calculate arrival rates
        kappa_b = 1 / max((fair_price - best_bid) - 1, 1)
        kappa_a = 1 / max((best_ask - fair_price) - 1, 1)
        
        # Constants for optimal spread calculation
        A_b = 0.25
        A_a = 0.25
        
        # Calculate optimal distances from fair price
        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) + \
                 (-mu / (gamma * sigma**2) + (2 * q + 1) / 2) * \
                 math.sqrt((sigma**2 * gamma) / (2 * kappa_b * A_b) * \
                         (1 + gamma / kappa_b)**(1 + kappa_b / gamma))
        
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + \
                 (mu / (gamma * sigma**2) - (2 * q - 1) / 2) * \
                 math.sqrt((sigma**2 * gamma) / (2 * kappa_a * A_a) * \
                         (1 + gamma / kappa_a)**(1 + kappa_a / gamma))
        
        # Adjust for adverse selection risk
        """if adverse_risk:
            if imbalance > 0.2:  # More buying pressure
                delta_a *= 1.5   # Widen ask spread
            elif imbalance < -0.2:  # More selling pressure
                delta_b *= 1.5   # Widen bid spread"""
        
        # Calculate optimal prices
        """p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)
        
        # Apply constraints
        p_b = min(p_b, fair_price - 1)  # Never buy above fair price
        p_b = min(p_b, best_bid + 1)  # Stay competitive
        if max_bid_price is not None:
            p_b = max(p_b, max_bid_price + 1)  # Don't place too far from max volume
        
        p_a = max(p_a, fair_price)  # Never sell below fair price
        p_a = max(p_a, best_ask - 1)  # Stay competitive
        if max_ask_price is not None:
            p_a = min(p_a, max_ask_price - 1)  # Don't place too far from max volume"""
            
        
        p_a = best_ask - 1
        p_b = best_bid + 1
        
        # Calculate order sizes with position limits
        position_limit = self.position_limits[product]
        buy_limit = left_to_buy
        sell_limit = left_to_sell
        
        buy_amount = buy_limit
        logger.print(buy_amount)
        sell_amount = sell_limit
        
        
        # Skip placing orders in high adverse selection scenarios
        #skip_buy = adverse_risk and imbalance < -0.3
        #skip_sell = adverse_risk and imbalance > 0.3
        if p_a > p_b:
            if buy_amount > 0: #and not skip_buy:
                orders.append(Order(product, int(p_b), int(buy_amount)))
                logger.print(f"GLFT BUY {product}: {buy_amount}x @ {p_b}")
            
            if sell_amount > 0: #and not skip_sell:
                orders.append(Order(product, int(p_a), -int(sell_amount)))
                logger.print(f"GLFT SELL {product}: {sell_amount}x @ {p_a}")
        
        return orders
    
    
    def run(self, state: TradingState):

		# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "RAINFOREST_RESIN":
                self.positions[product] = state.position.get(product,0)
                acceptable_price = 10000
                buy_limit = self.position_limits[product] - self.positions[product]
                sell_limit = - self.position_limits[product] - self.positions[product] #negative limit
                bought_amount, sold_amount = 0, 0
                
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.positions[product] < self.position_limits[product]:
                        if int(best_ask) < acceptable_price:
                            buy_amount = min(self.position_limits[product] - self.positions[product],-best_ask_amount)
                            logger.print("BUY RAINFOREST_RESIN", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            #logger.print(self.positions[product], state.position.get(product,0)) #TESTING IF POSITION ARE UPDATED IMMEDIATELY, TRADES EXECUTING IMMEDIATELY ::: CONFIRMED STATE.POSITIONS UPDATED
                            bought_amount += buy_amount
                            if buy_amount == -best_ask_amount:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]
                        if int(best_ask) == acceptable_price and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            logger.print("BUY RAINFOREST_RESIN", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                    
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limits[product]:
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limits[product], best_bid_amount)
                            logger.print("SELL RAINFOREST_RESIN", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                            if sell_amount == best_bid_amount:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                        if int(best_bid) == acceptable_price and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            logger.print("SELL RAINFOREST_RESIN", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                
                left_to_sell = sold_amount - sell_limit #positive number
                left_to_buy = buy_limit - bought_amount 
                
                logger.print("MAKING RAINFOREST_RESIN BUY", str(left_to_buy) + "x", best_bid+1)
                orders.append(Order(product,best_bid+1,left_to_buy))
                logger.print("MAKING RAINFOREST_RESIN SELL", str(left_to_sell) + "x", best_ask-1)
                orders.append(Order(product,best_ask-1,-left_to_sell))
                
                            
                result[product] = orders
            
            
                
            elif product == "KELP":
                fair_value = self.calculate_kelp_price(order_depth)
                #fair_value = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
                logger.print("Fair value kelp: " + str(fair_value))
                if fair_value is not None:
                    # Initialize tracking variables
                    position = state.position.get(product,0)
                    position_limit = self.position_limits[product]
                    buy_limit = position_limit - position
                    sell_limit = -position_limit - position  # negative limit
                    bought_amount, sold_amount = 0, 0
                    
                    
                    
                                
                    
                    # Market taking - sell side (buying)
                    if len(order_depth.sell_orders) != 0:
                        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                        
                        # Check if we can buy at favorable price
                        i = 0
                        while position < self.position_limits[product] and best_ask < fair_value:
                            buy_amount = min(position_limit - position, -best_ask_amount)
                            orders.append(Order(product, best_ask, buy_amount))
                            logger.print("BUY KELP", str(buy_amount) + "x", best_ask)
                            position += buy_amount
                            bought_amount += buy_amount
                            if buy_amount == -best_ask_amount and len(order_depth.sell_orders) > i + 1:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[i+1]
                            else:
                                break
                        
                        #position reduction to reduce risk
                        if int(best_ask) == fair_value and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            logger.print("BUY KELP", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
            
                    
                    # Market taking - buy side (selling)
                    if len(order_depth.buy_orders) != 0:
                        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                        
                        # Check if we can sell at favorable price
                        i = 0
                        while position > -self.position_limits[product] and best_bid > fair_value:
                            sell_amount = min(position + position_limit, best_bid_amount)
                            logger.print("SELL KELP", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product, best_bid, -sell_amount))
                            position -= sell_amount
                            sold_amount -= sell_amount
                            if sell_amount == best_bid_amount and len(order_depth.buy_orders) > i + 1:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[i+1]
                                i += 1
                            else:
                                break
                            
                        #position reduction to reduce risk
                        if int(best_bid) == fair_value and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            logger.print("SELL KELP", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                    
                    # Market making
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    bb, ba = self.get_max_volume_prices(order_depth)
                    
                    min_edge = 1
                    aaf = [
                        price
                        for price in order_depth.sell_orders.keys()
                        if price >= round(fair_value + min_edge)
                    ]
                    bbf = [
                        price
                        for price in order_depth.buy_orders.keys()
                        if price <= round(fair_value - min_edge)
                    ]
                    baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
                    bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
                    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                        if best_ask - 1 > best_bid + 1:
                            left_to_buy = buy_limit - bought_amount
                            left_to_sell = sold_amount - sell_limit
                            
                            # Make market on both sides
                            if left_to_buy > 0:
                                orders.append(Order(product, bb + 1, left_to_buy))
                                logger.print("MAKING KELP BUY", str(left_to_buy) + "x", bb+1)
                            if left_to_sell > 0:
                                orders.append(Order(product, ba - 1, -left_to_sell))
                                logger.print("MAKING KELP SELL", str(left_to_sell) + "x", ba-1)
                        elif ba - 1 > bb+ 1:
                            # Calculate remaining capacity
                            left_to_buy = buy_limit - bought_amount
                            left_to_sell = sold_amount - sell_limit
                            
                            # Make market on both sides
                            if left_to_buy > 0:
                                orders.append(Order(product, bb + 1, left_to_buy))
                                logger.print("MAKING KELP BUY", str(left_to_buy) + "x", bb+1)
                            if left_to_sell > 0:
                                orders.append(Order(product, ba - 1, -left_to_sell))
                                logger.print("MAKING KELP SELL", str(left_to_sell) + "x", ba-1)
                            
                    # Add orders to result
                    result[product] = orders
            
            elif product == "SQUID_INK":
                product_orders = []
                product_orders = self.trade_squid_ink(state, product_orders)
                result[product] = product_orders

        # String value holding Trader state data required.
        traderData = json.dumps({
            "positions": self.positions,
            "mid_prices": self.mid_prices
        })
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        logger.print(result)
        return result, conversions, traderData
    
    