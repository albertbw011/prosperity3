import json
from typing import Any, List
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import pandas as pd
import _pickle as cPickle
import io
import base64

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
        self.historical_data = {"RAINFOREST_RESIN": None, "KELP": None, "SQUID_INK": None}

        # Strategy parameters
        self.params = {
            "KELP": {
                "take_width": 1,         # Threshold for taking market orders
                "clear_width": 0,         # Threshold for clearing positions
                "prevent_adverse": True,  # Avoid trading against large orders
                "adverse_volume": 15,     # Volume threshold for large orders
                "disregard_edge": 1,      # Ignore orders near fair value
                "join_edge": 0,           # Join existing orders within this range
                "default_edge": 1         # Default spread from fair value
            }
        }
        
    def calculate_kelp_price(self, order_depth: OrderDepth):
        """Calculate fair price of kelp based on adverse volume market making"""
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [ #adverse volume
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= 15
            ]
            filtered_bid = [ #adverse volume
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= 15
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            return mmmid_price
        return None
    
    def bollinger_band(self, prices: pd.Series, order_depth: OrderDepth, window=25, num_std=0.03):
        """Calculate Bollinger Bands and generate trading signals"""
        # Check if we have enough data
        if len(prices) < 1:
            return "HOLD", 0
        
        # Get the most recent price (last row)
        current_price = prices.iloc[-1]
        
        # If we don't have enough data for the window, use all available data
        actual_window = min(window, len(prices))
        if actual_window < 2:  # Need at least 2 points for a meaningful calculation
            return "HOLD", 0
        
        # Calculate moving average and standard deviation
        recent_prices = prices.iloc[-actual_window:]
        ma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        # Calculate upper and lower bands
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        # Generate trading signals with safety checks on order_depth
        if current_price < lower_band:
            # Buy signal - price below lower band
            if len(order_depth.buy_orders) > 0:
                best_bid_price = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid_price]
                return "BUY", best_bid_amount
            else:
                return "HOLD", 0
        elif current_price > upper_band:
            # Sell signal - price above upper band
            if len(order_depth.sell_orders) > 0:
                best_ask_price = min(order_depth.sell_orders.keys())
                best_ask_amount = abs(order_depth.sell_orders[best_ask_price])
                return "SELL", best_ask_amount
            else:
                return "HOLD", 0
        else:
            return "HOLD", 0
        
    def calculate_mid_price(self, order_depth: OrderDepth):
        """Calculate mid price based on order depth"""
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    
    def update_historical_data(self, product: str, order_depth: OrderDepth):
        """Update historical data dictionary with new data"""
        mid_price = self.calculate_mid_price(order_depth)
        
        # If mid_price is None, use 0 as a fallback
        price_value = mid_price if mid_price is not None else 0
        
        # Check if product exists in historical_data and initialize if needed
        if self.historical_data[product] is None:
            # Initialize with a DataFrame containing a mid_price column
            self.historical_data[product] = pd.DataFrame({'mid_price': [price_value]})
        else:
            # Add new row to existing DataFrame
            new_row = pd.DataFrame({'mid_price': [price_value]})
            self.historical_data[product] = pd.concat([self.historical_data[product], new_row], ignore_index=True)
        
        return None
    
    #def take_best_orders(self, product: str, fair_value: float, order_depth: OrderDepth, orders: List[Order]):
        """Take immediately profitable orders based on fair value"""
        position = self.positions[product]
        position_limit = self.position_limits[product]
        take_width = self.params[product]["take_width"]
        prevent_adverse = self.params[product]["prevent_adverse"]
        adverse_volume = self.params[product]["adverse_volume"]
        
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Check sell orders for buying opportunities
        if len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            
            # Only trade if not preventing adverse selection or volume is small enough
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                # Take if price is favorable compared to fair value
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
        
        # Check buy orders for selling opportunities
        if len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            # Only trade if not preventing adverse selection or volume is small enough
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                # Take if price is favorable compared to fair value
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
        
        return buy_order_volume, sell_order_volume

    #def clear_position(self, product: str, fair_value: float, order_depth: OrderDepth, orders: List[Order], buy_volume: int, sell_volume: int):
        """Clear existing position if possible at favorable prices"""
        position = self.positions[product]
        position_after_take = position + buy_volume - sell_volume
        clear_width = self.params[product]["clear_width"]
        position_limit = self.position_limits[product]
        
        # Calculate prices for clearing
        fair_for_bid = round(fair_value - clear_width)
        fair_for_ask = round(fair_value + clear_width)
        
        # Calculate remaining capacity
        buy_capacity = position_limit - (position + buy_volume)
        sell_capacity = position_limit + (position - sell_volume)
        
        # Try to clear positive position by selling
        if position_after_take > 0:
            # Find buy orders at or above our ask price
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_capacity, clear_quantity)
            
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_volume += abs(sent_quantity)
        
        # Try to clear negative position by buying
        if position_after_take < 0:
            # Find sell orders at or below our bid price
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_capacity, clear_quantity)
            
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_volume += abs(sent_quantity)
        
        return buy_volume, sell_volume
    
    #def make_market(self, product: str, fair_value: float, order_depth: OrderDepth, orders: List[Order], buy_volume: int, sell_volume: int):
        """Place market making orders on both sides of the book"""
        position = self.positions[product]
        position_limit = self.position_limits[product]
        disregard_edge = self.params[product]["disregard_edge"]
        join_edge = self.params[product]["join_edge"]
        default_edge = self.params[product]["default_edge"]
        
        # Find orders outside the disregard zone
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        # Find best prices to reference
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        
        # Calculate ask price - either join, penny, or use default
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join existing order
            else:
                ask = best_ask_above_fair - 1  # penny existing order
        
        # Calculate bid price - either join, penny, or use default
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # join existing order
            else:
                bid = best_bid_below_fair + 1  # penny existing order
        
        # Calculate order quantities
        buy_quantity = position_limit - (position + buy_volume)
        sell_quantity = position_limit + (position - sell_volume)
        
        # Place orders if quantities are positive
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
            buy_volume += buy_quantity
            
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
            sell_volume += sell_quantity
            
        return buy_volume, sell_volume
    
    def run(self, state: TradingState):
        # Extract historical data from traderData
        if state.traderData is not None and len(state.traderData):
            try:
                binary_data = base64.b64decode(state.traderData)
                buffer = io.BytesIO(binary_data)
                self.historical_data = cPickle.load(buffer)
            except Exception as e:
                logger.print("Error loading traderData:", str(e))

		# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            # Extract order depth for the product
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Update historical data
            self.update_historical_data(product, order_depth)
            
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
                            # logger.print("BUY RAINFOREST_RESIN", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            #logger.print(self.positions[product], state.position.get(product,0)) #TESTING IF POSITION ARE UPDATED IMMEDIATELY, TRADES EXECUTING IMMEDIATELY ::: CONFIRMED STATE.POSITIONS UPDATED
                            bought_amount += buy_amount

                            if buy_amount == -best_ask_amount:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]

                        if int(best_ask) == acceptable_price and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            # logger.print("BUY RAINFOREST_RESIN", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                    if self.positions[product] > -self.position_limits[product]:
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limits[product], best_bid_amount)
                            # logger.print("SELL RAINFOREST_RESIN", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount

                            if sell_amount == best_bid_amount:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]

                        if int(best_bid) == acceptable_price and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            # logger.print("SELL RAINFOREST_RESIN", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                
                left_to_sell = sold_amount - sell_limit #positive number
                left_to_buy = buy_limit - bought_amount 
                
                
                # Calculate current spread
                """current_spread = best_ask - best_bid

                # Set minimum spread to maintain
                min_spread = 2  # Based on observed minimum spread in the market

                # Determine how much to improve prices based on current spread
                if current_spread <= min_spread + 2:  # If spread is tight (≤4)
                    bid_improvement = 0  # Don't improve the bid
                    ask_improvement = 0  # Don't improve the ask
                else:  
                    bid_improvement = 1  # Standard improvement
                    ask_improvement = 1  # Standard improvement

                # Calculate base prices with dynamic improvements
                base_buy_price = best_bid + bid_improvement
                base_sell_price = best_ask - ask_improvement

                # Calculate position ratio (between -1 and 1)
                inventory_ratio = self.positions[product] / self.position_limits[product]

                # Mild position-based adjustments due to strong mean reversion
                # Only adjust by at most 1 additional tick when approaching limits
                adjustment_factor = 0.2  # Very mild adjustments due to strong mean reversion

                if inventory_ratio > 0:
                    # When long, be slightly more aggressive selling
                    sell_adjustment = round(inventory_ratio * adjustment_factor * current_spread)
                    # Ensure we maintain minimum spread and don't cross market
                    sell_price = max(base_buy_price + min_spread, base_sell_price - sell_adjustment)
                else:
                    sell_price = base_sell_price

                if inventory_ratio < 0:
                    # When short, be slightly more aggressive buying
                    buy_adjustment = round(abs(inventory_ratio) * adjustment_factor * current_spread)
                    # Ensure we maintain minimum spread and don't cross market
                    buy_price = min(base_sell_price - min_spread, base_buy_price + buy_adjustment)
                else:
                    buy_price = base_buy_price
                    
                orders.append(Order(product, buy_price, left_to_buy))
                orders.append(Order(product, sell_price, -left_to_sell))"""
                
                # logger.print("MAKING RAINFOREST_RESIN BUY", str(left_to_buy) + "x", best_bid+1)
                orders.append(Order(product,best_bid+1,left_to_buy))
                # logger.print("MAKING RAINFOREST_RESIN SELL", str(left_to_sell) + "x", best_ask-1)
                orders.append(Order(product,best_ask-1,-left_to_sell))
                            
                result[product] = orders
            
            elif product == "KELP":
                fair_value = self.calculate_kelp_price(order_depth)
                logger.print("Fair value: " + str(fair_value))

                if fair_value is not None:
                    # Initialize tracking variables
                    position = state.position.get(product,0)
                    position_limit = self.position_limits[product]
                    buy_limit = position_limit - position
                    sell_limit = -position_limit - position  # negative limit
                    bought_amount, sold_amount = 0, 0
                    left_to_buy = buy_limit - bought_amount
                    left_to_sell = sold_amount - sell_limit
                    
                    # Market taking - sell side (buying)
                    if len(order_depth.sell_orders) != 0:
                        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                        
                        # Check if we can buy at favorable price
                        if position < self.position_limits[product] and best_ask < fair_value:
                            buy_amount = min(position_limit - position, -best_ask_amount)
                            orders.append(Order(product, best_ask, buy_amount))
                            logger.print("BUY KELP", str(buy_amount) + "x", best_ask)
                            position += buy_amount
                            bought_amount += buy_amount

                            if buy_amount == -best_ask_amount and len(order_depth.sell_orders) > 1:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]    
                    
                    # Market taking - buy side (selling)
                    if len(order_depth.buy_orders) != 0:
                        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                        
                        # Check if we can sell at favorable price
                        if position > -self.position_limits[product] and best_bid > fair_value:
                            sell_amount = min(position + position_limit, best_bid_amount)
                            logger.print("SELL KELP", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product, best_bid, -sell_amount))
                            position -= sell_amount
                            sold_amount -= sell_amount

                            if sell_amount == best_bid_amount and len(order_depth.buy_orders) > 1:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                    
                    # # Market making
                    if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                        if best_ask - 1 > best_bid + 1:
                            # Calculate remaining capacity
                            left_to_buy = buy_limit - bought_amount
                            left_to_sell = sold_amount - sell_limit
                            
                            # Make market on both sides
                            if left_to_buy > 0:
                                orders.append(Order(product, best_bid + 1, left_to_buy))
                                logger.print("MAKING KELP BUY", str(left_to_buy) + "x", best_bid+1)

                            if left_to_sell > 0:
                                orders.append(Order(product, best_ask - 1, -left_to_sell))
                                logger.print("MAKING KELP SELL", str(left_to_sell) + "x", best_ask-1)

                    # Add orders to result
                    result[product] = orders
            elif product == "SQUID_INK":
                price_data = self.historical_data.get(product, None)
                # fair_value = self.calculate_kelp_price(order_depth)
                # if fair_value is not None:
                if price_data is not None:
                    # Initialize tracking variables
                    position = state.position.get(product,0)
                    position_limit = self.position_limits[product]
                    buy_limit = position_limit - position
                    sell_limit = -position_limit - position  # negative limit
                    bought_amount, sold_amount = 0, 0
                    left_to_buy = buy_limit - bought_amount
                    left_to_sell = sold_amount - sell_limit
                    
                    indicator, amount_to_trade = self.bollinger_band(price_data['mid_price'], order_depth)
                    if indicator == "BUY":
                        if position < self.position_limits[product]:
                            best_ask_price, best_ask_amount = list(order_depth.sell_orders.items())[0] 
                            orders.append(Order(product, best_ask_price, best_ask_amount))
                            logger.print("BUY SQUID_INK", str(amount_to_trade) + "x", price_data['mid_price'].iloc[-1])
                            position += amount_to_trade
                            bought_amount += amount_to_trade
                    elif indicator == "SELL":
                        if position > -self.position_limits[product]:
                            best_bid_price, best_bid_amount = list(order_depth.buy_orders.items())[0]
                            orders.append(Order(product, best_bid_price, -best_bid_amount))
                            logger.print("SELL SQUID_INK", str(amount_to_trade) + "x", price_data['mid_price'].iloc[-1])
                            position -= amount_to_trade
                            sold_amount -= amount_to_trade

                    # Add orders to result
                    result[product] = orders
                    
            """acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders"""
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.

        buffer = io.BytesIO()
        cPickle.dump(self.historical_data, buffer)
        binary_data = buffer.getvalue()
        traderData = base64.b64encode(binary_data).decode('ascii')
        # traderData = "SAMPLE"
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    