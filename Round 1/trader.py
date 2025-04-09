from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

class Trader:
    def __init__(self):
        # Position limits for each product
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
        
        # Product-specific parameters
        self.params = {
            "RAINFOREST_RESIN": {
                "acceptable_price": 10000,  # Fixed threshold for RESIN
                "make_market": True         # Should make market for this product
            },
            "KELP": {
                "adverse_volume": 15,       # Volume threshold for identifying large orders
                "make_market": True         # Should make market for this product
            },
            "SQUID_INK": {
                "adverse_volume": 15,       # Volume threshold for INK may differ
                "make_market": True         # Should make market for this product
            }
        }
    
    def calculate_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        """Calculate fair value based on product-specific logic"""
        
        # Need both buy and sell orders to calculate fair value
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        # Get best bid and ask
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        if product == "RAINFOREST_RESIN":
            # For RESIN we use a fixed acceptable price
            return self.params[product]["acceptable_price"]
        else:
            # For KELP and SQUID_INK we use adverse volume approach
            adverse_volume = self.params[product]["adverse_volume"]
            
            # Filter large orders that might indicate informed trading
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= adverse_volume
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= adverse_volume
            ]
            
            # Find best prices among large orders
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            
            # Calculate fair value
            if mm_ask is None or mm_bid is None:
                return (best_ask + best_bid) / 2
            else:
                return (mm_ask + mm_bid) / 2
    
    def run(self, state: TradingState):
        # Orders to be placed on exchange matching engine
        result = {}
        
        # Process each product
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            # Skip if no orders on either side
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue
            
            # Get current position
            position = state.position.get(product, 0)
            
            # Calculate fair value
            fair_value = self.calculate_fair_value(product, order_depth)
            
            if fair_value is None:
                result[product] = orders
                continue
            
            # Get best bid and ask
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = order_depth.sell_orders[best_ask]
            
            # Calculate position limits
            buy_limit = self.position_limits[product] - position
            sell_limit = self.position_limits[product] + position
            bought_amount = 0
            sold_amount = 0
            
            # Market taking: Buying at favorable prices
            if position < self.position_limits[product] and best_ask < fair_value:
                buy_amount = min(buy_limit, -best_ask_amount)
                if buy_amount > 0:
                    orders.append(Order(product, best_ask, buy_amount))
                    position += buy_amount
                    bought_amount += buy_amount
            
            # Market taking: Selling at favorable prices
            if position > -self.position_limits[product] and best_bid > fair_value:
                sell_amount = min(sell_limit, best_bid_amount)
                if sell_amount > 0:
                    orders.append(Order(product, best_bid, -sell_amount))
                    position -= sell_amount
                    sold_amount += sell_amount
            
            # Market making: Place orders if there's still capacity and sufficient spread
            if self.params[product]["make_market"] and best_ask - 1 > best_bid + 1:
                left_to_buy = buy_limit - bought_amount
                left_to_sell = sell_limit - sold_amount
                
                # Only place orders if we have capacity
                if left_to_buy > 0:
                    orders.append(Order(product, best_bid + 1, left_to_buy))
                
                if left_to_sell > 0:
                    orders.append(Order(product, best_ask - 1, -left_to_sell))
            
            # Add orders to result
            result[product] = orders
        
        # Return results with no conversions
        traderData = ""
        conversions = 0
        return result, conversions, traderData