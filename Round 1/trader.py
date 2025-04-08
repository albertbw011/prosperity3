from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}
    positions = {"RAINFOREST_RESIN": 0, "KELP": 0, "SQUID_INK": 0}
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

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
                            print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                            if buy_amount == -best_ask_amount:
                                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[1]
                        if int(best_ask) == acceptable_price and self.positions[product] < 0:
                            buy_amount = min(-self.positions[product],-best_ask_amount)
                            print("BUY", str(buy_amount) + "x", best_ask)
                            orders.append(Order(product,best_ask,buy_amount))
                            self.positions[product] += buy_amount
                            bought_amount += buy_amount
                    
                
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.positions[product] > -self.position_limits[product]:
                        if int(best_bid) > acceptable_price:
                            sell_amount = min(self.positions[product] + self.position_limits[product], best_bid_amount)
                            print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                            if sell_amount == best_bid_amount:
                                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[1]
                        if int(best_bid) == acceptable_price and self.positions[product] > 0:
                            sell_amount = min(self.positions[product],best_bid_amount)
                            print("SELL", str(sell_amount) + "x", best_bid)
                            orders.append(Order(product,best_bid,-sell_amount))
                            self.positions[product] -= sell_amount
                            sold_amount -= sell_amount
                
                left_to_sell = sold_amount - sell_limit #positive number
                left_to_buy = buy_limit - bought_amount 

                print("MAKING BUY", str(left_to_buy) + "x", best_bid+1)
                orders.append(Order(product,best_bid+1,left_to_buy))
                print("MAKING SELL", str(left_to_sell) + "x", best_ask-1)
                orders.append(Order(product,best_ask-1,-left_to_sell))
                    

                            
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
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
    
print("Hello")