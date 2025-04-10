import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any

import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# Product Aliases
RAINFOREST_RESIN: Symbol = "RAINFOREST_RESIN"
KELP: Symbol = "KELP"
SQUID_INK: Symbol = "SQUID_INK"

# Product Limits
PRODUCT_LIMITS = {
    RAINFOREST_RESIN: 50,
    KELP: 50,
    SQUID_INK: 50
}


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
                    self.compress_state(state, self.truncate(
                        state.traderData, max_item_length)),
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
            compressed.append(
                [listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

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

# Helper functions
def get_best_ask(state: TradingState, symbol: Symbol):
    order_depth = state.order_depths.get(symbol)

    if order_depth is None or len(order_depth.sell_orders) == 0:
        return None, None

    return list(order_depth.sell_orders.items())[0]


def get_best_bid(state: TradingState, symbol: Symbol):
    order_depth = state.order_depths.get(symbol)

    if order_depth is None or len(order_depth.buy_orders) == 0:
        return None, None

    return list(order_depth.buy_orders.items())[0]


def get_mid_price(state: TradingState, symbol: Symbol):
    best_ask, _ = get_best_ask(state, symbol)
    best_bid, _ = get_best_bid(state, symbol)

    if best_ask and best_bid:
        return (best_ask + best_bid) // 2

    return None

class Orders:
    def __init__(self, state: TradingState) -> None:
        self._orders: dict[Symbol, list[Order]] = {}
        self._state = state

    def remaining_quantity(self, symbol: Symbol) -> int:
        """The quantity of units of a symbol between the current orders and the position limit"""
        remaining = PRODUCT_LIMITS[symbol] - self.get_current_quantity(symbol)

        return remaining if remaining > 0 else 0


    def get_orders(self) -> dict[Symbol, list[Order]]:
        return self._orders

    def get_current_quantity(self, symbol: Symbol) -> int:
        return self._state.position.get(
            symbol, 0) + sum(order.quantity for order in self._orders.get(symbol, []))

    def add_order(self, symbol: Symbol, price: int, quantity: int) -> None:
        if quantity == 0:
            logger.print(f"ERROR: No quantity provided.")
            # TODO raise error
            return

        if price < 0:
            logger.print(f"ERROR: invalid price {price} provided.")
            return

        BUY_SELL = "BUY" if quantity > 0 else "SELL"

        # We want to prevent the orders being sent exceeding the position limit.
        limit = PRODUCT_LIMITS[symbol]

        next_position = self.get_current_quantity(symbol) + quantity

        if -limit > next_position or next_position > limit:
            # TODO raise error
            logger.print(
                f"ERROR: Position Limit will be exceeded if the current order is added in addition to orders in queue.")
            return

        logger.print(f"Added order {BUY_SELL} {quantity} at {price}")
        order = Order(symbol, price, quantity)
        self._orders.setdefault(symbol, []).append(order)


class Strategy(ABC):
    def __init__(self, state: TradingState, orders: Orders) -> None:
        self._state = state
        self._orders = orders

    def run(self):
        self._run()

    @abstractmethod
    def _run(self):
        raise NotImplementedError()

class DummyStrategy(Strategy):
    def _run(self):
        for product in self._state.order_depths:
            acceptable_price = 100  # Participant should calculate this value
            logger.print("Acceptable price : " + str(acceptable_price))
            best_ask, best_ask_amount=get_best_ask(self._state, product)

            if best_ask_amount is not None:
                if int(best_ask) < acceptable_price:
                    self._orders.add_order(product, best_ask, -best_ask)

            best_bid, best_bid_amount=get_best_bid(self._state, product)

            if best_bid_amount is not None:
                if int(best_bid) > acceptable_price:
                    self._orders.add_order(product, best_bid, -best_bid_amount)


class AcceptablePriceStrategy(Strategy):
    """
    Buys and sell product based on an acceptable price.
    """

    def __init__(self,
                 state: TradingState,
                 orders: Orders, product: Symbol,
                 acceptable_bid_price: int,
                 acceptable_ask_price: int,
                 best_only: bool = True) -> None:

        super().__init__(state, orders)
        self._product = product
        self._acceptable_bid_price = acceptable_bid_price
        self._acceptable_ask_price = acceptable_ask_price
        self._best_only = best_only

    def _run(self):
        order_depth = self._state.order_depths.get(self._product)

        if order_depth is None:
            return

        if len(order_depth.sell_orders) != 0:
            for ask, amount in order_depth.sell_orders.items():
                if int(ask) < self._acceptable_ask_price:
                    self._orders.add_order(self._product, ask, -amount)

                if self._best_only:
                    break

        if len(order_depth.buy_orders) != 0:
            for bid, amount in order_depth.buy_orders.items():
                if int(bid) > self._acceptable_bid_price:
                    self._orders.add_order(self._product, bid, -amount)

                if self._best_only:
                    break

def _ema(price_history: list[float], span: int) -> int:
    """
    Method to calculate exponential moving average.

    :param span: the number of periods for the ema.
    """
    data_series = pd.Series(price_history[-span:])
    return int(data_series.ewm(span=span, adjust=False).mean().tail(1))


def ema(state, trader_data, product: Symbol, span: int) -> int | None:
    key = f"{product}_PRICES"

    price = _ema(trader_data[key], span) if trader_data.get(key) else None

    mid_price = get_mid_price(state, product)
    prices = trader_data.setdefault(key, [])

    if len(prices) > span:
        prices.pop(0)

    prices.append(mid_price)

    return price


class AcceptablePriceWithEmaStrategy(Strategy):
    def __init__(self, state: TradingState, orders: Orders, trader_data: dict, product: Symbol, span: int, best_only: bool = True) -> None:
        super().__init__(state, orders)
        self._trading_data = trader_data
        self._product = product
        self._span = span
        self._best_only = best_only

    def _run(self):
        price = ema(self._state, self._trading_data, self._product, self._span)
        if price is None:
            return

        s = AcceptablePriceStrategy(state=self._state, orders=self._orders, product=self._product,
                                    acceptable_ask_price=price, acceptable_bid_price=price, best_only=self._best_only)
        s.run()


def _sma(price_history: list[float], span: int) -> int:
    """
    Method to calculate exponential moving average.

    :param span: the number of periods for the ema.
    """
    data_series = pd.Series(price_history[-span:])
    return int(data_series.rolling(window=span, min_periods=1).mean().tail(1))


def sma(state, trader_data, product: Symbol, span: int) -> int | None:
    key = f"{product}_PRICES"

    price = _sma(trader_data[key], span) if trader_data.get(key) else None

    mid_price = get_mid_price(state, product)
    prices = trader_data.setdefault(key, [])

    if len(prices) > span:
        prices.pop(0)

    prices.append(mid_price)

    return price



class AcceptablePriceWithSmaStrategy(Strategy):
    def __init__(self, state: TradingState, orders: Orders, trader_data: dict, product: Symbol, span: int, best_only: bool = True) -> None:
        super().__init__(state, orders)
        self._trading_data = trader_data
        self._product = product
        self._span = span
        self._best_only = best_only

    def _run(self):
        price = sma(self._state, self._trading_data, self._product, self._span)
        if price is None:
            return

        s = AcceptablePriceStrategy(state=self._state, orders=self._orders, product=self._product,
                                    acceptable_ask_price=price, acceptable_bid_price=price, best_only=self._best_only)
        s.run()

class MarketMaking(Strategy):
    def __init__(self, state: TradingState, orders: Orders, symbol: Symbol, spread: int, quantity: int):
        super().__init__(state, orders)
        self._symbol = symbol
        self._spread = spread

        if quantity < 0:
            quantity *= -1

        self._quantity = quantity


    def _get_prices(self) -> [int, int]:
        mid_price = get_mid_price(self._state, self._symbol)

        if self._spread > mid_price:
            return None, None

        if mid_price:
            return (mid_price - self._spread) // 2, (mid_price + self._spread) // 2

        return None, None

    def _run(self):
        bid_price, ask_price = self._get_prices()

        if bid_price is None and ask_price is None:
            return

        self._orders.add_order(self._symbol, bid_price, self._quantity)
        self._orders.add_order(self._symbol, ask_price, -self._quantity)



class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}

        orders = Orders(state)

        # Run strategies
        # DummyStrategy(state, orders).run()
        AcceptablePriceStrategy(state, orders, RAINFOREST_RESIN, 10_000, 10_000).run()
        AcceptablePriceWithEmaStrategy(state, orders, trader_data, KELP, 5).run()
        # AcceptablePriceWithEmaStrategy(state, orders, trader_data, SQUID_INK, 5).run()

        MarketMaking(state, orders, SQUID_INK, 10, 10).run()
        # MarketMaking(state, orders, SQUID_INK, 10, PRODUCT_LIMITS[SQUID_INK]).run()

        trader_data = jsonpickle.encode(trader_data)

        conversions = 1
        result = orders.get_orders()
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
