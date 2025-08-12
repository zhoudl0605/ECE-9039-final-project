import numpy as np

from numba import njit, uint64
from numba.typed import Dict

from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    GTX,
    LIMIT,
    BUY,
    SELL,
    BUY_EVENT,
    SELL_EVENT,
    Recorder
)
from hftbacktest.stats import LinearAssetRecord


@njit
def pure_pmm_step(
    hbt,                    # Backtest engine object
    stat,                   # Statistics recorder
    half_spread,            # Half spread (in ticks)
    skew,                   # Skew coefficient (in ticks), ticks offset per normalized position
    interval,               # Strategy execution interval (nanoseconds)
    order_qty_dollar,       # Order amount in dollars
    max_position_dollar,    # Maximum position in dollars
    grid_num,               # Number of grid levels
    grid_interval,          # Grid interval (price units)
):
    """
    Pure PMM market making strategy - single step execution version

    Returns:
    - data_finished: Boolean indicating whether data has been exhausted
    """
    asset_no = 0

    # Try to advance time
    if hbt.elapse(interval) != 0:
        # Data exhausted
        return True

    # Get trading parameters
    tick_size = hbt.depth(0).tick_size    # Minimum price increment
    lot_size = hbt.depth(0).lot_size      # Minimum trade size

    half_spread = half_spread * tick_size
    grid_interval = grid_interval * tick_size

    # Clear expired orders
    hbt.clear_inactive_orders(asset_no)

    # Get current market state
    depth = hbt.depth(asset_no)        # Order book depth data
    position = hbt.position(asset_no)   # Current position
    orders = hbt.orders(asset_no)       # Current active orders

    # Get best bid/ask prices
    best_bid = depth.best_bid
    best_ask = depth.best_ask

    if best_bid <= 0 or best_ask <= 0:
        return False

    # Calculate mid price
    mid_price = (best_bid + best_ask) / 2.0

    # Calculate order quantity based on mid price and order amount
    order_qty = max(round((order_qty_dollar / mid_price) /
                    lot_size) * lot_size, lot_size)

    # Fair price using mid price
    fair_price = mid_price

    # Calculate normalized position (in order quantity units)
    normalized_position = position / order_qty

    # Calculate reservation price
    reservation_price = fair_price - (skew * tick_size) * normalized_position

    bid_price = min(reservation_price - half_spread, best_bid)
    ask_price = max(reservation_price + half_spread, best_ask)

    bid_price = np.floor(bid_price / grid_interval) * grid_interval
    ask_price = np.ceil(ask_price / grid_interval) * grid_interval

    # === Update quote grid ===

    # Create new bid order grid
    new_bid_orders = Dict.empty(np.uint64, np.float64)
    # Check position limits and price validity
    if position * mid_price < max_position_dollar and np.isfinite(bid_price):
        for i in range(grid_num):
            bid_price_tick = int(bid_price / tick_size)
            new_bid_orders[uint64(bid_price_tick)] = bid_price
            bid_price -= grid_interval

    # Create new ask order grid
    new_ask_orders = Dict.empty(np.uint64, np.float64)
    # Check position limits and price validity
    if position * mid_price > -max_position_dollar and np.isfinite(ask_price):
        for i in range(grid_num):
            ask_price_tick = int(ask_price / tick_size)
            new_ask_orders[uint64(ask_price_tick)] = ask_price
            ask_price += grid_interval

    # Cancel existing orders not in new grid
    order_values = orders.values()
    while order_values.has_next():
        order = order_values.get()
        if order.cancellable:
            if (
                (order.side == BUY and order.order_id not in new_bid_orders)
                or (order.side == SELL and order.order_id not in new_ask_orders)
            ):
                hbt.cancel(asset_no, order.order_id, False)

    # Submit new buy orders
    for order_id, order_price in new_bid_orders.items():
        if order_id not in orders:
            hbt.submit_buy_order(asset_no, order_id,
                                 order_price, order_qty, GTX, LIMIT, False)

    # Submit new sell orders
    for order_id, order_price in new_ask_orders.items():
        if order_id not in orders:
            hbt.submit_sell_order(
                asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

    # Record current state for statistical analysis
    stat.record(hbt)

    return False


@njit
def pure_pmm(
    hbt,                    # Backtest engine object
    stat,                   # Statistics recorder
    half_spread,            # Half spread (in ticks)
    skew,                   # Skew coefficient (in ticks), ticks offset per normalized position
    interval,               # Strategy execution interval (nanoseconds)
    order_qty_dollar,       # Order amount in dollars
    max_position_dollar,    # Maximum position in dollars
    grid_num,               # Number of grid levels
    grid_interval,          # Grid interval (price units)
):
    """
    Pure PMM market making strategy

    Core concept:
    1. Calculate base price based on mid price
    2. Adjust reservation price based on position risk
    3. Place market making orders at reservation price ± half spread
    """
    while True:
        data_finished = pure_pmm_step(
            hbt=hbt,
            stat=stat,
            half_spread=half_spread,
            skew=skew,
            interval=interval,
            order_qty_dollar=order_qty_dollar,
            max_position_dollar=max_position_dollar,
            grid_num=grid_num,
            grid_interval=grid_interval,
        )

        if data_finished:
            break
