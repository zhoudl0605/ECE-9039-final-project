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
def pure_pmm(
    hbt,                    # 回测引擎对象
    stat,                   # 统计记录器
    half_spread,            # 半价差（tick数）
    skew,                   # 偏度系数，控制仓位对报价的影响
    interval,               # 策略执行间隔（纳秒）
    order_qty_dollar,       # 单笔订单金额（美元）
    max_position_dollar,    # 最大持仓金额（美元）
    grid_num,               # 网格订单层数
    grid_interval,          # 网格间隔（价格单位）
):
    """
    纯PMM做市策略 - 去除OBI影响的版本

    核心思想：
    1. 基于中间价计算基础价格
    2. 根据持仓风险调整保留价格
    3. 在保留价格±半价差位置下单做市
    4. 不考虑订单簿失衡信号

    返回值：
    - tick_count: 总tick数
    """
    asset_no = 0

    # 获取交易参数
    tick_size = hbt.depth(0).tick_size    # 最小价格变动单位
    lot_size = hbt.depth(0).lot_size      # 最小交易数量单位

    t = 0  # 时间索引计数器

    # 主循环：按指定间隔执行策略
    while hbt.elapse(interval) == 0:
        # 清理过期订单
        hbt.clear_inactive_orders(asset_no)

        # 获取当前市场状态
        depth = hbt.depth(asset_no)        # 订单簿深度数据
        position = hbt.position(asset_no)   # 当前持仓
        orders = hbt.orders(asset_no)       # 当前活跃订单

        # 获取最优买卖价
        best_bid = depth.best_bid
        best_ask = depth.best_ask

        if best_bid <= 0 or best_ask <= 0:
            continue

        # 计算中间价
        mid_price = (best_bid + best_ask) / 2.0

        # === 计算买卖价格（不使用OBI） ===

        # 根据中间价和订单金额计算订单数量
        order_qty = max(round((order_qty_dollar / mid_price) /
                        lot_size) * lot_size, lot_size)
        # order_qty = 0.1

        # 公允价格直接使用中间价（不加OBI调整）
        fair_price = mid_price

        # 计算标准化持仓（以订单数量为单位）
        normalized_position = position / order_qty

        # 计算保留价格（仅考虑持仓风险的调整）
        reservation_price = fair_price - skew * normalized_position
        # reservation_price = mid_price - skew * tick_size * position

        # 计算买卖报价（half_spread是tick数），直接计算对齐后的价格
        reservation_price_tick = reservation_price / tick_size
        bid_price = np.floor(reservation_price_tick - half_spread) * tick_size
        ask_price = np.ceil(reservation_price_tick + half_spread) * tick_size

        # 确保价格不超过最优买卖价
        bid_price = min(bid_price, best_bid)
        ask_price = max(ask_price, best_ask)

        # === 更新报价网格 ===

        # 创建新的买单网格
        new_bid_orders = Dict.empty(np.uint64, np.float64)
        # 检查仓位限制和价格有效性
        if position * mid_price < max_position_dollar and np.isfinite(bid_price):
            for i in range(grid_num):
                bid_price_tick = round(bid_price / tick_size)
                # 使用价格tick作为订单ID
                new_bid_orders[uint64(bid_price_tick)] = bid_price
                bid_price -= grid_interval

        # 创建新的卖单网格
        new_ask_orders = Dict.empty(np.uint64, np.float64)
        # 检查仓位限制和价格有效性
        if position * mid_price > -max_position_dollar and np.isfinite(ask_price):
            for i in range(grid_num):
                ask_price_tick = round(ask_price / tick_size)
                # 使用价格tick作为订单ID
                new_ask_orders[uint64(ask_price_tick)] = ask_price
                ask_price += grid_interval

        # 撤销不在新网格中的现有订单
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id not in new_bid_orders)
                    or (order.side == SELL and order.order_id not in new_ask_orders)
                ):
                    hbt.cancel(asset_no, order.order_id, False)

        # 提交新的买单
        for order_id, order_price in new_bid_orders.items():
            # 只在该价格位置没有现有订单时提交新订单
            if order_id not in orders:
                hbt.submit_buy_order(asset_no, order_id,
                                     order_price, order_qty, GTX, LIMIT, False)

        # 提交新的卖单
        for order_id, order_price in new_ask_orders.items():
            # 只在该价格位置没有现有订单时提交新订单
            if order_id not in orders:
                hbt.submit_sell_order(
                    asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        t += 1

        # 记录当前状态用于统计分析
        stat.record(hbt)

    # return t
