import numpy as np
import torch
from typing import Optional, Dict
import warnings

from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
from tensordict import TensorDict, TensorDictBase

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from .pmm import PMMStrategy


class PMMRLEnv(EnvBase):
    """
    基于PMM策略的TorchRL强化学习环境
    适用于SAC等连续控制算法
    """

    def __init__(
        self,
        data_asset: BacktestAsset,
        initial_balance: float = 1000.0,
        max_steps: int = 200,
        step_interval_ns: int = 500_000_000,
        device: torch.device = torch.device("cpu"),
        risk_penalty_weight: float = 0.05,
        transaction_cost_rate: float = 0.0005,
        order_amount_range: tuple = (1, 20),
        reward_normalization: bool = True,
        observation_normalization: bool = True,
    ):
        """
        初始化PMM强化学习环境

        Args:
            data_asset: hftbacktest数据资产
            initial_balance: 初始余额
            max_steps: 最大步数
            step_interval_ns: 每步时间间隔（纳秒）
            device: 计算设备
            risk_penalty_weight: 风险惩罚权重
            transaction_cost_rate: 交易成本率
            order_amount_range: 订单张数范围 (min, max)

        动作空间说明 (9维连续动作):
            1. bid_spread (0.05%-0.25%): 买入价差，负费率优化
            2. ask_spread (0.05%-0.25%): 卖出价差，负费率优化
            3. order_refresh_time (0.5-3.0): 订单刷新时间（秒）
            4. price_deviation_pct (0.1%-0.5%): 价格偏差百分比
            5. hang_order_time_limit (5.0-30.0): 挂单时间限制（秒）
            6. max_open_orders (20-200): 最大挂单数量，负费率市场做市
            7. take_profit_pct (0.1%-1%): 止盈百分比
            8. stop_loss_pct (0.2%-2%): 止损百分比
            9. order_amount: 每个订单的合约张数
        """
        super().__init__(device=device)

        self.data_asset = data_asset
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.step_interval_ns = step_interval_ns
        self.risk_penalty_weight = risk_penalty_weight
        self.transaction_cost_rate = transaction_cost_rate
        self.order_amount_range = order_amount_range
        self.reward_normalization = reward_normalization
        self.observation_normalization = observation_normalization

        # 环境状态
        self.current_step = 0
        self.hbt = None
        self.pmm_strategy = None
        self.last_balance = initial_balance
        self.last_position = 0.0
        self.price_history = []
        self.max_price_history = 50

        # 性能优化相关状态
        self.reward_history = []
        self.step_count = 0
        self.episode_count = 0

        # 定义动作空间 - 扩展为9维连续动作空间
        # 动作维度：[bid_spread, ask_spread, order_refresh_time, price_deviation_pct,
        #          hang_order_time_limit, max_open_orders, take_profit_pct, stop_loss_pct, order_amount]
        self.action_spec = CompositeSpec(
            action=BoundedTensorSpec(
                low=torch.tensor([
                    0.0005,  # bid_spread: 0.05%
                    0.0005,  # ask_spread: 0.05%
                    0.5,     # order_refresh_time: 0.5秒
                    0.001,   # price_deviation_pct: 0.1%
                    5.0,     # hang_order_time_limit: 5秒
                    20.0,    # max_open_orders: 20个（负费率市场做市）
                    0.001,   # take_profit_pct: 0.1%
                    0.002,   # stop_loss_pct: 0.2%
                    float(self.order_amount_range[0]),
                ], device=self.device),
                high=torch.tensor([
                    0.0025,  # bid_spread: 0.25%（负费率优化）
                    0.0025,  # ask_spread: 0.25%（负费率优化）
                    3.0,     # order_refresh_time: 3秒
                    0.005,   # price_deviation_pct: 0.5%
                    30.0,    # hang_order_time_limit: 30秒
                    200.0,   # max_open_orders: 200个（负费率市场做市）
                    0.01,    # take_profit_pct: 1%
                    0.02,    # stop_loss_pct: 2%
                    float(self.order_amount_range[1]),
                ], device=self.device),
                shape=torch.Size([9]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # 定义观测空间 - 扩展为10维状态向量
        # 观测维度：[mid_price, spread, position, balance_ratio, volatility,
        #          order_book_imbalance, active_orders_ratio, recent_pnl, avg_order_fill_time, strategy_efficiency]
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=torch.Size([10]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # 奖励规格
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

        # 完成标志规格
        self.done_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=torch.Size([1]),
            dtype=torch.bool,
            device=self.device,
        )

    def _initialize_backtest(self):
        """初始化回测环境"""
        self.hbt = HashMapMarketDepthBacktest([self.data_asset])

        # 注意：hftbacktest可能没有直接的set_balance方法
        # 这里使用更保守的初始化方式
        try:
            # 尝试各种可能的初始化方法
            if hasattr(self.hbt, 'set_balance'):
                self.hbt.set_balance(self.initial_balance)  # type: ignore
            elif hasattr(self.hbt, 'balance') and callable(getattr(self.hbt, 'balance')):
                # 如果有balance方法，尝试调用
                pass
        except Exception as e:
            warnings.warn(f"Balance initialization may not be available: {e}")

    def _get_market_state(self) -> Dict[str, float]:
        """获取市场状态"""
        if self.hbt is None:
            return {
                'mid_price': 0.0,
                'spread': 0.0,
                'best_bid': 0.0,
                'best_ask': 0.0,
                'volatility': 0.0,
                'order_book_imbalance': 0.0,
            }

        try:
            depth = self.hbt.depth(0)
            best_bid = float(depth.best_bid) if depth.best_bid > 0 else 0.0
            best_ask = float(depth.best_ask) if depth.best_ask > 0 else 0.0

            # 计算中间价和价差
            mid_price = (best_bid + best_ask) / \
                2.0 if best_bid > 0 and best_ask > 0 else 0.0
            spread = (best_ask - best_bid) / \
                mid_price if mid_price > 0 else 0.0

            # 更新价格历史
            if mid_price > 0:
                self.price_history.append(mid_price)
                if len(self.price_history) > self.max_price_history:
                    self.price_history.pop(0)

            # 计算波动率
            volatility = 0.0
            if len(self.price_history) >= 2:
                returns = np.diff(self.price_history) / self.price_history[:-1]
                volatility = float(np.std(returns)) if len(
                    returns) > 1 else 0.0

            # 计算订单簿不平衡度（简化版本）
            order_book_imbalance = 0.0
            if best_bid > 0 and best_ask > 0:
                # 这里可以扩展为更复杂的订单簿分析
                order_book_imbalance = (
                    best_ask - best_bid) / (best_ask + best_bid)

            return {
                'mid_price': mid_price,
                'spread': spread,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'volatility': volatility,
                'order_book_imbalance': order_book_imbalance,
            }
        except Exception as e:
            warnings.warn(f"Failed to get market state: {e}")
            return {
                'mid_price': 0.0,
                'spread': 0.0,
                'best_bid': 0.0,
                'best_ask': 0.0,
                'volatility': 0.0,
                'order_book_imbalance': 0.0,
            }

    def _get_strategy_state(self) -> Dict[str, float]:
        """获取策略状态"""
        if self.hbt is None or self.pmm_strategy is None:
            return {
                'position': 0.0,
                'balance': self.initial_balance,
                'active_orders_ratio': 0.0,
            }

        try:
            # 获取当前仓位
            position = 0.0
            if hasattr(self.hbt, 'position'):
                position = self.hbt.position(0)

            # 获取当前余额（使用正确的state_values方法）
            balance = self.initial_balance
            try:
                balance = self.hbt.state_values(0).balance
            except (AttributeError, TypeError):
                # 如果state_values方法不可用，使用默认值
                pass

            # 获取活跃订单比例
            active_orders_ratio = 0.0
            if self.pmm_strategy is not None and hasattr(self.pmm_strategy, 'executors'):
                active_orders = len(self.pmm_strategy.executors)
                max_orders = self.pmm_strategy.max_open_orders
                active_orders_ratio = active_orders / max_orders if max_orders > 0 else 0.0

            return {
                'position': position,
                'balance': balance,
                'active_orders_ratio': active_orders_ratio,
            }
        except Exception as e:
            warnings.warn(f"Failed to get strategy state: {e}")
            return {
                'position': 0.0,
                'balance': self.initial_balance,
                'active_orders_ratio': 0.0,
            }

    def _get_observation(self) -> torch.Tensor:
        """获取环境观测 - 优化版本"""
        market_state = self._get_market_state()
        strategy_state = self._get_strategy_state()

        # 计算余额比例
        balance_ratio = strategy_state['balance'] / self.initial_balance

        # 计算最近的盈亏
        recent_pnl = strategy_state['balance'] - self.last_balance

        # 🔧 优化：简化平均订单成交时间计算
        avg_order_fill_time = 0.0
        if self.pmm_strategy is not None and hasattr(self.pmm_strategy, 'executors'):
            avg_order_fill_time = min(
                1.0, len(self.pmm_strategy.executors) / 5.0)  # 更敏感的缩放

        # 🔧 优化：改进策略效率计算
        strategy_efficiency = 0.0
        if self.current_step > 0:
            # 基于累积收益率的效率指标
            cumulative_return = (
                strategy_state['balance'] - self.initial_balance) / self.initial_balance
            strategy_efficiency = max(-1.0,
                                      min(1.0, cumulative_return * 10))  # 放大信号

        # 🔧 优化：更好的价格标准化
        price_norm = market_state['mid_price'] / \
            1000.0 if market_state['mid_price'] > 0 else 0.0
        position_norm = strategy_state['position'] / 1000.0  # 更合理的仓位标准化

        # 组合观测向量（10维）
        obs = torch.tensor([
            price_norm,  # 🔧 优化：标准化价格
            market_state['spread'] * 1000,  # 🔧 优化：放大价差信号
            position_norm,  # 🔧 优化：标准化仓位
            balance_ratio,
            market_state['volatility'] * 100,  # 🔧 优化：放大波动率信号
            market_state['order_book_imbalance'] * 10,  # 🔧 优化：放大不平衡信号
            strategy_state['active_orders_ratio'],
            recent_pnl / self.initial_balance * 100,  # 🔧 优化：放大盈亏信号
            avg_order_fill_time,
            strategy_efficiency,
        ], dtype=torch.float32, device=self.device)

        # 🔧 新增：观测标准化
        if self.observation_normalization:
            obs = self._normalize_observation(obs)

        return obs

    def _calculate_reward(self, action: torch.Tensor) -> float:
        """计算奖励函数 - 优化版本"""
        strategy_state = self._get_strategy_state()

        # 🔧 优化：基础收益奖励（更强调短期收益）
        current_balance = strategy_state['balance']
        profit = current_balance - self.last_balance
        profit_reward = profit / self.initial_balance * 10  # 放大收益信号

        # 🔧 优化：改进风险惩罚（考虑仓位变化）
        position = strategy_state['position']
        position_change = abs(position - self.last_position)
        risk_penalty = self.risk_penalty_weight * (
            (position / self.initial_balance) ** 2 +
            position_change / self.initial_balance * 0.1  # 仓位变化惩罚
        )

        # 🔧 优化：更现实的交易成本
        transaction_cost = 0.0
        if self.pmm_strategy is not None and hasattr(self.pmm_strategy, 'executors'):
            active_orders = len(self.pmm_strategy.executors)
            transaction_cost = active_orders * self.transaction_cost_rate * 0.1  # 降低交易成本权重

        # 🔧 优化：简化的动作稳定性奖励
        action_center = torch.tensor([
            0.002,   # 🔧 调整：bid_spread中心值
            0.002,   # 🔧 调整：ask_spread中心值
            1.5,     # 🔧 调整：order_refresh_time中心值
            0.003,   # 🔧 调整：price_deviation_pct中心值
            15.0,    # 🔧 调整：hang_order_time_limit中心值
            25.0,    # 🔧 调整：max_open_orders中心值
            0.005,   # 🔧 调整：take_profit_pct中心值
            0.01,    # 🔧 调整：stop_loss_pct中心值
            float(sum(self.order_amount_range) / 2),  # 动态订单张数中心值
        ], device=action.device)

        # 🔧 优化：更平衡的权重
        action_weights = torch.tensor([
            200.0,   # 🔧 降低：bid_spread权重
            200.0,   # 🔧 降低：ask_spread权重
            0.5,     # 🔧 降低：order_refresh_time权重
            100.0,   # 🔧 降低：price_deviation_pct权重
            0.05,    # 🔧 降低：hang_order_time_limit权重
            0.005,   # 🔧 降低：max_open_orders权重
            200.0,   # 🔧 降低：take_profit_pct权重
            100.0,   # 🔧 降低：stop_loss_pct权重
            0.005,   # 🔧 降低：order_amount权重
        ], device=action.device)

        action_penalty = 0.00005 * torch.sum(  # 🔧 降低惩罚强度
            action_weights * torch.abs(action - action_center)
        ).item()

        # 🔧 优化：改进参数合理性奖励
        param_reward = 0.0
        if action[0] > 0 and action[1] > 0:  # 价差合理
            param_reward += 0.0005  # 增加奖励
        if action[6] > action[7] * 0.5:  # 止盈应该比止损更保守
            param_reward += 0.0005
        if 0.001 <= action[0] <= 0.004 and 0.001 <= action[1] <= 0.004:  # 价差在合理范围
            param_reward += 0.001

        # 🔧 新增：市场适应性奖励
        market_reward = 0.0
        market_state = self._get_market_state()
        if market_state['volatility'] > 0.001:  # 高波动时
            if action[0] > 0.002 and action[1] > 0.002:  # 更大价差
                market_reward += 0.0005
        else:  # 低波动时
            if action[0] < 0.003 and action[1] < 0.003:  # 更小价差
                market_reward += 0.0005

        # 综合奖励
        total_reward = profit_reward - risk_penalty - transaction_cost - \
            action_penalty + param_reward + market_reward

        # 🔧 新增：奖励标准化
        if self.reward_normalization:
            total_reward = self._normalize_reward(total_reward)

        return total_reward

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """重置环境"""
        self.current_step = 0
        self.last_balance = self.initial_balance
        self.last_position = 0.0
        self.price_history = []

        # 重新初始化回测环境
        self._initialize_backtest()

        # 🔧 优化：初始化PMM策略（使用更合理的参数）
        self.pmm_strategy = PMMStrategy(
            hbt=self.hbt,
            asset_no=0,
            bid_spread=0.002,  # 🔧 优化：更合理的初始价差
            ask_spread=0.002,
            max_open_orders=25,  # 🔧 优化：适中的订单数
            order_amount=int(sum(self.order_amount_range) / 2),  # 使用动态范围的中心值
            take_profit_pct=0.005,  # 🔧 优化：更合理的止盈
            stop_loss_pct=0.01,     # 🔧 优化：更合理的止损
            hang_order_time_limit=15.0,  # 🔧 优化：15秒
            order_refresh_time=1.5,      # 🔧 优化：1.5秒
            price_deviation_pct=0.003,  # 🔧 优化：适中的价格偏差
        )

        # 🔧 新增：重置状态计数器
        self.episode_count += 1
        self.step_count = 0

        # 获取初始观测
        observation = self._get_observation()

        # 初始化完成标志
        done = torch.tensor([False], dtype=torch.bool, device=self.device)

        return TensorDict(
            {
                "observation": observation,
                "done": done,
            },
            batch_size=(),
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """执行一步"""
        action = tensordict["action"]

        # 解析动作（9个参数）
        bid_spread = action[0].item()
        ask_spread = action[1].item()
        order_refresh_time = action[2].item()
        price_deviation_pct = action[3].item()
        hang_order_time_limit = action[4].item()
        max_open_orders = action[5].item()
        take_profit_pct = action[6].item()
        stop_loss_pct = action[7].item()
        order_amount = action[8].item()

        # 更新策略参数
        if self.pmm_strategy is not None:
            self.pmm_strategy.bid_spread = bid_spread
            self.pmm_strategy.ask_spread = ask_spread

            # 时间参数直接使用秒为单位
            self.pmm_strategy.order_refresh_time = order_refresh_time
            self.pmm_strategy.hang_order_time_limit = hang_order_time_limit

            self.pmm_strategy.price_deviation_pct = price_deviation_pct
            self.pmm_strategy.take_profit_pct = take_profit_pct
            self.pmm_strategy.stop_loss_pct = stop_loss_pct

            # 🔧 优化：整数参数转换并确保在正确范围内
            self.pmm_strategy.max_open_orders = max(
                5, min(50, int(max_open_orders)))  # 更合理范围
            self.pmm_strategy.order_amount = max(self.order_amount_range[0], min(
                self.order_amount_range[1], int(order_amount)))

        # 执行策略步骤
        if self.hbt is not None and self.pmm_strategy is not None:
            try:
                # 运行策略一个时间步
                if hasattr(self.hbt, 'elapse'):
                    self.hbt.elapse(self.step_interval_ns)
                self.pmm_strategy.process()
            except Exception as e:
                warnings.warn(f"Strategy execution failed: {e}")

        self.current_step += 1
        self.step_count += 1  # 🔧 新增：步数计数

        # 计算奖励
        reward = self._calculate_reward(action)

        # 🔧 新增：记录奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:  # 保持最近100个奖励
            self.reward_history.pop(0)

        # 更新历史状态
        strategy_state = self._get_strategy_state()
        self.last_balance = strategy_state['balance']
        self.last_position = strategy_state['position']

        # 获取新的观测
        next_observation = self._get_observation()

        # 检查是否结束
        done = (
            self.current_step >= self.max_steps or
            strategy_state['balance'] <= self.initial_balance * 0.1  # 损失90%时停止
        )

        return TensorDict(
            {
                "observation": next_observation,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            },
            batch_size=(),
            device=self.device,
        )

    def _normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """🔧 新增：观测标准化"""
        # 简单的观测标准化：限制在[-1, 1]范围内
        return torch.clamp(obs, -1.0, 1.0)

    def _normalize_reward(self, reward: float) -> float:
        """🔧 新增：奖励标准化"""
        if len(self.reward_history) < 2:
            return reward

        # 使用历史奖励的均值和标准差进行标准化
        reward_mean = np.mean(self.reward_history)
        reward_std = np.std(self.reward_history)

        if reward_std > 0:
            normalized = (reward - reward_mean) / reward_std
            return max(-3.0, min(3.0, float(normalized)))  # 限制在[-3, 3]范围内
        else:
            return reward

    def _set_seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def close(self, *, raise_if_closed: bool = True):
        """关闭环境"""
        if self.hbt is not None:
            try:
                if hasattr(self.hbt, 'close'):
                    self.hbt.close()
            except Exception as e:
                warnings.warn(f"Failed to close backtest: {e}")

        self.hbt = None
        self.pmm_strategy = None


def create_pmm_env(
    data_asset: BacktestAsset,
    initial_balance: float = 1000.0,  # 🔧 优化：降低初始余额
    max_steps: int = 200,  # 🔧 优化：合理的步数
    device: str = "cpu",
    order_amount_range: tuple = (1, 20),  # 🔧 优化：更合理的订单张数范围
    **kwargs
) -> PMMRLEnv:
    """
    创建PMM强化学习环境的便捷函数

    Args:
        data_asset: hftbacktest数据资产
        initial_balance: 初始余额
        max_steps: 最大步数
        device: 计算设备
        order_amount_range: 订单张数范围 (min, max)，如 (1, 50) 表示每个订单1-50张合约
        **kwargs: 其他环境参数

    Returns:
        PMMRLEnv: 配置好的环境实例

    注意:
        现在所有PMM策略参数都通过9维动作空间控制：
        1. bid_spread (0.0001-0.01): 买入价差
        2. ask_spread (0.0001-0.01): 卖出价差
        3. order_refresh_time (0.5-5.0): 订单刷新时间（秒）
        4. price_deviation_pct (0.001-0.01): 价格偏差百分比
        5. hang_order_time_limit (5.0-60.0): 挂单时间限制（秒）
        6. max_open_orders (10-200): 最大挂单数量
        7. take_profit_pct (0.0005-0.005): 止盈百分比
        8. stop_loss_pct (0.001-0.01): 止损百分比
        9. order_amount (可自定义): 每个订单的合约张数
    """
    torch_device = torch.device(device)

    return PMMRLEnv(
        data_asset=data_asset,
        initial_balance=initial_balance,
        max_steps=max_steps,
        device=torch_device,
        order_amount_range=order_amount_range,
        **kwargs
    )
