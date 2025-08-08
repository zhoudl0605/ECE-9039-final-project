import numpy as np
import torch
from typing import Optional, Dict, List
import warnings

from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, Bounded
from tensordict import TensorDict, TensorDictBase

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, ROIVectorMarketDepthBacktest
from hftbacktest.stats import LinearAssetRecord
from .pmm_pure import pure_pmm, pure_pmm_step


class PMMRLEnv(EnvBase):
    """
    基于PMM策略的TorchRL强化学习环境
    适用于SAC等连续控制算法
    """

    def __init__(
        self,
        data_asset: BacktestAsset,
        action_low: List[float],
        action_high: List[float],
        max_steps: int = 200,
        step_interval_ns: int = 500_000_000,
        device: torch.device = torch.device("cpu"),
        risk_penalty_weight: float = 0.05,
        transaction_cost_rate: float = 0.0005,
        reward_normalization: bool = True,
        observation_normalization: bool = True,
    ):
        """
        初始化PMM强化学习环境

        Args:
            data_asset: hftbacktest数据资产
            action_low: 动作空间下界 [half_spread, skew, grid_num, grid_interval_multiplier] (必需)
            action_high: 动作空间上界 [half_spread, skew, grid_num, grid_interval_multiplier] (必需)
            max_steps: 最大步数
            step_interval_ns: 每步时间间隔（纳秒）
            device: 计算设备
            risk_penalty_weight: 风险惩罚权重
            transaction_cost_rate: 交易成本率
            reward_normalization: 是否对奖励进行归一化
            observation_normalization: 是否对观测进行归一化

        动作空间说明 (4维连续动作):
            1. half_spread: 半价差（tick数）
            2. skew: 偏度系数，控制仓位对报价的影响
            3. grid_num: 网格订单层数
            4. grid_interval_multiplier: 网格间隔（tick_size的倍数）
        """
        super().__init__(device=device)

        self.data_asset = data_asset
        self.max_steps = max_steps
        self.step_interval_ns = step_interval_ns
        self.risk_penalty_weight = risk_penalty_weight
        self.transaction_cost_rate = transaction_cost_rate
        self.reward_normalization = reward_normalization
        self.observation_normalization = observation_normalization

        # 环境状态
        self.current_step = 0
        self.hbt = None
        self.stat = None
        self.recorder = None
        self.data_finished = False
        self.last_pnl = 0.0  # PnL从0开始
        self.last_position = 0.0

        # PMM策略参数
        self.half_spread = 40
        self.skew = 10  # 现在表示tick数，默认10个tick
        self.order_qty_dollar = 50.0  # 固定值
        self.max_position_dollar = 1000.0
        self.grid_num = 10
        self.grid_interval = 0.5

        # 定义动作空间 - 4维连续动作空间
        # 动作维度：[half_spread, skew, grid_num, grid_interval_multiplier]
        # grid_interval_multiplier 将被转换为 tick_size 的整数倍
        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor(action_low, device=self.device),
                high=torch.tensor(action_high, device=self.device),
                shape=torch.Size([4]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # 定义观测空间 - 4维状态向量
        # 观测维度：[mid_price, spread, position, balance_ratio]
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=torch.Size([4]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # 奖励规格
        self.reward_spec = Unbounded(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

        # 完成标志规格
        self.done_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([1]),
            dtype=torch.bool,
            device=self.device,
        )

    def _initialize_backtest(self):
        """初始化回测环境"""
        self.hbt = HashMapMarketDepthBacktest([self.data_asset])
        # 注意：HFTBacktest中balance初始值为0，表示余额变化量
        # 不需要也不能直接设置初始balance

        # 初始化统计记录器
        from hftbacktest import Recorder
        self.recorder = Recorder(1, 100_000)  # 记录1个资产，最多100k条记录
        self.stat = self.recorder.recorder

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

            # 计算简单波动率 (基于价差比例)
            volatility = spread if spread > 0 else 0.0

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
        if self.hbt is None:
            return {
                'position': 0.0,
                'pnl': 0.0,
            }

        try:
            position = float(self.hbt.position(0))
            pnl = float(self.hbt.state_values(0).balance)

            return {
                'position': position,
                'pnl': pnl,
            }
        except Exception as e:
            warnings.warn(f"Failed to get strategy state: {e}")
            return {
                'position': 0.0,
                'pnl': 0.0,
            }

    def _get_observation(self) -> torch.Tensor:
        """获取环境观测 - 4维版本"""
        market_state = self._get_market_state()
        strategy_state = self._get_strategy_state()

        # 直接使用原始值，无需标准化
        # 组合观测向量（4维）
        obs = torch.tensor([
            market_state['mid_price'],        # 中间价（美元）
            # 价差（美元）
            market_state['best_ask'] -
            market_state['best_bid'] if market_state['best_bid'] > 0 else 0,
            strategy_state['position'],       # 仓位（币数量）
            strategy_state['pnl'],            # PnL（美元）
        ], dtype=torch.float32, device=self.device)

        # 观测标准化
        if self.observation_normalization:
            obs = self._normalize_observation(obs)

        return obs

    def _calculate_reward(self, action: torch.Tensor) -> float:
        """计算奖励函数"""
        strategy_state = self._get_strategy_state()

        # PnL奖励（已包含负费率返佣）
        current_pnl = strategy_state['pnl']
        pnl_change = current_pnl - self.last_pnl
        # 直接使用PnL变化作为奖励（以美元为单位）
        pnl_reward = pnl_change  # 赚1美元奖励1，亏1美元奖励-1

        # 风险惩罚（基于仓位价值）
        position = strategy_state['position']
        # 仓位价值 ≈ 仓位数量 * 当前价格
        market_state = self._get_market_state()
        position_value = abs(
            position * market_state['mid_price']) if market_state['mid_price'] > 0 else 0
        # 风险惩罚：仓位价值超过1000美元开始惩罚
        risk_penalty = self.risk_penalty_weight * \
            (position_value / 1000.0) ** 2

        # 负费率环境下没有交易成本，返佣已包含在PnL中
        # transaction_cost = 0 (移除)

        # 综合奖励
        total_reward = pnl_reward - risk_penalty

        # 奖励标准化
        if self.reward_normalization:
            total_reward = self._normalize_reward(total_reward)

        return total_reward

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """重置环境"""
        self.current_step = 0
        self.data_finished = False
        self.last_pnl = 0.0  # PnL从0开始
        self.last_position = 0.0

        # 重新初始化回测环境
        self._initialize_backtest()

        # 重置PMM策略参数为默认值
        self.half_spread = 40
        self.skew = 10  # tick数
        self.order_qty_dollar = 50.0
        self.grid_num = 10
        self.grid_interval = 0.5

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

        # 解析动作（4个参数）
        self.half_spread = int(action[0].item())
        self.skew = action[1].item()
        self.grid_num = int(action[2].item())

        # grid_interval 必须是 tick_size 的整数倍
        # 将连续值转换为tick_size的倍数
        tick_size = self.hbt.depth(0).tick_size if self.hbt else 0.01
        grid_interval_ticks = max(1, int(action[3].item() / tick_size))
        self.grid_interval = grid_interval_ticks * tick_size

        # 执行策略步骤
        if self.hbt is not None:
            try:
                self.data_finished = pure_pmm_step(
                    hbt=self.hbt,
                    stat=self.stat,
                    half_spread=self.half_spread,
                    skew=self.skew,
                    interval=self.step_interval_ns,
                    order_qty_dollar=self.order_qty_dollar,
                    max_position_dollar=self.max_position_dollar,
                    grid_num=self.grid_num,
                    grid_interval=self.grid_interval,
                )
            except Exception as e:
                warnings.warn(f"Strategy execution failed: {e}")

        self.current_step += 1

        # 计算奖励
        reward = self._calculate_reward(action)

        # 更新历史状态
        strategy_state = self._get_strategy_state()
        self.last_pnl = strategy_state['pnl']
        self.last_position = strategy_state['position']

        # 获取新的观测
        next_observation = self._get_observation()

        # 检查是否结束
        done = (
            self.data_finished or  # 数据用完
            self.current_step >= self.max_steps or
            # PnL损失超过阈值时停止
            strategy_state['pnl'] <= -500.0  # 损失500美元（更合理的阈值）
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
        """观测处理"""
        # 不做标准化，直接返回原始值
        return obs

    def _normalize_reward(self, reward: float) -> float:
        """奖励处理"""
        # 不做裁剪，直接返回原始奖励
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
        self.stat = None
        self.recorder = None


def create_pmm_env(
    data_asset: BacktestAsset,
    action_low: List[float],
    action_high: List[float],
    max_steps: int = 200,
    device: str = "cpu",
    **kwargs
) -> PMMRLEnv:
    """
    创建PMM强化学习环境的便捷函数

    Args:
        data_asset: hftbacktest数据资产
        action_low: 动作空间下界 (必需)
        action_high: 动作空间上界 (必需)
        max_steps: 最大步数
        device: 计算设备
        **kwargs: 其他环境参数

    Returns:
        PMMRLEnv: 配置好的环境实例

    示例:
        action_low = [1.0, 0.0001, 1.0, 1.0]     # [half_spread, skew, grid_num, grid_interval_multiplier]
        action_high = [50.0, 0.001, 10.0, 50.0]

        env = create_pmm_env(
            data_asset=asset,
            action_low=action_low,
            action_high=action_high,
        )
    """
    torch_device = torch.device(device)

    return PMMRLEnv(
        data_asset=data_asset,
        action_low=action_low,
        action_high=action_high,
        max_steps=max_steps,
        device=torch_device,
        **kwargs
    )
