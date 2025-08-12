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
    TorchRL reinforcement learning environment based on PMM strategy
    Suitable for continuous control algorithms like SAC
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
        Initialize PMM reinforcement learning environment

        Args:
            data_asset: hftbacktest data asset
            action_low: Action space lower bounds [half_spread, skew, grid_num, grid_interval_multiplier] (required)
            action_high: Action space upper bounds [half_spread, skew, grid_num, grid_interval_multiplier] (required)
            max_steps: Maximum number of steps
            step_interval_ns: Time interval per step (nanoseconds)
            device: Computing device
            risk_penalty_weight: Risk penalty weight
            transaction_cost_rate: Transaction cost rate
            reward_normalization: Whether to normalize rewards
            observation_normalization: Whether to normalize observations

        Action space description (4D continuous actions):
            1. half_spread: Half spread (in ticks)
            2. skew: Skew coefficient, controls position impact on quotes
            3. grid_num: Number of grid order levels
            4. grid_interval_multiplier: Grid interval (multiples of tick_size)
        """
        super().__init__(device=device)

        self.data_asset = data_asset
        self.max_steps = max_steps
        self.step_interval_ns = step_interval_ns
        self.risk_penalty_weight = risk_penalty_weight
        self.transaction_cost_rate = transaction_cost_rate
        self.reward_normalization = reward_normalization
        self.observation_normalization = observation_normalization

        # Environment state
        self.current_step = 0
        self.hbt = None
        self.stat = None
        self.recorder = None
        self.data_finished = False
        self.last_pnl = 0.0  # PnL starts from 0
        self.last_position = 0.0

        # PMM strategy parameters
        self.half_spread = 40
        self.skew = 10  # Now represents tick count, default 10 ticks
        self.order_qty_dollar = 50.0  # Fixed value
        self.max_position_dollar = 1000.0
        self.grid_num = 10
        self.grid_interval = 0.5

        # Define action space - 4D continuous action space
        # Action dimensions: [half_spread, skew, grid_num, grid_interval_multiplier]
        # grid_interval_multiplier will be converted to integer multiples of tick_size
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

        # Define observation space - 4D state vector
        # Observation dimensions: [mid_price, spread, position, balance_ratio]
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=torch.Size([4]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # Reward specification
        self.reward_spec = Unbounded(
            shape=torch.Size([1]),
            dtype=torch.float32,
            device=self.device,
        )

        # Done flag specification
        self.done_spec = Bounded(
            low=0,
            high=1,
            shape=torch.Size([1]),
            dtype=torch.bool,
            device=self.device,
        )

    def _initialize_backtest(self):
        """Initialize backtest environment"""
        self.hbt = HashMapMarketDepthBacktest([self.data_asset])
        # Note: In HFTBacktest, balance initial value is 0, representing balance change
        # No need and cannot directly set initial balance

        # Initialize statistics recorder
        from hftbacktest import Recorder
        self.recorder = Recorder(1, 100_000)  # Record 1 asset, max 100k records
        self.stat = self.recorder.recorder

    def _get_market_state(self) -> Dict[str, float]:
        """Get market state"""
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

            # Calculate mid price and spread
            mid_price = (best_bid + best_ask) / \
                2.0 if best_bid > 0 and best_ask > 0 else 0.0
            spread = (best_ask - best_bid) / \
                mid_price if mid_price > 0 else 0.0

            # Calculate simple volatility (based on spread ratio)
            volatility = spread if spread > 0 else 0.0

            # Calculate order book imbalance
            order_book_imbalance = 0.0
            if best_bid > 0 and best_ask > 0:
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
        """Get strategy state"""
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
        """Get environment observation - 4D version"""
        market_state = self._get_market_state()
        strategy_state = self._get_strategy_state()

        # Use raw values directly, no normalization needed
        # Combine observation vector (4D)
        obs = torch.tensor([
            market_state['mid_price'],        # Mid price (USD)
            # Spread (USD)
            market_state['best_ask'] -
            market_state['best_bid'] if market_state['best_bid'] > 0 else 0,
            strategy_state['position'],       # Position (coin quantity)
            strategy_state['pnl'],            # PnL (USD)
        ], dtype=torch.float32, device=self.device)

        # Observation normalization
        if self.observation_normalization:
            obs = self._normalize_observation(obs)

        return obs

    def _calculate_reward(self, action: torch.Tensor) -> float:
        """Calculate reward function"""
        strategy_state = self._get_strategy_state()

        # PnL reward (already includes negative fee rebate)
        current_pnl = strategy_state['pnl']
        pnl_change = current_pnl - self.last_pnl
        # Use PnL change directly as reward (in USD)
        pnl_reward = pnl_change  # Earn $1 reward 1, lose $1 reward -1

        # Risk penalty (based on position value)
        position = strategy_state['position']
        # Position value ≈ position quantity * current price
        market_state = self._get_market_state()
        position_value = abs(
            position * market_state['mid_price']) if market_state['mid_price'] > 0 else 0
        # Risk penalty: penalty starts when position value exceeds $1000
        risk_penalty = self.risk_penalty_weight * \
            (position_value / 1000.0) ** 2

        # Negative fee environment: rebate already included in PnL

        # Total reward
        total_reward = pnl_reward - risk_penalty

        # Reward normalization
        if self.reward_normalization:
            total_reward = self._normalize_reward(total_reward)

        return total_reward

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        """Reset environment"""
        self.current_step = 0
        self.data_finished = False
        self.last_pnl = 0.0  # PnL starts from 0
        self.last_position = 0.0

        # Reinitialize backtest environment
        self._initialize_backtest()

        # Reset PMM strategy parameters to defaults
        self.half_spread = 40
        self.skew = 10  # tick count
        self.order_qty_dollar = 50.0
        self.grid_num = 10
        self.grid_interval = 0.5

        # Get initial observation
        observation = self._get_observation()

        # Initialize done flag
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
        """Execute one step"""
        action = tensordict["action"]

        # Parse action (4 parameters)
        self.half_spread = int(action[0].item())
        self.skew = action[1].item()
        self.grid_num = int(action[2].item())

        # Convert grid_interval to integer multiples of tick_size
        tick_size = self.hbt.depth(0).tick_size if self.hbt else 0.01
        grid_interval_ticks = max(1, int(action[3].item() / tick_size))
        self.grid_interval = grid_interval_ticks * tick_size

        # Execute strategy step
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

        # Calculate reward
        reward = self._calculate_reward(action)

        # Update historical state
        strategy_state = self._get_strategy_state()
        self.last_pnl = strategy_state['pnl']
        self.last_position = strategy_state['position']

        # Get new observation
        next_observation = self._get_observation()

        # Check if done
        done = (
            self.data_finished or  # Data exhausted
            self.current_step >= self.max_steps or
            # Stop when PnL loss exceeds threshold
            strategy_state['pnl'] <= -500.0  # Loss of $500 (more reasonable threshold)
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
        """Observation processing"""
        return obs

    def _normalize_reward(self, reward: float) -> float:
        """Reward processing"""
        return reward

    def _set_seed(self, seed: Optional[int] = None):
        """Set random seed"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def close(self, *, raise_if_closed: bool = True):
        """Close environment"""
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
    Convenience function to create PMM reinforcement learning environment

    Args:
        data_asset: hftbacktest data asset
        action_low: Action space lower bounds (required)
        action_high: Action space upper bounds (required)
        max_steps: Maximum number of steps
        device: Computing device
        **kwargs: Other environment parameters

    Returns:
        PMMRLEnv: Configured environment instance

    Example:
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
