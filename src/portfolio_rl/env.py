import gym
from gym import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, price_series: np.ndarray, window_size: int = 50, initial_cash: float = 1e6):
        super().__init__()
        self.prices = price_series
        self.window = window_size
        self.initial_cash = initial_cash
        self.action_space = spaces.Discrete(3)  # 0: sell, 1: hold, 2: buy
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(window_size,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.step_idx = self.window
        return self._get_obs()

    def _get_obs(self):
        return self.prices[self.step_idx - self.window : self.step_idx]

    def step(self, action):
        price = self.prices[self.step_idx]
        # simple buy/sell logic
        if action == 2:  # buy
            self.shares += self.cash / price
            self.cash = 0
        elif action == 0 and self.shares > 0:  # sell
            self.cash += self.shares * price
            self.shares = 0
        self.step_idx += 1

        done = self.step_idx >= len(self.prices)
        portfolio_value = self.cash + self.shares * price
        reward = portfolio_value - self.initial_cash
        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode="human"):
        print(f"Step {self.step_idx}, Cash: {self.cash:.2f}, Shares: {self.shares:.4f}")
