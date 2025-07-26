import numpy as np
from portfolio_rl.env import PortfolioEnv

def test_env_step_reset():
    prices = np.linspace(1, 100, 1000)
    env = PortfolioEnv(prices, window_size=10)
    obs = env.reset()
    assert len(obs) == 10
    obs2, reward, done, _ = env.step(2)  # buy
    assert isinstance(reward, float)
