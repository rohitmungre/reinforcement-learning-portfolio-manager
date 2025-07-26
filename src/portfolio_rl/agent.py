from stable_baselines3 import PPO
from .env import PortfolioEnv

def train(price_series, timesteps: int = 100_000, model_path: str = "model.zip"):
    env = PortfolioEnv(price_series)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    return model

def load(model_path: str = "model.zip"):
    return PPO.load(model_path)
