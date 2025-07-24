# reinforcement-learning-portfolio-manager

A Python framework that uses deep reinforcement learning to allocate capital across a basket of assets. Train agents (DQN, PPO, A2C, etc.) in a custom OpenAI Gym environment, then evaluate performance against benchmarks in a backtest.

### Features
#### Custom Trading Environment
- Built on Gym API (gym.Env): observations include prices, indicators, and portfolio weights
- Supports discrete or continuous action spaces (e.g. fixed-weight buckets vs. direct weight outputs)
- Includes transaction costs, slippage, and leverage constraints

#### Multiple RL Algorithms
- Plug‑and‑play support for Stable Baselines 3 algorithms: DQN, PPO, A2C, SAC, DDPG
- Easy extension to RLlib or custom PyTorch agents
- Configurable hyperparameters via YAML/JSON files

#### Data Handling & Feature Engineering
- Ingestion of OHLCV data from CSV, yfinance, or Parquet sources
- Built‑in technical indicators (moving averages, RSI, MACD) and custom feature pipelines
- Normalization, clipping, and rolling-window observation support

### License
This project is released under the MIT License. Feel free to use, modify, and share!
