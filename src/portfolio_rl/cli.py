import typer
import pandas as pd
from .data import fetch_daily
from .agent import train, load

app = typer.Typer()

@app.command()
def train_model(
    symbol: str = typer.Argument(...), 
    timesteps: int = typer.Option(100_000, "--timesteps", "-t")
):
    """Fetch DATA, train RL agent, and save model."""
    df = fetch_daily(symbol)
    prices = df["5. adjusted close"].values
    typer.echo(f"Training on {len(prices)} days of {symbol}…")
    train(prices, timesteps)
    typer.secho("Model saved to model.zip", fg="green")

@app.command()
def simulate(
    symbol: str = typer.Argument(...),
    model_path: str = typer.Option("model.zip", "--model", "-m")
):
    """Run a backtest simulation with a trained model."""
    df = fetch_daily(symbol, outputsize="full")
    prices = df["5. adjusted close"].values
    model = load(model_path)
    env = model.get_env()
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
    env.render()
