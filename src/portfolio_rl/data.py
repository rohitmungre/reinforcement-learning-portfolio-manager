import requests
import pandas as pd
from .config import settings

def fetch_daily(symbol: str, outputsize: str = "compact") -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": settings.alphavantage_key,
        "datatype": "json",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(data, orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
