from __future__ import annotations
import os, requests
import pandas as pd

class TwelveDataProvider:
    BASE = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["TWELVEDATA_API_KEY"]

    def fetch_15m(self, symbol: str, exchange: str, interval: str = "15min", outputsize: int = 5000) -> pd.DataFrame:
        params = dict(symbol=symbol, exchange=exchange, interval=interval, outputsize=outputsize, apikey=self.api_key, format="JSON")
        r = requests.get(self.BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.rename(columns={"datetime":"ts"}).sort_values("ts").set_index("ts")
        return df[["open","high","low","close","volume"]]
