from __future__ import annotations

import os
import time
import requests
import pandas as pd

# Stooq via pandas-datareader (gratis EOD)
from pandas_datareader import data as pdr


class TwelveDataProvider:
    BASE = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["TWELVEDATA_API_KEY"]

    def fetch(self, symbol: str, exchange: str, interval: str, outputsize: int) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "outputsize": int(outputsize),
            "apikey": self.api_key,
            "format": "JSON",
        }

        # 2 intentos rápidos
        for attempt in (1, 2):
            try:
                r = requests.get(self.BASE, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()

                # Si no hay 'values', Twelve Data devuelve error con code/message
                if "values" not in data:
                    msg = str(data)[:300]
                    print(f"[WARN] TwelveData sin 'values' para {exchange}:{symbol}. Resp: {msg}")
                    return pd.DataFrame()

                df = pd.DataFrame(data["values"])
                if df.empty:
                    print(f"[WARN] TwelveData values vacío para {exchange}:{symbol}.")
                    return pd.DataFrame()

                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                for c in ["open", "high", "low", "close", "volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                df = df.rename(columns={"datetime": "ts"}).sort_values("ts").set_index("ts")
                return df[["open", "high", "low", "close", "volume"]]

            except Exception as e:
                print(f"[WARN] TwelveData fetch fallo ({attempt}/2) para {exchange}:{symbol} -> {e}")
                if attempt == 2:
                    return pd.DataFrame()
                time.sleep(2)

        return pd.DataFrame()


class StooqProvider:
    """
    Stooq (gratis) – funciona muy bien para daily.
    Usamos pandas-datareader StooqDailyReader (EOD).
    """

    def fetch_daily_candidates(self, candidates: list[str], outputsize: int = 800) -> pd.DataFrame:
        # Stooq devuelve el índice como fecha, columnas Open/High/Low/Close/Volume
        # (a veces en orden descendente), lo normalizamos.
        for s in candidates:
            try:
                df = pdr.DataReader(s, "stooq")
                if df is None or df.empty:
                    continue

                # normaliza columnas y orden
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.sort_index()

                # recorta a outputsize
                if outputsize and len(df) > outputsize:
                    df = df.tail(int(outputsize))

                # asegurar columnas
                cols = ["open", "high", "low", "close", "volume"]
                for c in cols:
                    if c not in df.columns:
                        df[c] = pd.NA
                return df[cols]

            except Exception as e:
                print(f"[WARN] Stooq fallo para {s}: {e}")
                continue

        return pd.DataFrame()


class MarketDataProvider:
    """
    Provider unificado:
      - intenta Twelve Data
      - si falla, intenta Stooq (daily)
    """

    def __init__(self):
        self.td = TwelveDataProvider()
        self.stooq = StooqProvider()

    def fetch(self, symbol: str, exchange: str, interval: str, outputsize: int, stooq_candidates: list[str] | None = None) -> pd.DataFrame:
        df = self.td.fetch(symbol=symbol, exchange=exchange, interval=interval, outputsize=outputsize)
        if df is not None and not df.empty:
            return df

        # fallback stooq SOLO tiene sentido para daily
        if interval not in ("1day", "1week", "1month"):
            return pd.DataFrame()

        candidates = stooq_candidates or []
        if not candidates:
            return pd.DataFrame()

        df2 = self.stooq.fetch_daily_candidates(candidates, outputsize=outputsize)
        return df2
