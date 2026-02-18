from __future__ import annotations
import os, time, requests
import pandas as pd

class TwelveDataProvider:
    BASE = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["TWELVEDATA_API_KEY"]

    def fetch_15m(self, symbol: str, exchange: str, interval: str = "15min", outputsize: int = 800) -> pd.DataFrame:
        """
        outputsize reducido por defecto para jobs frecuentes (15m).
        800 velas = ~8-10 días de trading intradía (suficiente para señales rápidas).
        """
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
                r = requests.get(self.BASE, params=params, timeout=25)
                r.raise_for_status()
                data = r.json()

                if "values" not in data:
                    # Twelve Data suele devolver {"code":..., "message":...} cuando el símbolo no existe
                    print(f"[WARN] TwelveData sin 'values' para {exchange}:{symbol}. Resp: {str(data)[:200]}")
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
