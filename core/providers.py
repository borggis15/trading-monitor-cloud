from __future__ import annotations

import os
import time
import requests
import pandas as pd
import yfinance as yf


class TwelveDataProvider:
    BASE = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TWELVEDATA_API_KEY", "")

    def fetch(self, symbol: str, exchange: str, interval: str, outputsize: int) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()

        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "outputsize": int(outputsize),
            "apikey": self.api_key,
            "format": "JSON",
        }

        for attempt in (1, 2):
            try:
                r = requests.get(self.BASE, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()

                if "values" not in data:
                    print(f"[WARN] TwelveData sin 'values' para {exchange}:{symbol}. Resp: {str(data)[:250]}")
                    return pd.DataFrame()

                df = pd.DataFrame(data["values"])
                if df.empty:
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
    Stooq CSV endpoint:
      https://stooq.com/q/d/l/?s=SYMBOL&i=d
    Ejemplo: s=pg.us
    """
    BASE = "https://stooq.com/q/d/l/"

    def fetch_daily(self, candidate: str, outputsize: int = 400) -> pd.DataFrame:
        try:
            r = requests.get(self.BASE, params={"s": candidate, "i": "d"}, timeout=20)
            r.raise_for_status()
            txt = r.text
        except Exception:
            return pd.DataFrame()

        if not txt or "Date,Open,High,Low,Close" not in txt:
            return pd.DataFrame()

        from io import StringIO

        df = pd.read_csv(StringIO(txt))
        if df is None or df.empty or "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.rename(
            columns={
                "Date": "ts",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = pd.NA

        df = df.set_index("ts")
        if outputsize and len(df) > int(outputsize):
            df = df.tail(int(outputsize))

        return df[["open", "high", "low", "close", "volume"]]


class YahooProvider:
    def fetch_daily(self, ticker: str, outputsize: int = 400) -> pd.DataFrame:
        try:
            df = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
            )
        except Exception:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # ✅ FIX: aplanar MultiIndex (p.ej. ('Open','LUN.TO')) -> 'Open'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()].sort_index()

        if outputsize and len(df) > int(outputsize):
            df = df.tail(int(outputsize))

        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                df[c] = pd.NA
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df[["open", "high", "low", "close", "volume"]]


class MarketDataProvider:
    def __init__(self):
        self.td = TwelveDataProvider()
        self.stooq = StooqProvider()
        self.yh = YahooProvider()

    def fetch_best(
        self,
        td_symbol: str,
        td_exchange: str,
        interval: str,
        outputsize: int,
        stooq_candidates: list[str] | None = None,
        yahoo_candidates: list[str] | None = None,
    ) -> tuple[pd.DataFrame, str, str]:
        """
        Returns: (df, source, used_symbol)
          source: 'twelvedata' | 'stooq' | 'yahoo' | 'none'
        """
        df = self.td.fetch(symbol=td_symbol, exchange=td_exchange, interval=interval, outputsize=outputsize)
        if df is not None and not df.empty:
            return df, "twelvedata", td_symbol

        if interval in ("1day", "1week", "1month"):
            for c in (stooq_candidates or []):
                df2 = self.stooq.fetch_daily(c, outputsize=outputsize)
                if df2 is not None and not df2.empty:
                    return df2, "stooq", c

            for y in (yahoo_candidates or []):
                df3 = self.yh.fetch_daily(y, outputsize=outputsize)
                if df3 is not None and not df3.empty:
                    return df3, "yahoo", y

        return pd.DataFrame(), "none", ""
