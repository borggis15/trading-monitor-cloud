from __future__ import annotations

import io
import pandas as pd
import requests
from sqlalchemy import text
from datetime import datetime, timezone

from core.config import load_config
from core.db import get_engine


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    # Stooq daily CSV: https://stooq.com/q/d/l/?s={symbol}&i=d
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={
        "Date": "ts",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].dropna(subset=["ts"]).sort_values("ts")


def upsert_bars_1d(engine, exchange: str, symbol: str, df: pd.DataFrame, source: str, asset_id: str, asset_name: str) -> int:
    if df is None or df.empty:
        return 0

    d = df.copy()
    d["exchange"] = exchange
    d["symbol"] = symbol
    d["source"] = source
    d["asset_id"] = asset_id
    d["asset_name"] = asset_name

    cols = ["exchange","symbol","ts","open","high","low","close","volume","source","asset_id","asset_name"]

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.bars_1d(exchange,symbol,ts,open,high,low,close,volume,source,asset_id,asset_name)
                values (:exchange,:symbol,:ts,:open,:high,:low,:close,:volume,:source,:asset_id,:asset_name)
                on conflict (exchange,symbol,ts) do update set
                  open=excluded.open,
                  high=excluded.high,
                  low=excluded.low,
                  close=excluded.close,
                  volume=excluded.volume,
                  source=excluded.source,
                  asset_id=excluded.asset_id,
                  asset_name=excluded.asset_name
                """
            ),
            d[cols].to_dict(orient="records"),
        )

    return len(d)


def resolve_daily_series(inst: dict) -> list[tuple[str,str,str]]:
    """
    Returns candidates as (exchange, symbol, source)
    Prefer STOOQ if available because it gives long daily history.
    You can add more candidates later.
    """
    cands = []

    # Prefer stooq_candidates first if present
    for s in inst.get("stooq_candidates", []) or []:
        cands.append(("STOOQ", s, "stooq"))

    # Yahoo daily can be added later if you want (yfinance), but stooq is enough for free daily growth.

    # As a final fallback, try xetra symbol via stooq as-is (sometimes works)
    xs = inst.get("xetra_symbol")
    if xs:
        cands.append(("XETR", xs.lower(), "stooq"))  # sometimes stooq uses lowercase, but depends

    # Primary fallback as stooq too (many US tickers exist)
    ps = inst.get("primary_symbol")
    if ps:
        cands.append(("PRIMARY", ps.lower(), "stooq"))

    return cands


def main():
    cfg = load_config()
    engine = get_engine()

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name

        print(f"[INGEST_1D] {name}")

        ok = False
        for ex, sym, src in resolve_daily_series(inst):
            try:
                if src == "stooq":
                    df = fetch_stooq_daily(sym)
                else:
                    df = pd.DataFrame()

                if df is None or df.empty:
                    continue

                n = upsert_bars_1d(engine, ex, sym, df, source=src, asset_id=asset_id, asset_name=name)
                print(f"[OK] {name}: {ex}:{sym} {n} rows ({src})")
                ok = True
                processed += 1
                break

            except Exception as e:
                print(f"[WARN] {name} {ex}:{sym} failed: {e}")

        if not ok:
            print(f"[SKIP] {name}: no daily data from free sources")

    print(f"INGEST_1D OK: {processed} assets")


if __name__ == "__main__":
    main()
