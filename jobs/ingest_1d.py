from __future__ import annotations

import io
import time
import pandas as pd
import requests
from sqlalchemy import text

from core.config import load_config
from core.db import get_engine


# 2500 filas ~ ~10 años bursátiles (252 sesiones/año)
INITIAL_MAX_ROWS = 2500

# Buenas prácticas en CI
HTTP_TIMEOUT = 30
UA = "Mozilla/5.0 (compatible; TradingMonitor/1.0; +https://github.com/)"

session = requests.Session()
session.headers.update({"User-Agent": UA})


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """
    STOOQ CSV daily: https://stooq.com/q/d/l/?s=pg.us&i=d
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

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
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df[["ts", "open", "high", "low", "close", "volume"]]


def fetch_yahoo_daily_chart(ticker: str, range_: str = "10y") -> pd.DataFrame:
    """
    Yahoo public JSON chart endpoint (no API key, usually works in GitHub Actions):
    https://query2.finance.yahoo.com/v8/finance/chart/LUN.TO?range=10y&interval=1d&includePrePost=false&events=div%7Csplit
    """
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={range_}&interval=1d&includePrePost=false&events=div%7Csplit"
    )
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()

    chart = (j or {}).get("chart", {})
    if chart.get("error"):
        return pd.DataFrame()

    result = (chart.get("result") or [])
    if not result:
        return pd.DataFrame()

    res0 = result[0]
    ts_list = res0.get("timestamp") or []
    ind = (((res0.get("indicators") or {}).get("quote") or [None])[0]) or {}
    if not ts_list or not ind:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts_list, unit="s", utc=True, errors="coerce"),
            "open": ind.get("open"),
            "high": ind.get("high"),
            "low": ind.get("low"),
            "close": ind.get("close"),
            "volume": ind.get("volume"),
        }
    )
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    return df[["ts", "open", "high", "low", "close", "volume"]]


def get_latest_ts(engine, exchange: str, symbol: str):
    q = """
    select max(ts) as max_ts
    from public.bars_1d
    where exchange=:exchange and symbol=:symbol
    """
    row = pd.read_sql(text(q), engine, params={"exchange": exchange, "symbol": symbol})
    if row.empty or pd.isna(row.loc[0, "max_ts"]):
        return None
    return pd.to_datetime(row.loc[0, "max_ts"], utc=True, errors="coerce")


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


def resolve_daily_series(inst: dict) -> list[tuple[str, str, str]]:
    """
    candidates as (exchange, symbol, source)
    Prefer STOOQ for free daily history.
    Then Yahoo chart as fallback (still free, no API key).
    """
    cands: list[tuple[str, str, str]] = []

    # 1) STOOQ first
    for s in inst.get("stooq_candidates", []) or []:
        cands.append(("STOOQ", s, "stooq"))

    # 2) YAHOO fallback
    for y in inst.get("yahoo_candidates", []) or []:
        cands.append(("YAHOO", y, "yahoo"))

    return cands


def main():
    print("[BOOT] INGEST_1D v3 (STOOQ + YAHOO chart fallback)")
    cfg = load_config()
    engine = get_engine()

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name

        print(f"[INGEST_1D] {name}")

        candidates = resolve_daily_series(inst)
        print(f"[CANDS] {name}: {candidates}")

        ok = False
        last_err = None

        for ex, sym, src in candidates:
            try:
                print(f"[TRY] {name}: {ex}:{sym} src={src}")

                if src == "stooq":
                    df = fetch_stooq_daily(sym)
                elif src == "yahoo":
                    # Para no timeoutear, usamos 10y. Si quieres más, sube a "max".
                    df = fetch_yahoo_daily_chart(sym, range_="10y")
                else:
                    df = pd.DataFrame()

                if df is None or df.empty:
                    print(f"[EMPTY] {name}: {ex}:{sym}")
                    continue

                latest = get_latest_ts(engine, ex, sym)

                # Incremental:
                if latest is not None:
                    df_new = df[df["ts"] > latest].copy()
                else:
                    # First fill: limit rows to avoid timeouts
                    df_new = df.tail(INITIAL_MAX_ROWS).copy()

                if df_new.empty:
                    print(f"[OK] {name}: {ex}:{sym} no new rows")
                    ok = True
                    processed += 1
                    break

                n = upsert_bars_1d(engine, ex, sym, df_new, source=src, asset_id=asset_id, asset_name=name)
                print(f"[OK] {name}: {ex}:{sym} inserted={n} (latest_in_db={latest})")
                ok = True
                processed += 1
                break

            except Exception as e:
                last_err = e
                print(f"[WARN] {name} {ex}:{sym} failed: {e}")
                # pequeña pausa para ser amable con endpoints públicos
                time.sleep(0.5)

        if not ok:
            print(f"[SKIP] {name}: no daily data from free sources (last_err={last_err})")

    print(f"INGEST_1D OK: {processed} assets")


if __name__ == "__main__":
    main()
