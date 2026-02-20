from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from sqlalchemy import text

from core.config import load_config
from core.db import get_engine

# Limita el backfill inicial para no timeoutear
# (daily: 2500 filas ~ 10 años si ~252 sesiones/año)
INITIAL_MAX_ROWS = 2500


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=30)
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


def fetch_yahoo_daily(ticker: str, start_dt_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Usa yfinance (gratis) para descargar histórico diario.
    - Si start_dt_utc existe, pedimos desde ahí para reducir carga.
    """
    start = None
    if start_dt_utc is not None and pd.notna(start_dt_utc):
        # pedir desde el día siguiente para evitar duplicados
        start = (start_dt_utc + pd.Timedelta(days=1)).date().isoformat()

    try:
        df = yf.download(
            tickers=ticker,
            start=start,
            period=None if start else "max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance devuelve el índice como datetime (a veces tz-naive). Lo normalizamos a UTC.
    df = df.reset_index()

    # El nombre de la columna puede ser "Date" o "Datetime" según versión
    if "Date" in df.columns:
        ts_col = "Date"
    elif "Datetime" in df.columns:
        ts_col = "Datetime"
    else:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    out["open"] = pd.to_numeric(df.get("Open"), errors="coerce")
    out["high"] = pd.to_numeric(df.get("High"), errors="coerce")
    out["low"] = pd.to_numeric(df.get("Low"), errors="coerce")
    out["close"] = pd.to_numeric(df.get("Close"), errors="coerce")

    # Volume puede venir como int/float
    out["volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")

    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    return out[["ts", "open", "high", "low", "close", "volume"]]


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


def upsert_bars_1d(
    engine,
    exchange: str,
    symbol: str,
    df: pd.DataFrame,
    source: str,
    asset_id: str,
    asset_name: str,
) -> int:
    if df is None or df.empty:
        return 0

    d = df.copy()
    d["exchange"] = exchange
    d["symbol"] = symbol
    d["source"] = source
    d["asset_id"] = asset_id
    d["asset_name"] = asset_name

    cols = [
        "exchange",
        "symbol",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "asset_id",
        "asset_name",
    ]

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
    Prefer STOOQ for free daily history, fallback to YAHOO.
    """
    cands: list[tuple[str, str, str]] = []

    # 1) STOOQ primero
    for s in inst.get("stooq_candidates", []) or []:
        cands.append(("STOOQ", s, "stooq"))

    # 2) YAHOO fallback
    for y in inst.get("yahoo_candidates", []) or []:
        cands.append(("YAHOO", y, "yahoo"))

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
                latest = get_latest_ts(engine, ex, sym)

                if src == "stooq":
                    df = fetch_stooq_daily(sym)
                    source_label = "stooq"
                elif src == "yahoo":
                    df = fetch_yahoo_daily(sym, start_dt_utc=latest)
                    source_label = "yahoo"
                else:
                    df = pd.DataFrame()
                    source_label = src

                if df is None or df.empty:
                    continue

                # Incremental:
                if latest is not None and src != "yahoo":
                    # para stooq filtramos localmente (yahoo ya lo pide desde start)
                    df_new = df[df["ts"] > latest].copy()
                else:
                    # First fill: limit rows to avoid timeouts
                    df_new = df.tail(INITIAL_MAX_ROWS).copy()

                if df_new.empty:
                    print(f"[OK] {name}: {ex}:{sym} no new rows")
                    ok = True
                    processed += 1
                    break

                n = upsert_bars_1d(
                    engine,
                    ex,
                    sym,
                    df_new,
                    source=source_label,
                    asset_id=asset_id,
                    asset_name=name,
                )
                print(f"[OK] {name}: {ex}:{sym} inserted={n} (latest_in_db={latest})")
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
