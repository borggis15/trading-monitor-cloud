from __future__ import annotations

import io
import pandas as pd
import requests
from sqlalchemy import text

from core.config import load_config
from core.db import get_engine

# Limita el backfill inicial para no timeoutear
# (daily: 2500 filas ~ 10 años de cotización si hay 252 sesiones/año)
INITIAL_MAX_ROWS = 2500


# ----------------------------
# Fetchers
# ----------------------------
def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """
    Stooq daily CSV: https://stooq.com/q/d/l/?s=SYMBOL&i=d
    symbol examples: pg.us, lly.us, lun.to, etc.
    """
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


def fetch_yahoo_daily(ticker: str) -> pd.DataFrame:
    """
    Yahoo daily via yfinance. ticker examples: LUN.TO
    """
    import yfinance as yf  # noqa: F401

    df = yf.download(
        tickers=ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance devuelve índice DateTimeIndex
    df = df.reset_index()

    # Column names can be: Date, Open, High, Low, Close, Adj Close, Volume
    # A veces 'Date' o 'Datetime' según versión
    if "Date" in df.columns:
        ts_col = "Date"
    elif "Datetime" in df.columns:
        ts_col = "Datetime"
    else:
        # fallback: primera columna
        ts_col = df.columns[0]

    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == ts_col.lower():
            rename[c] = "ts"
        elif cl == "open":
            rename[c] = "open"
        elif cl == "high":
            rename[c] = "high"
        elif cl == "low":
            rename[c] = "low"
        elif cl == "close":
            rename[c] = "close"
        elif cl == "volume":
            rename[c] = "volume"

    df = df.rename(columns=rename)

    need = {"ts", "open", "high", "low", "close", "volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df[["ts", "open", "high", "low", "close", "volume"]]


# ----------------------------
# DB helpers
# ----------------------------
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


# ----------------------------
# Candidate resolver
# ----------------------------
def resolve_daily_series(inst: dict) -> list[tuple[str, str, str]]:
    """
    candidates as (exchange, symbol, source)

    Prefer STOOQ for free daily history.
    Fallback to YAHOO (yfinance) when STOOQ not available.
    """
    cands: list[tuple[str, str, str]] = []

    # 1) STOOQ
    for s in inst.get("stooq_candidates", []) or []:
        if s:
            cands.append(("STOOQ", s, "stooq"))

    # 2) YAHOO fallback
    for y in inst.get("yahoo_candidates", []) or []:
        if y:
            cands.append(("YAHOO", y, "yahoo"))

    return cands


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_config()
    engine = get_engine()

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name

        print(f"[INGEST_1D] {name}")

        ok = False
        last_err = None

        for ex, sym, src in resolve_daily_series(inst):
            try:
                if src == "stooq":
                    df = fetch_stooq_daily(sym)
                elif src == "yahoo":
                    df = fetch_yahoo_daily(sym)
                else:
                    df = pd.DataFrame()

                if df is None or df.empty:
                    continue

                latest = get_latest_ts(engine, ex, sym)

                # incremental
                if latest is not None:
                    df_new = df[df["ts"] > latest].copy()
                else:
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
                    source=src,
                    asset_id=asset_id,
                    asset_name=name,
                )
                print(f"[OK] {name}: {ex}:{sym} inserted={n} (latest_in_db={latest})")
                ok = True
                processed += 1
                break

            except Exception as e:
                last_err = e
                print(f"[WARN] {name} {ex}:{sym} failed: {e}")

        if not ok:
            print(f"[SKIP] {name}: no daily data from free sources (last_err={last_err})")

    print(f"INGEST_1D OK: {processed} assets")


if __name__ == "__main__":
    main()
