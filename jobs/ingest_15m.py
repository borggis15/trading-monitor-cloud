from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from core.config import load_config
from core.db import get_engine
from core.providers import MarketDataProvider


def _normalize_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    # ✅ FIX: aplanar MultiIndex de columnas si existiera
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()]

    out = out.sort_index()
    out.index.name = "ts"
    return out


def upsert_bars(engine, exchange: str, symbol: str, df: pd.DataFrame, source: str):
    if df is None or df.empty:
        return 0

    df = _normalize_ts_index(df)
    if df is None or df.empty:
        return 0

    d = df.reset_index()

    # ✅ FIX: aplanar MultiIndex tras reset_index si existiera
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]

    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["exchange"] = exchange
    d["symbol"] = symbol
    d["source"] = source

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d.columns:
            d[c] = pd.NA
        else:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    payload = d[["exchange", "symbol", "ts", "open", "high", "low", "close", "volume", "source"]].to_dict(orient="records")

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.bars_15m(exchange,symbol,ts,open,high,low,close,volume,source)
                values (:exchange,:symbol,:ts,:open,:high,:low,:close,:volume,:source)
                on conflict (exchange,symbol,ts) do update set
                  open=excluded.open,
                  high=excluded.high,
                  low=excluded.low,
                  close=excluded.close,
                  volume=excluded.volume,
                  source=excluded.source
                """
            ),
            payload,
        )
    return len(payload)


def main():
    cfg = load_config()
    engine = get_engine()
    provider = MarketDataProvider()

    interval = cfg["data"]["interval"]
    out = int(cfg["data"].get("outputsize", 400))
    xetr = cfg["data"]["exchange_primary_try"]

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        print(f"[INGEST] {name}")

        stooq_c = inst.get("stooq_candidates", []) or []
        yahoo_c = inst.get("yahoo_candidates", []) or []

        # intento XETR (TD->STOOQ->YAHOO)
        df, src, used = provider.fetch_best(
            td_symbol=inst["xetra_symbol"],
            td_exchange=xetr,
            interval=interval,
            outputsize=out,
            stooq_candidates=stooq_c,
            yahoo_candidates=yahoo_c,
        )
        if df is not None and not df.empty:
            if src == "stooq":
                ex, sym = "STOOQ", used
            elif src == "yahoo":
                ex, sym = "YAHOO", used
            else:
                ex, sym = xetr, inst["xetra_symbol"]

            n = upsert_bars(engine, ex, sym, df, source=src)
            print(f"[OK] {name}: {ex}:{sym} {n} filas ({src})")
            processed += 1
            continue

        # intento primary (TD->STOOQ->YAHOO)
        pex = inst.get("primary_exchange", "") or ""
        df2, src2, used2 = provider.fetch_best(
            td_symbol=inst["primary_symbol"],
            td_exchange=pex,
            interval=interval,
            outputsize=out,
            stooq_candidates=stooq_c,
            yahoo_candidates=yahoo_c,
        )
        if df2 is not None and not df2.empty:
            if src2 == "stooq":
                ex, sym = "STOOQ", used2
            elif src2 == "yahoo":
                ex, sym = "YAHOO", used2
            else:
                ex, sym = ("PRIMARY" if not pex else pex), inst["primary_symbol"]

            n = upsert_bars(engine, ex, sym, df2, source=src2)
            print(f"[OK] {name}: {ex}:{sym} {n} filas ({src2})")
            processed += 1
            continue

        print(f"[SKIP] {name}: sin datos")

    print(f"INGEST OK: {processed} inst")


if __name__ == "__main__":
    main()
