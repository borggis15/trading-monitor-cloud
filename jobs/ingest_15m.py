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
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["exchange"] = exchange
    d["symbol"] = symbol
    d["source"] = source

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d.columns:
            d[c] = pd.NA

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
            d[["exchange","symbol","ts","open","high","low","close","volume","source"]].to_dict(orient="records"),
        )
    return len(d)


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

        # intento XETR (TD->STOOQ)
        df, src, used = provider.fetch_best(
            td_symbol=inst["xetra_symbol"],
            td_exchange=xetr,
            interval=interval,
            outputsize=out,
            stooq_candidates=inst.get("stooq_candidates", []),
        )
        if df is not None and not df.empty:
            ex = "STOOQ" if src == "stooq" else xetr
            sym = used if src == "stooq" else inst["xetra_symbol"]
            n = upsert_bars(engine, ex, sym, df, source=src)
            print(f"[OK] {name}: {ex}:{sym} {n} filas")
            processed += 1
            continue

        # intento primary (TD->STOOQ)
        pex = inst.get("primary_exchange", "") or ""
        df2, src2, used2 = provider.fetch_best(
            td_symbol=inst["primary_symbol"],
            td_exchange=pex,
            interval=interval,
            outputsize=out,
            stooq_candidates=inst.get("stooq_candidates", []),
        )
        if df2 is not None and not df2.empty:
            ex = "STOOQ" if src2 == "stooq" else ("PRIMARY" if not pex else pex)
            sym = used2 if src2 == "stooq" else inst["primary_symbol"]
            n = upsert_bars(engine, ex, sym, df2, source=src2)
            print(f"[OK] {name}: {ex}:{sym} {n} filas")
            processed += 1
            continue

        print(f"[SKIP] {name}: sin datos")

    print(f"INGEST OK: {processed} inst")


if __name__ == "__main__":
    main()
