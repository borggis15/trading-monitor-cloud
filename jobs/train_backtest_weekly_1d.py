import pandas as pd
from sqlalchemy import text

def read_bars_1d(engine, exchange: str, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            select ts, open, high, low, close, volume
            from public.bars_1d
            where exchange=:exchange and symbol=:symbol
            order by ts
            """
        ),
        engine,
        params={"exchange": exchange, "symbol": symbol},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    return df


def resolve_series_1d(engine, inst: dict) -> tuple[str | None, str | None]:
    """
    Prefer STOOQ (gratis, estable) y fallback a YAHOO (gratis via yfinance).
    """
    candidates: list[tuple[str, str]] = []

    # 1) STOOQ candidates
    for s in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", s))

    # 2) YAHOO candidates
    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))

    # Opcional: si quieres, podrías añadir PRIMARY aquí, pero no lo recomiendo para 1D gratis.
    # candidates.append(("PRIMARY", inst.get("primary_symbol")))

    for ex, sym in candidates:
        if not sym:
            continue
        bars = read_bars_1d(engine, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {inst['name']} using {ex}:{sym} bars={len(bars)}")
            return ex, sym

    return None, None
