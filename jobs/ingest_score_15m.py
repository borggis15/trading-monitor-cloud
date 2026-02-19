from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone

from core.config import load_config
from core.db import get_engine
from core.providers import MarketDataProvider
from core.features import compute_features
from core.ml import train_models, predict
from core.risk import size_from_atr, ev_bps


def _normalize_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame tenga:
      - índice datetime UTC
      - nombre de índice = 'ts'
    Compatible con Twelve Data (ya viene bien) y Stooq (suele venir con índice fecha sin nombre).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Si ya tiene índice datetime, perfecto
    if not isinstance(out.index, pd.DatetimeIndex):
        # Si existe una columna con fecha típica, la usamos
        for col in ["ts", "datetime", "date", "Date", "time", "Time"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
                out = out.dropna(subset=[col]).set_index(col)
                break

    # Forzamos DatetimeIndex
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

    d = df.copy().reset_index()  # ahora seguro existe columna ts
    d["ts"] = pd.to_datetime(d["ts"], utc=True)

    d["exchange"] = exchange
    d["symbol"] = symbol
    d["source"] = source

    # asegurar columnas
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d.columns:
            d[c] = pd.NA

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into bars_15m(exchange,symbol,ts,open,high,low,close,volume,source)
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


def read_bars(engine, exchange: str, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            select ts, open, high, low, close, volume
            from bars_15m
            where exchange=:exchange and symbol=:symbol
            order by ts
            """
        ),
        engine,
        params={"exchange": exchange, "symbol": symbol},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def upsert_features(engine, exchange: str, symbol: str, feat: pd.DataFrame):
    if feat is None or feat.empty:
        return 0

    feat = _normalize_ts_index(feat)
    if feat is None or feat.empty:
        return 0

    f = feat.copy().reset_index()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f["exchange"] = exchange
    f["symbol"] = symbol

    cols = ["exchange", "symbol", "ts", "rsi", "ema_fast", "ema_slow", "atr", "zscore", "ret_fwd"]
    for c in cols:
        if c not in f.columns:
            f[c] = pd.NA

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into features_15m(exchange,symbol,ts,rsi,ema_fast,ema_slow,atr,zscore,ret_fwd)
                values (:exchange,:symbol,:ts,:rsi,:ema_fast,:ema_slow,:atr,:zscore,:ret_fwd)
                on conflict (exchange,symbol,ts) do update set
                  rsi=excluded.rsi,
                  ema_fast=excluded.ema_fast,
                  ema_slow=excluded.ema_slow,
                  atr=excluded.atr,
                  zscore=excluded.zscore,
                  ret_fwd=excluded.ret_fwd
                """
            ),
            f[cols].to_dict(orient="records"),
        )
    return len(f)


def upsert_signal(engine, row: dict):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into signals(exchange,symbol,ts,action,proba_up,ret_exp,risk_est,ev_bps,size_eur,sl_price,tp_price,horizon,explanation,model_id)
                values (:exchange,:symbol,:ts,:action,:proba_up,:ret_exp,:risk_est,:ev_bps,:size_eur,:sl_price,:tp_price,:horizon,:explanation,:model_id)
                on conflict (exchange,symbol,ts) do update set
                  action=excluded.action,
                  proba_up=excluded.proba_up,
                  ret_exp=excluded.ret_exp,
                  risk_est=excluded.risk_est,
                  ev_bps=excluded.ev_bps,
                  size_eur=excluded.size_eur,
                  sl_price=excluded.sl_price,
                  tp_price=excluded.tp_price,
                  horizon=excluded.horizon,
                  explanation=excluded.explanation,
                  model_id=excluded.model_id
                """
            ),
            row,
        )


def fetch_best(provider: MarketDataProvider, cfg: dict, inst: dict):
    interval = cfg["data"]["interval"]
    out = int(cfg["data"].get("outputsize", 400))
    xetr = cfg["data"]["exchange_primary_try"]

    df = provider.fetch(
        symbol=inst["xetra_symbol"],
        exchange=xetr,
        interval=interval,
        outputsize=out,
        stooq_candidates=inst.get("stooq_candidates", []),
    )
    if df is not None and not df.empty:
        return xetr, inst["xetra_symbol"], df, "mixed"

    print(f"[WARN] Sin datos para {xetr}:{inst['xetra_symbol']}. Probando primary...")

    pex = inst.get("primary_exchange", "") or ""
    df2 = provider.fetch(
        symbol=inst["primary_symbol"],
        exchange=pex,
        interval=interval,
        outputsize=out,
        stooq_candidates=inst.get("stooq_candidates", []),
    )
    if df2 is not None and not df2.empty:
        ex_label = pex if pex else "PRIMARY"
        return ex_label, inst["primary_symbol"], df2, "mixed"

    return None, None, pd.DataFrame(), ""


def main():
    cfg = load_config()
    provider = MarketDataProvider()
    engine = get_engine()

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        print(f"[START] {name}")

        ex, sym, df, source = fetch_best(provider, cfg, inst)
        if df is None or df.empty or ex is None or sym is None:
            print(f"[SKIP] {name}: sin datos")
            continue

        df = _normalize_ts_index(df)
        n = upsert_bars(engine, ex, sym, df, source=source)
        print(f"[INFO] {name}: bars {ex}:{sym} -> {n} filas")

        bars = read_bars(engine, ex, sym)
        if bars is None or bars.empty:
            print(f"[SKIP] {name}: bars vacío tras upsert")
            continue

        feat = compute_features(bars, horizon_bars=int(cfg["ml"]["horizon_bars"]))
        if feat is None or feat.empty:
            print(f"[SKIP] {name}: features vacío")
            continue

        upsert_features(engine, ex, sym, feat)

        clf, reg, train_rows = train_models(
            feat,
            model_type=cfg["ml"]["model_type"],
            min_rows=int(cfg["ml"]["min_train_rows"]),
        )
        proba, ret_exp = predict(clf, reg, feat)

        risk_est = None
        last = feat.dropna().tail(1)
        if not last.empty and "atr" in last.columns:
            atr = float(last["atr"].iloc[0])
            price = float(bars["close"].iloc[-1])
            if price > 0:
                risk_est = atr / price

        ev = ev_bps(
            ret_exp,
            risk_est,
            float(cfg["backtest"]["fee_bps"]),
            float(cfg["backtest"]["slippage_bps"]),
        )

        action = "HOLD"
        expl = [name, f"Fuente={ex}:{sym}", f"interval={cfg['data']['interval']}"]
        if proba is not None:
            expl.append(f"Prob(subida)= {proba:.2f}")
        if ret_exp is not None:
            expl.append(f"Ret exp ≈ {ret_exp*100:.2f}%")
        if ev is not None:
            expl.append(f"EV ≈ {ev:.1f} bps")

        if proba is not None and ret_exp is not None and ev is not None:
            if proba > 0.60 and ev > 2.0:
                action = "BUY"
            elif proba < 0.45 and ev < -2.0:
                action = "SELL"

        size_eur = sl = tp = None
        if action == "BUY" and not last.empty and "atr" in last.columns:
            atr = float(last["atr"].iloc[0])
            price = float(bars["close"].iloc[-1])
            size_eur, sl = size_from_atr(
                capital_eur=float(cfg["signals"]["capital_eur"]),
                risk_pct=float(cfg["signals"]["risk_per_trade_pct"]),
                max_pos_pct=float(cfg["signals"]["max_position_pct"]),
                price=price,
                atr=atr,
                sl_atr_mult=float(cfg["signals"]["sl_atr_mult"]),
            )
            tp = float(price + float(cfg["signals"]["tp_atr_mult"]) * atr)

        ts = bars.index[-1] if len(bars) else datetime.now(timezone.utc)

        upsert_signal(engine, {
            "exchange": ex,
            "symbol": sym,
            "ts": ts,
            "action": action,
            "proba_up": proba,
            "ret_exp": ret_exp,
            "risk_est": risk_est,
            "ev_bps": ev,
            "size_eur": size_eur,
            "sl_price": sl,
            "tp_price": tp,
            "horizon": f"{int(cfg['ml']['horizon_bars'])}d",
            "explanation": " | ".join(expl),
            "model_id": f"{cfg['ml']['model_type']}_{cfg['ml']['horizon_bars']}",
        })

        processed += 1
        print(f"[END] {name}: señal={action}, train_rows={train_rows}")

    print(f"OK: {processed} symbols processed")


if __name__ == "__main__":
    main()
