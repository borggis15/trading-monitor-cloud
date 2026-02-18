from __future__ import annotations

import time
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone

from core.config import load_config
from core.db import get_engine
from core.providers import TwelveDataProvider
from core.features import compute_features
from core.ml import train_models, predict
from core.risk import size_from_atr, ev_bps


# --- Timeout duro por símbolo (Linux/GitHub runner) ---
import signal

class TimeoutError(Exception):
    pass

def _alarm_handler(signum, frame):
    raise TimeoutError("Timeout por símbolo")

def run_with_timeout(seconds: int, fn, *args, **kwargs):
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)


def upsert_bars(engine, exchange: str, symbol: str, df: pd.DataFrame, source: str):
    if df is None or df.empty:
        return 0

    d = df.copy().reset_index()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["exchange"] = exchange
    d["symbol"] = symbol
    d["source"] = source

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
            d.to_dict(orient="records"),
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

    f = feat.copy().reset_index()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f["exchange"] = exchange
    f["symbol"] = symbol

    cols = ["exchange", "symbol", "ts", "rsi", "ema_fast", "ema_slow", "atr", "zscore", "ret_fwd"]

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


def fetch_with_fallback(provider: TwelveDataProvider, cfg: dict, inst: dict):
    xetr = cfg["data"]["exchange_primary_try"]
    interval = cfg["data"]["interval"]

    df = provider.fetch_15m(symbol=inst["xetra_symbol"], exchange=xetr, interval=interval, outputsize=800)
    if df is not None and not df.empty:
        return xetr, inst["xetra_symbol"], df

    print(f"[WARN] Sin datos para {xetr}:{inst['xetra_symbol']}. Probando fallback primary...")

    df2 = provider.fetch_15m(symbol=inst["primary_symbol"], exchange="", interval=interval, outputsize=800)
    if df2 is not None and not df2.empty:
        return "PRIMARY", inst["primary_symbol"], df2

    return None, None, pd.DataFrame()


def process_one(engine, provider, cfg, inst):
    name = inst["name"]
    t0 = time.time()
    print(f"[START] {name}")

    ex, sym, df = fetch_with_fallback(provider, cfg, inst)
    if df is None or df.empty or ex is None or sym is None:
        print(f"[SKIP] {name}: sin datos")
        return False

    n = upsert_bars(engine, ex, sym, df, source="twelvedata")
    print(f"[INFO] {name}: upsert bars {ex}:{sym} -> {n} filas")

    bars = read_bars(engine, ex, sym)
    if bars is None or bars.empty:
        print(f"[SKIP] {name}: bars vacío tras upsert")
        return False

    feat = compute_features(bars, horizon_bars=int(cfg["ml"]["horizon_bars"]))
    if feat is None or feat.empty:
        print(f"[SKIP] {name}: features vacío")
        return False

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
    expl = [name, f"Fuente={ex}:{sym}"]
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
        "horizon": f"{int(cfg['ml']['horizon_bars']) * 15}min",
        "explanation": " | ".join(expl),
        "model_id": f"{cfg['ml']['model_type']}_{cfg['ml']['horizon_bars']}",
    })

    dt = time.time() - t0
    print(f"[END] {name} en {dt:.1f}s (train_rows={train_rows})")
    return True


def main():
    cfg = load_config()
    provider = TwelveDataProvider()
    engine = get_engine()

    ok = 0
    for inst in cfg["universe"]:
        try:
            # ✅ 60s máx por símbolo: si se cuelga, lo saltamos
            if run_with_timeout(60, process_one, engine, provider, cfg, inst):
                ok += 1
        except TimeoutError:
            print(f"[TIMEOUT] {inst['name']} >60s, se omite.")
        except Exception as e:
            print(f"[ERROR] {inst['name']}: {e}")

    print(f"OK: {ok} symbols processed")


if __name__ == "__main__":
    main()
