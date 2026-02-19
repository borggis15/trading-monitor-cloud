import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Trading Monitor", layout="wide")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    st.error("Falta DATABASE_URL en variables de entorno.")
    st.stop()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=60)
def load_signals_current():
    q = """
    select asset_name, exchange, symbol, action, ev_bps, explanation, ts
    from public.signals_current_by_asset
    order by asset_name;
    """
    df = pd.read_sql(text(q), engine)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_latest_metrics():
    q = """
    with ranked as (
      select *,
        row_number() over (partition by exchange, symbol order by trained_at desc) as rn
      from public.model_metrics
    )
    select exchange, symbol, model_id, trained_at, sharpe, max_dd, hit_rate, profit_factor, n_test, notes
    from ranked
    where rn=1
    """
    df = pd.read_sql(text(q), engine)
    if not df.empty:
        df["trained_at"] = pd.to_datetime(df["trained_at"], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=60)
def load_bars(exchange, symbol, limit=400):
    q = """
    select ts, close
    from public.bars_15m
    where exchange=:exchange and symbol=:symbol
    order by ts desc
    limit :limit
    """
    df = pd.read_sql(text(q), engine, params={"exchange": exchange, "symbol": symbol, "limit": int(limit)})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


@st.cache_data(ttl=60)
def load_signal_history(exchange, symbol, limit=200):
    q = """
    select ts, action, ev_bps, proba_up, ret_exp, size_eur, sl_price, tp_price, explanation, model_id
    from public.signals
    where exchange=:exchange and symbol=:symbol
    order by ts desc
    limit :limit
    """
    df = pd.read_sql(text(q), engine, params={"exchange": exchange, "symbol": symbol, "limit": int(limit)})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


def parse_explanation(expl: str) -> dict:
    # explanation es un string con " | " separando tokens. Lo convertimos en dict útil.
    out = {}
    if not expl:
        return out
    parts = [p.strip() for p in expl.split("|")]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            # el primer token suele ser el nombre
            if "name" not in out and p:
                out["name"] = p
    return out


def badge_action(action: str):
    a = (action or "").upper()
    if a == "BUY":
        st.success("BUY")
    elif a == "SELL":
        st.error("SELL")
    else:
        st.info("HOLD")


st.title("📈 Trading Monitor (6 activos)")

signals = load_signals_current()
metrics = load_latest_metrics()

if signals.empty:
    st.warning("No hay señales actuales todavía. Ejecuta score_15m.")
    st.stop()

# Enriquecemos con métricas
merged = signals.merge(metrics, on=["exchange", "symbol"], how="left")

# Ranking simple: EV desc, y si hay HOLD con EV alto que se vea.
merged["ev_bps_num"] = pd.to_numeric(merged["ev_bps"], errors="coerce")
ranked = merged.sort_values(["ev_bps_num"], ascending=False)

# Barra superior: “mejor oportunidad”
top = ranked.iloc[0]
c1, c2, c3, c4 = st.columns([2, 1, 1, 2], vertical_alignment="center")
with c1:
    st.subheader("Mejor oportunidad actual")
    st.write(f"**{top['asset_name']}** ({top['exchange']}:{top['symbol']})")
with c2:
    badge_action(top["action"])
with c3:
    st.metric("EV (bps)", value=f"{top['ev_bps_num']:.2f}" if pd.notna(top["ev_bps_num"]) else "—")
with c4:
    st.caption("EV (bps) = retorno esperado neto en puntos básicos (≈ 0.01%).\n"
               "Un EV alto NO garantiza beneficio: se filtra además por robustez (Sharpe/DD/hit-rate).")

st.divider()

# Tabla resumen
st.subheader("Resumen (ranking por EV)")
show_cols = ["asset_name", "exchange", "symbol", "action", "ev_bps", "sharpe", "max_dd", "hit_rate", "n_test", "model_id"]
st.dataframe(ranked[show_cols], use_container_width=True)

st.divider()

# Selector de activo
names = ranked["asset_name"].tolist()
sel = st.selectbox("Selecciona un activo", options=names, index=0)
row = ranked[ranked["asset_name"] == sel].iloc[0]
ex = row["exchange"]
sym = row["symbol"]

left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader(f"{row['asset_name']}")
    st.write(f"**Serie:** `{ex}:{sym}`")
    badge_action(row["action"])

    ev = row["ev_bps_num"]
    st.metric("EV (bps)", value=f"{ev:.2f}" if pd.notna(ev) else "—")

    # métricas
    st.markdown("### Robustez (último backtest)")
    st.write(f"- **Sharpe:** {row['sharpe'] if pd.notna(row['sharpe']) else '—'}")
    st.write(f"- **Max DD:** {row['max_dd'] if pd.notna(row['max_dd']) else '—'}")
    st.write(f"- **Hit rate:** {row['hit_rate'] if pd.notna(row['hit_rate']) else '—'}")
    st.write(f"- **n_test:** {row['n_test'] if pd.notna(row['n_test']) else '—'}")
    st.write(f"- **Modelo:** {row['model_id'] if pd.notna(row['model_id']) else '—'}")

    st.markdown("### Explicación")
    st.caption(row.get("explanation", "") or "")

with right:
    st.subheader("Precio + señales recientes")

    bars = load_bars(ex, sym, limit=400)
    hist = load_signal_history(ex, sym, limit=200)

    if bars.empty:
        st.warning("No hay barras para graficar.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bars["ts"], y=bars["close"], mode="lines", name="Close"))

        if not hist.empty:
            # Marcadores por acción
            for act, marker in [("BUY", "triangle-up"), ("SELL", "triangle-down"), ("HOLD", "circle")]:
                sub = hist[hist["action"] == act]
                if not sub.empty:
                    # para marcar en el precio usamos el close más cercano
                    tmp = pd.merge_asof(
                        sub[["ts", "action"]].sort_values("ts"),
                        bars[["ts", "close"]].sort_values("ts"),
                        on="ts",
                        direction="nearest",
                    )
                    fig.add_trace(go.Scatter(
                        x=tmp["ts"],
                        y=tmp["close"],
                        mode="markers",
                        name=act,
                        marker_symbol=marker,
                        marker_size=10,
                    ))

        fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Histórico de señales (últimas 200)")
    if hist.empty:
        st.info("No hay histórico de señales aún.")
    else:
        show = hist[["ts", "action", "ev_bps", "proba_up", "ret_exp", "size_eur", "sl_price", "tp_price", "model_id"]].copy()
        st.dataframe(show.tail(200), use_container_width=True)
