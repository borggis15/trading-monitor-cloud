import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Trading Monitor Pro", layout="wide")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    st.error("Falta DATABASE_URL en variables de entorno (Streamlit Secrets).")
    st.stop()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ----------------------------
# Utils
# ----------------------------
def _to_dt_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def fmt(x, nd=2, empty="—"):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return empty
        return f"{float(x):.{nd}f}"
    except Exception:
        return empty


def fmt_pct(x, nd=2, empty="—"):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return empty
        return f"{100*float(x):.{nd}f}%"
    except Exception:
        return empty


def action_badge(action: str):
    a = (action or "").upper()
    if a == "BUY":
        st.success("BUY")
    elif a == "SELL":
        st.error("SELL")
    else:
        st.info("HOLD")


def trend_label(ret_exp):
    """
    ret_exp es retorno esperado a horizonte (ej 10d). Lo convertimos en lenguaje:
    """
    if ret_exp is None or (isinstance(ret_exp, float) and pd.isna(ret_exp)):
        return "Tendencia: desconocida (sin estimación)"
    if ret_exp > 0.02:
        return "Tendencia: alcista (estimada)"
    if ret_exp < -0.02:
        return "Tendencia: bajista (estimada)"
    return "Tendencia: lateral (estimada)"


def explain_in_plain(action, ev_bps, proba_up, ret_exp, sharpe, max_dd, n_test):
    """
    Explicación humana, basada SOLO en métricas ya existentes.
    """
    a = (action or "HOLD").upper()

    # Mensaje principal (qué hacer)
    if a == "BUY":
        main = "✅ Señal de compra (según el modelo)."
        risk = "Compra con prudencia: entra por partes y usa stop."
    elif a == "SELL":
        main = "⛔ Señal de venta / evitar estar comprado."
        risk = "Si estás dentro, considera reducir o salir."
    else:
        main = "🟦 Mantener / no hacer nada ahora."
        risk = "Esperar una señal más clara."

    # Qué significa EV
    # ev_bps = puntos básicos: 100 bps = 1%
    ev_txt = "—"
    if ev_bps is not None and not (isinstance(ev_bps, float) and pd.isna(ev_bps)):
        ev_txt = f"{fmt(ev_bps,1)} bps (~{fmt(ev_bps/100,2)}%)"

    # Confianza (aprox) usando proba + robustez
    conf_bits = []
    if proba_up is not None and not (isinstance(proba_up, float) and pd.isna(proba_up)):
        conf_bits.append(f"Prob↑ {fmt(proba_up,2)}")
    if sharpe is not None and not (isinstance(sharpe, float) and pd.isna(sharpe)):
        conf_bits.append(f"Sharpe {fmt(sharpe,2)}")
    if n_test is not None and not (isinstance(n_test, float) and pd.isna(n_test)):
        conf_bits.append(f"n_test {int(n_test)}")
    if max_dd is not None and not (isinstance(max_dd, float) and pd.isna(max_dd)):
        conf_bits.append(f"MaxDD {fmt(max_dd,2)}")

    conf = " · ".join(conf_bits) if conf_bits else "—"

    # Texto final
    lines = [
        main,
        f"🎯 Beneficio esperado (EV): **{ev_txt}** en el horizonte del modelo.",
        f"📈 {trend_label(ret_exp)} · Retorno esperado: **{fmt_pct(ret_exp,2)}**",
        f"🧪 Calidad del modelo (último backtest): {conf}",
        f"⚠️ Gestión: {risk}",
    ]
    return "\n".join(lines)


# ----------------------------
# Data loaders
# ----------------------------
@st.cache_data(ttl=60)
def load_signals_current():
    q = """
    select
      asset_name, asset_id,
      exchange, symbol, ts,
      action, ev_bps,
      proba_up, ret_exp,
      size_eur, sl_price, tp_price,
      horizon, explanation, model_id
    from public.signals_current_by_asset
    order by asset_name;
    """
    df = pd.read_sql(text(q), engine)
    if not df.empty:
        df["ts"] = _to_dt_utc(df["ts"])
    return df


@st.cache_data(ttl=120)
def load_latest_metrics():
    q = """
    with ranked as (
      select *,
        row_number() over (partition by exchange, symbol order by trained_at desc) as rn
      from public.model_metrics
    )
    select exchange, symbol, model_id as metrics_model_id, trained_at,
           sharpe, max_dd, hit_rate, profit_factor, n_test, notes
    from ranked
    where rn=1
    """
    df = pd.read_sql(text(q), engine)
    if not df.empty:
        df["trained_at"] = _to_dt_utc(df["trained_at"])
    return df


@st.cache_data(ttl=60)
def load_bars(exchange, symbol, limit=800):
    q = """
    select ts, open, high, low, close, volume
    from public.bars_15m
    where exchange=:exchange and symbol=:symbol
    order by ts desc
    limit :limit
    """
    df = pd.read_sql(
        text(q),
        engine,
        params={"exchange": exchange, "symbol": symbol, "limit": int(limit)},
    )
    if df.empty:
        return df
    df["ts"] = _to_dt_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


@st.cache_data(ttl=60)
def load_signal_history(exchange, symbol, limit=300):
    q = """
    select ts, action, ev_bps, proba_up, ret_exp, size_eur, sl_price, tp_price, model_id
    from public.signals
    where exchange=:exchange and symbol=:symbol
    order by ts desc
    limit :limit
    """
    df = pd.read_sql(text(q), engine, params={"exchange": exchange, "symbol": symbol, "limit": int(limit)})
    if df.empty:
        return df
    df["ts"] = _to_dt_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


# ----------------------------
# Sidebar
# ----------------------------
st.title("📈 Trading Monitor Pro")

with st.sidebar:
    st.header("Controles")

    if st.button("🔄 Actualizar ahora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    actions_filter = st.multiselect(
        "Filtrar por acción",
        options=["BUY", "HOLD", "SELL"],
        default=["BUY", "HOLD", "SELL"],
    )

    st.subheader("Filtros de calidad")
    min_ev = st.slider("EV mínimo (bps)", min_value=-1000, max_value=1500, value=-1000, step=10)
    min_sharpe = st.slider("Sharpe mínimo", min_value=-2.0, max_value=15.0, value=-2.0, step=0.1)
    min_n_test = st.slider("n_test mínimo", min_value=0, max_value=500, value=0, step=10)

    st.subheader("Urgencia (banner)")
    urgent_ev = st.slider("EV urgencia (bps)", min_value=0, max_value=1500, value=800, step=10)
    urgent_proba = st.slider("Prob↑ urgencia", min_value=0.50, max_value=0.99, value=0.80, step=0.01)

    st.subheader("Orden")
    sort_by = st.selectbox("Ordenar ranking por", ["EV (bps)", "Sharpe", "Hit rate", "Max DD", "n_test"], index=0)

    st.caption("Tip: el dashboard muestra lo último de Supabase. Usa “Actualizar ahora” si acabas de correr workflows.")


# ----------------------------
# Load & merge
# ----------------------------
signals = load_signals_current()
metrics = load_latest_metrics()

if signals.empty:
    st.warning("No hay señales actuales todavía. Ejecuta score_15m.")
    st.stop()

df = signals.merge(metrics, on=["exchange", "symbol"], how="left")

for c in ["ev_bps", "proba_up", "ret_exp", "sharpe", "max_dd", "hit_rate", "profit_factor", "n_test"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df_view = df[df["action"].isin(actions_filter)].copy()
df_view = df_view[df_view["ev_bps"].fillna(-1e9) >= float(min_ev)]
df_view = df_view[df_view["sharpe"].fillna(-1e9) >= float(min_sharpe)]
df_view = df_view[df_view["n_test"].fillna(-1e9) >= float(min_n_test)]

sort_map = {"EV (bps)": "ev_bps", "Sharpe": "sharpe", "Hit rate": "hit_rate", "Max DD": "max_dd", "n_test": "n_test"}
sort_col = sort_map[sort_by]
ascending = True if sort_by == "Max DD" else False
df_rank = df_view.sort_values(sort_col, ascending=ascending if sort_by == "Max DD" else False).copy()

# ----------------------------
# Urgency banner
# ----------------------------
df_urgent = df.copy()
df_urgent = df_urgent[(df_urgent["action"] == "BUY") &
                      (df_urgent["ev_bps"].fillna(-1e9) >= urgent_ev) &
                      (df_urgent["proba_up"].fillna(0) >= urgent_proba)].copy()

if not df_urgent.empty:
    best = df_urgent.sort_values("ev_bps", ascending=False).iloc[0]
    st.warning(
        f"⚡ **ALERTA**: Señal BUY fuerte — **{best['asset_name']}** "
        f"({best['exchange']}:{best['symbol']}) · EV={fmt(best['ev_bps'],1)} bps · Prob↑={fmt(best['proba_up'],2)}",
        icon="⚠️"
    )

# KPIs
last_ts = df["ts"].max()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Última señal (UTC)", value=str(last_ts) if pd.notna(last_ts) else "—")
k2.metric("BUY", int((df["action"] == "BUY").sum()))
k3.metric("HOLD", int((df["action"] == "HOLD").sum()))
k4.metric("SELL", int((df["action"] == "SELL").sum()))

st.divider()

# ----------------------------
# Cards
# ----------------------------
st.subheader("📌 Señales actuales (interpretadas)")
cards = df_rank.sort_values("asset_name").to_dict(orient="records")

cols = st.columns(3)
for i, r in enumerate(cards):
    col = cols[i % 3]
    with col:
        with st.container(border=True):
            st.markdown(f"### {r.get('asset_name','—')}")
            st.caption(f"`{r.get('exchange','—')}:{r.get('symbol','—')}`  ·  modelo: `{r.get('model_id','—')}`")

            cA, cB = st.columns([1, 1], vertical_alignment="center")
            with cA:
                action_badge(r.get("action"))
            with cB:
                st.metric("EV (bps)", value=fmt(r.get("ev_bps"), 1))

            # Texto humano
            plain = explain_in_plain(
                action=r.get("action"),
                ev_bps=r.get("ev_bps"),
                proba_up=r.get("proba_up"),
                ret_exp=r.get("ret_exp"),
                sharpe=r.get("sharpe"),
                max_dd=r.get("max_dd"),
                n_test=r.get("n_test"),
            )
            st.write(plain)

            # Detalles opcionales SL/TP si existe
            if pd.notna(r.get("size_eur")):
                st.markdown("**Gestión sugerida (si BUY):**")
                st.write(
                    f"- Tamaño aprox: **{fmt(r.get('size_eur'),0)} €**\n"
                    f"- SL: **{fmt(r.get('sl_price'),2)}** · TP: **{fmt(r.get('tp_price'),2)}**"
                )

            # Botón para ver explicación técnica completa (sin ensuciar)
            with st.expander("Ver explicación técnica (texto completo)"):
                st.code((r.get("explanation") or "").strip(), language="text")

st.divider()

# ----------------------------
# Ranking table
# ----------------------------
st.subheader("📊 Ranking (filtrado)")
show_cols = [
    "asset_name", "exchange", "symbol", "action", "ev_bps",
    "proba_up", "ret_exp",
    "sharpe", "max_dd", "hit_rate", "n_test",
    "model_id"
]
st.dataframe(df_rank[show_cols], use_container_width=True)

st.divider()

# ----------------------------
# Detail
# ----------------------------
st.subheader("🔎 Detalle por activo")

names = df_rank["asset_name"].tolist() if not df_rank.empty else df["asset_name"].tolist()
if not names:
    st.info("Con los filtros actuales no hay activos. Ajusta filtros.")
    st.stop()

sel = st.selectbox("Selecciona un activo", options=names, index=0)
row = df[df["asset_name"] == sel].iloc[0]
ex = row["exchange"]
sym = row["symbol"]

tab1, tab2, tab3 = st.tabs(["Resumen", "Gráfico", "Histórico señales"])

with tab1:
    left, right = st.columns([1, 2], vertical_alignment="top")

    with left:
        st.markdown(f"## {row['asset_name']}")
        st.caption(f"`{ex}:{sym}`  ·  horizonte: `{row.get('horizon','—')}`")
        action_badge(row.get("action"))

        st.metric("EV (bps)", value=fmt(row.get("ev_bps"), 1))
        st.write(f"**Prob↑:** {fmt(row.get('proba_up'),2)}")
        st.write(f"**Retorno esperado:** {fmt_pct(row.get('ret_exp'),2)}")
        st.write(trend_label(row.get("ret_exp")))

        st.markdown("### Robustez (último backtest)")
        st.write(f"- Sharpe: **{fmt(row.get('sharpe'),2)}**")
        st.write(f"- Max DD: **{fmt(row.get('max_dd'),2)}**")
        st.write(f"- Hit rate: **{fmt(row.get('hit_rate'),2)}**")
        st.write(f"- n_test: **{int(row.get('n_test')) if pd.notna(row.get('n_test')) else '—'}**")
        st.write(f"- Entrenado (UTC): **{row.get('trained_at','—')}**")

    with right:
        st.markdown("### Explicación técnica completa")
        st.code((row.get("explanation") or "").strip(), language="text")

with tab2:
    bars = load_bars(ex, sym, limit=800)
    hist = load_signal_history(ex, sym, limit=300)

    if bars.empty:
        st.warning("No hay barras para graficar.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bars["ts"], y=bars["close"], mode="lines", name="Close"))

        if not hist.empty:
            for act, marker in [("BUY", "triangle-up"), ("SELL", "triangle-down"), ("HOLD", "circle")]:
                sub = hist[hist["action"] == act]
                if not sub.empty:
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

        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    hist = load_signal_history(ex, sym, limit=300)
    if hist.empty:
        st.info("No hay histórico de señales aún.")
    else:
        show = hist[["ts", "action", "ev_bps", "proba_up", "ret_exp", "size_eur", "sl_price", "tp_price", "model_id"]].copy()
        st.dataframe(show.tail(300), use_container_width=True)

st.caption("Nota: Las señales son una ayuda cuantitativa; no garantizan beneficios. Revisa siempre riesgo y contexto.")
