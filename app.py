import os
import re
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


def _num(x, default=None):
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def action_color(action: str) -> str:
    a = (action or "").upper()
    if a == "BUY":
        return "#1f8a3b"
    if a == "SELL":
        return "#b42318"
    return "#0b5cab"


def parse_explanation(expl: str) -> dict:
    """
    explanation típico:
    Name | serie=XETR:LLY | model=hgb_10d | proba=0.191 | ret_exp=-0.0515 | ev_bps=-595.32 | robust(...) | buy_gate=...
    """
    out = {}
    if not expl:
        return out
    parts = [p.strip() for p in expl.split("|")]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            if "name" not in out and p:
                out["name"] = p

    # robust(...) if present
    m = re.search(r"robust\((.*?)\)", expl)
    if m:
        out["robust_raw"] = m.group(1)

    return out


def summarize_card(row: dict) -> str:
    """
    Resumen claro para el reverso:
    - Tendencia (simple) basada en ret_exp y proba
    - Acción recomendada (BUY/SELL/HOLD) + por qué (EV/robustez/gate)
    """
    action = (row.get("action") or "HOLD").upper()
    ev = _num(row.get("ev_bps"))
    proba = _num(row.get("proba_up"))
    ret_exp = _num(row.get("ret_exp"))

    trend = "Lateral"
    if ret_exp is not None:
        if ret_exp > 0.02:
            trend = "Alcista"
        elif ret_exp < -0.02:
            trend = "Bajista"
        else:
            trend = "Lateral"

    # Decisión explicada
    why = []
    if ev is not None:
        why.append(f"EV {ev:.1f} bps")
    if proba is not None:
        why.append(f"Prob↑ {proba:.2f}")
    sh = _num(row.get("sharpe"))
    n_test = row.get("n_test")
    if sh is not None:
        why.append(f"Sharpe {sh:.2f}")
    if pd.notna(n_test):
        why.append(f"n_test {int(n_test)}")

    # Mensaje final según acción
    if action == "BUY":
        rec = "Comprar (entrada escalonada y respetar SL/TP)."
    elif action == "SELL":
        rec = "Vender / reducir exposición (evitar estar comprado)."
    else:
        rec = "Mantener (esperar mejor punto/confirmación)."

    return f"**Tendencia:** {trend}\n\n**Recomendación:** {rec}\n\n**Claves:** " + " · ".join(why)


# ----------------------------
# Data loaders (cached)
# ----------------------------
@st.cache_data(ttl=60)
def load_signals_current():
    q = """
    select
      asset_name, asset_id,
      exchange, symbol, ts,
      action, ev_bps,
      proba_up, ret_exp, risk_est,
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
    select exchange, symbol, model_id as metrics_model_id, trained_at, sharpe, max_dd, hit_rate, profit_factor, n_test, notes
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
    select ts, action, ev_bps, proba_up, ret_exp, size_eur, sl_price, tp_price, explanation, model_id
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
# CSS for 3D flip cards
# ----------------------------
CARD_CSS = """
<style>
.flip-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 1200px) { .flip-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 800px)  { .flip-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); } }

.flip-wrap { perspective: 1000px; }
.flip-card {
  position: relative;
  width: 100%;
  height: 235px;              /* tamaño fijo: NO crece al girar */
  transform-style: preserve-3d;
  transition: transform 650ms cubic-bezier(.2,.8,.2,1);
  border-radius: 18px;
}
.flip-card.flipped { transform: rotateY(180deg); }

.flip-face {
  position: absolute;
  inset: 0;
  backface-visibility: hidden;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  padding: 14px 14px 12px 14px;
  overflow: hidden;
}

.front {
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
}
.back {
  transform: rotateY(180deg);
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
}

.row-top { display:flex; align-items:flex-start; justify-content:space-between; gap:10px; }
.title { font-size: 18px; font-weight: 700; line-height: 1.15; margin: 0; }
.subtitle { font-size: 12px; opacity: 0.75; margin-top: 6px; }
.pill {
  font-size: 12px;
  font-weight: 800;
  padding: 6px 10px;
  border-radius: 999px;
  color: white;
  white-space: nowrap;
}
.metrics { margin-top: 12px; font-size: 13px; line-height: 1.35; }
.kv { display:flex; justify-content:space-between; gap:10px; margin-top: 6px; }
.k { opacity: 0.75; }
.v { font-weight: 700; }

.hr { height:1px; background: rgba(255,255,255,0.10); margin: 10px 0; }

.back h4 { margin: 0 0 8px 0; font-size: 14px; }
.back .text { font-size: 13px; line-height: 1.35; opacity: 0.92; }

.note { font-size: 11px; opacity: 0.65; position:absolute; bottom:10px; left:14px; right:14px; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


# ----------------------------
# Sidebar controls
# ----------------------------
st.title("📈 Trading Monitor Pro")

if "flip" not in st.session_state:
    st.session_state.flip = {}  # key: asset_id (or exchange:symbol), value: bool

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

    st.subheader("Visual")
    sort_by = st.selectbox("Ordenar ranking por", ["EV (bps)", "Sharpe", "Hit rate", "Max DD", "n_test"], index=0)

    st.caption("Tip: el botón “Ver explicación” gira la tarjeta (3D).")


# ----------------------------
# Load & merge
# ----------------------------
signals = load_signals_current()
metrics = load_latest_metrics()

if signals.empty:
    st.warning("No hay señales actuales todavía. Ejecuta score_15m.")
    st.stop()

df = signals.merge(metrics, on=["exchange", "symbol"], how="left")

# numeric coercions
for c in ["ev_bps", "proba_up", "ret_exp", "sharpe", "max_dd", "hit_rate", "profit_factor", "n_test"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# apply filters
df_view = df[df["action"].isin(actions_filter)].copy()
df_view = df_view[df_view["ev_bps"].fillna(-1e9) >= float(min_ev)]
df_view = df_view[df_view["sharpe"].fillna(-1e9) >= float(min_sharpe)]
df_view = df_view[df_view["n_test"].fillna(-1e9) >= float(min_n_test)]

# Sorting
sort_map = {
    "EV (bps)": "ev_bps",
    "Sharpe": "sharpe",
    "Hit rate": "hit_rate",
    "Max DD": "max_dd",
    "n_test": "n_test",
}
sort_col = sort_map[sort_by]
ascending = True if sort_by == "Max DD" else False  # max_dd: menos negativo es mejor
df_rank = df_view.sort_values(sort_col, ascending=ascending if sort_by == "Max DD" else False).copy()

# ----------------------------
# Urgency banner
# ----------------------------
# “Urgente” si hay BUY con EV y proba altos (y señal reciente)
df_urgent = df.copy()
df_urgent["ev_bps"] = pd.to_numeric(df_urgent["ev_bps"], errors="coerce")
df_urgent["proba_up"] = pd.to_numeric(df_urgent["proba_up"], errors="coerce")
df_urgent = df_urgent[(df_urgent["action"] == "BUY") &
                      (df_urgent["ev_bps"].fillna(-1e9) >= urgent_ev) &
                      (df_urgent["proba_up"].fillna(0) >= urgent_proba)].copy()

if not df_urgent.empty:
    # elegimos la mejor por EV
    best = df_urgent.sort_values("ev_bps", ascending=False).iloc[0]
    st.warning(
        f"⚠️ **ALERTA**: Señal BUY fuerte ahora mismo — **{best['asset_name']}** "
        f"({best['exchange']}:{best['symbol']}) · EV={fmt(best['ev_bps'],2)} bps · Prob↑={fmt(best['proba_up'],2)}",
        icon="⚡"
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
# Flip cards section
# ----------------------------
st.subheader("📌 Señales actuales (tarjetas giratorias)")
cards = df_rank.sort_values("asset_name").to_dict(orient="records")

# grid container
st.markdown('<div class="flip-grid">', unsafe_allow_html=True)

for r in cards:
    asset_name = r.get("asset_name") or "—"
    exchange = r.get("exchange") or "—"
    symbol = r.get("symbol") or "—"
    action = (r.get("action") or "HOLD").upper()
    model_id = r.get("model_id") or "—"
    ev = r.get("ev_bps")
    proba = r.get("proba_up")
    ret_exp = r.get("ret_exp")

    sharpe = r.get("sharpe")
    max_dd = r.get("max_dd")
    hit_rate = r.get("hit_rate")
    n_test = r.get("n_test")

    expl = (r.get("explanation") or "").strip()
    card_key = str(r.get("asset_id") or f"{exchange}:{symbol}")

    flipped = bool(st.session_state.flip.get(card_key, False))
    flip_class = "flipped" if flipped else ""
    pill_bg = action_color(action)

    # HTML card (front/back)
    back_text = summarize_card(r)
    back_text_html = (
        back_text.replace("\n\n", "<br><br>")
        .replace("\n", "<br>")
        .replace("**", "<b>").replace("<b>", "<b>").replace("</b>", "</b>")
    )
    # The ** conversion above is simplistic; we will just render in code-like later via st.caption if needed.
    # We'll keep back as plain HTML with <b> markers minimal:
    back_text_html = back_text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    back_text_html = back_text_html.replace("**", "<b>", 1).replace("**", "</b>", 1)  # best effort

    front_html = f"""
    <div class="flip-wrap">
      <div class="flip-card {flip_class}">
        <div class="flip-face front">
          <div class="row-top">
            <div>
              <div class="title">{asset_name}</div>
              <div class="subtitle">{exchange}:{symbol} · modelo {model_id}</div>
            </div>
            <div class="pill" style="background:{pill_bg};">{action}</div>
          </div>

          <div class="hr"></div>

          <div class="metrics">
            <div class="kv"><div class="k">EV (bps)</div><div class="v">{fmt(ev,2)}</div></div>
            <div class="kv"><div class="k">Prob↑</div><div class="v">{fmt(proba,3)}</div></div>
            <div class="kv"><div class="k">Ret exp</div><div class="v">{fmt(ret_exp,4)}</div></div>
          </div>

          <div class="hr"></div>

          <div class="metrics">
            <div class="kv"><div class="k">Sharpe</div><div class="v">{fmt(sharpe,2)}</div></div>
            <div class="kv"><div class="k">Max DD</div><div class="v">{fmt(max_dd,2)}</div></div>
            <div class="kv"><div class="k">Hit</div><div class="v">{fmt(hit_rate,2)}</div></div>
            <div class="kv"><div class="k">n_test</div><div class="v">{int(n_test) if pd.notna(n_test) else "—"}</div></div>
          </div>

          <div class="note">Pulsa “Ver explicación” para girar la tarjeta.</div>
        </div>

        <div class="flip-face back">
          <div class="row-top">
            <div>
              <div class="title">{asset_name}</div>
              <div class="subtitle">Resumen interpretado</div>
            </div>
            <div class="pill" style="background:{pill_bg};">{action}</div>
          </div>

          <div class="hr"></div>

          <h4>Qué significa</h4>
          <div class="text">{back_text_html}</div>

          <div class="hr"></div>

          <h4>Detalles técnicos (resumen)</h4>
          <div class="text">{(expl[:220] + "…") if len(expl) > 220 else expl}</div>

          <div class="note">Pulsa “Volver” para ver métricas.</div>
        </div>
      </div>
    </div>
    """

    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown(front_html, unsafe_allow_html=True)

    # Buttons under card (do rerun + flip state; animation is CSS)
    b1, b2 = st.columns([1, 1], vertical_alignment="center")
    with b1:
        if st.button("Ver explicación" if not flipped else "Volver", key=f"flipbtn_{card_key}", use_container_width=True):
            st.session_state.flip[card_key] = not flipped
            st.rerun()
    with b2:
        # Quick jump to detail selection
        if st.button("Abrir detalle", key=f"detailbtn_{card_key}", use_container_width=True):
            st.session_state["selected_asset_name"] = asset_name
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

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
# Detail section
# ----------------------------
st.subheader("🔎 Detalle por activo")

names = df_rank["asset_name"].tolist() if not df_rank.empty else df["asset_name"].tolist()
if not names:
    st.info("Con los filtros actuales no hay activos. Ajusta filtros.")
    st.stop()

default_sel = 0
if "selected_asset_name" in st.session_state and st.session_state["selected_asset_name"] in names:
    default_sel = names.index(st.session_state["selected_asset_name"])

sel = st.selectbox("Selecciona un activo", options=names, index=default_sel)
row = df[df["asset_name"] == sel].iloc[0]
ex = row["exchange"]
sym = row["symbol"]

tab1, tab2, tab3 = st.tabs(["Resumen", "Gráfico", "Histórico señales"])

with tab1:
    left, right = st.columns([1, 2], vertical_alignment="top")

    with left:
        st.markdown(f"## {row['asset_name']}")
        st.caption(f"`{ex}:{sym}`  ·  horizonte: `{row.get('horizon','—')}`")

        # Action pill
        st.markdown(
            f"<div class='pill' style='display:inline-block;background:{action_color(row.get('action'))};'>"
            f"{(row.get('action') or 'HOLD').upper()}</div>",
            unsafe_allow_html=True
        )

        st.metric("EV (bps)", value=fmt(row.get("ev_bps"), 2))
        st.write(f"**Prob↑:** {fmt(row.get('proba_up'),3)}")
        st.write(f"**Retorno esperado:** {fmt(row.get('ret_exp'),4)}")

        if pd.notna(row.get("size_eur")):
            st.markdown("### Gestión propuesta (si BUY)")
            st.write(f"**Tamaño (€):** {fmt(row.get('size_eur'),0)}")
            st.write(f"**Stop (SL):** {fmt(row.get('sl_price'),2)}")
            st.write(f"**Take profit (TP):** {fmt(row.get('tp_price'),2)}")

        st.markdown("### Robustez (último entrenamiento)")
        st.write(f"- Sharpe: **{fmt(row.get('sharpe'),2)}**")
        st.write(f"- Max DD: **{fmt(row.get('max_dd'),2)}**")
        st.write(f"- Hit rate: **{fmt(row.get('hit_rate'),2)}**")
        st.write(f"- n_test: **{int(row.get('n_test')) if pd.notna(row.get('n_test')) else '—'}**")
        st.write(f"- Entrenado (UTC): **{row.get('trained_at','—')}**")

    with right:
        st.markdown("### Explicación completa")
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
