import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Trading Monitor Pro", layout="wide")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    st.error("Missing DATABASE_URL in Streamlit Secrets.")
    st.stop()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ----------------------------
# Utils
# ----------------------------
def to_dt_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def num(x):
    try:
        v = pd.to_numeric(x, errors="coerce")
        return None if pd.isna(v) else float(v)
    except Exception:
        return None

def fmt(x, nd=2, empty="—"):
    v = num(x)
    return empty if v is None else f"{v:.{nd}f}"

def fmt_pct(x, nd=2, empty="—"):
    v = num(x)
    return empty if v is None else f"{100*v:.{nd}f}%"

def bps_to_pct_str(ev_bps, empty="—"):
    v = num(ev_bps)
    return empty if v is None else f"{v/100:.2f}%"

def action_style(action: str):
    a = (action or "").upper()
    if a == "BUY":
        return ("BUY", "#0f766e")   # teal
    if a == "SELL":
        return ("SELL", "#b42318")  # red
    return ("HOLD", "#1f2937")      # slate

def trend_from_ret(ret_exp):
    v = num(ret_exp)
    if v is None:
        return "Neutral"
    if v > 0.02:
        return "Bullish"
    if v < -0.02:
        return "Bearish"
    return "Neutral"

def confidence_score(proba_up, sharpe, n_test):
    """
    Score orientativo 0-100 para UI (no “verdad”, solo un resumen).
    - proba_up domina
    - sharpe y n_test suben confianza de forma suave
    """
    p = num(proba_up)
    sh = num(sharpe)
    nt = num(n_test)

    if p is None:
        return None

    base = max(0.0, min(1.0, p)) * 70.0  # 0..70
    sh_bonus = 0.0
    if sh is not None:
        sh_bonus = max(0.0, min(1.0, sh / 10.0)) * 20.0  # 0..20
    nt_bonus = 0.0
    if nt is not None:
        nt_bonus = max(0.0, min(1.0, nt / 250.0)) * 10.0  # 0..10

    return round(min(100.0, base + sh_bonus + nt_bonus), 0)

def horizon_days_from_row(row):
    # Si la vista ya trae horizon, lo usamos. Si no, por defecto 10 (tu setup actual).
    h = row.get("horizon", None)
    hv = num(h)
    if hv is not None and hv > 0:
        return int(hv)
    return 10

def expected_pnl_eur(size_eur, ret_exp):
    s = num(size_eur)
    r = num(ret_exp)
    if s is None or r is None:
        return None
    return s * r

# ----------------------------
# Loaders
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
        df["ts"] = to_dt_utc(df["ts"])
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
        df["trained_at"] = to_dt_utc(df["trained_at"])
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
    df = pd.read_sql(text(q), engine, params={"exchange": exchange, "symbol": symbol, "limit": int(limit)})
    if df.empty:
        return df
    df["ts"] = to_dt_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df

@st.cache_data(ttl=60)
def load_signal_history(exchange, symbol, limit=250):
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
    df["ts"] = to_dt_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df

# ----------------------------
# Sidebar
# ----------------------------
st.title("Trading Monitor Pro")

with st.sidebar:
    st.subheader("Controls")

    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    actions_filter = st.multiselect(
        "Actions",
        options=["BUY", "HOLD", "SELL"],
        default=["BUY", "HOLD", "SELL"],
    )

    st.subheader("Quality filters")
    min_ev = st.slider("Min EV (bps)", min_value=-1000, max_value=1500, value=-1000, step=10)
    min_sharpe = st.slider("Min Sharpe", min_value=-2.0, max_value=15.0, value=-2.0, step=0.1)
    min_n_test = st.slider("Min n_test", min_value=0, max_value=500, value=0, step=10)

    st.subheader("Urgency banner")
    urgent_ev = st.slider("Urgent EV (bps)", min_value=0, max_value=1500, value=900, step=10)
    urgent_conf = st.slider("Urgent confidence", min_value=50, max_value=95, value=80, step=1)

    st.subheader("Sizing (optional)")
    default_size = st.number_input("Default position size (€)", min_value=0, value=0, step=50)
    st.caption("Used only if the signal has no size_eur.")

    st.subheader("Ranking")
    sort_by = st.selectbox("Sort by", ["EV (bps)", "Confidence", "Sharpe", "Hit rate", "Max DD", "n_test"], index=0)

# ----------------------------
# Load & merge
# ----------------------------
signals = load_signals_current()
metrics = load_latest_metrics()

if signals.empty:
    st.warning("No current signals yet. Run score_15m.")
    st.stop()

df = signals.merge(metrics, on=["exchange", "symbol"], how="left")

# Coerce numeric
for c in ["ev_bps", "proba_up", "ret_exp", "size_eur", "sl_price", "tp_price",
          "sharpe", "max_dd", "hit_rate", "profit_factor", "n_test"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["confidence"] = df.apply(lambda r: confidence_score(r.get("proba_up"), r.get("sharpe"), r.get("n_test")), axis=1)

# Filters
df_view = df[df["action"].isin(actions_filter)].copy()
df_view = df_view[df_view["ev_bps"].fillna(-1e9) >= float(min_ev)]
df_view = df_view[df_view["sharpe"].fillna(-1e9) >= float(min_sharpe)]
df_view = df_view[df_view["n_test"].fillna(-1e9) >= float(min_n_test)]

# Sorting
if sort_by == "EV (bps)":
    df_rank = df_view.sort_values("ev_bps", ascending=False)
elif sort_by == "Confidence":
    df_rank = df_view.sort_values("confidence", ascending=False)
elif sort_by == "Sharpe":
    df_rank = df_view.sort_values("sharpe", ascending=False)
elif sort_by == "Hit rate":
    df_rank = df_view.sort_values("hit_rate", ascending=False)
elif sort_by == "Max DD":
    df_rank = df_view.sort_values("max_dd", ascending=True)  # less negative is better
elif sort_by == "n_test":
    df_rank = df_view.sort_values("n_test", ascending=False)
else:
    df_rank = df_view.copy()

# ----------------------------
# Urgency banner (professional, minimal)
# ----------------------------
urgent = df.copy()
urgent = urgent[(urgent["action"] == "BUY") &
                (urgent["ev_bps"].fillna(-1e9) >= urgent_ev) &
                (urgent["confidence"].fillna(0) >= urgent_conf)].copy()

if not urgent.empty:
    top = urgent.sort_values(["ev_bps", "confidence"], ascending=False).iloc[0]
    st.info(
        f"Urgent: BUY candidate — {top['asset_name']} ({top['exchange']}:{top['symbol']}) "
        f"| EV {fmt(top['ev_bps'],1)} bps | Confidence {int(top['confidence'])}/100",
        icon=None
    )

# KPIs
last_ts = df["ts"].max()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Last signal time (UTC)", value=str(last_ts) if pd.notna(last_ts) else "—")
k2.metric("BUY", int((df["action"] == "BUY").sum()))
k3.metric("HOLD", int((df["action"] == "HOLD").sum()))
k4.metric("SELL", int((df["action"] == "SELL").sum()))

st.divider()

# ----------------------------
# Cards (clean, not dense)
# ----------------------------
st.subheader("Current signals")

cols = st.columns(3)
cards = df_rank.sort_values("asset_name").to_dict(orient="records")

for i, r in enumerate(cards):
    with cols[i % 3]:
        with st.container(border=True):
            name = r.get("asset_name") or "—"
            ex = r.get("exchange") or "—"
            sym = r.get("symbol") or "—"
            action_txt, color = action_style(r.get("action"))
            h_days = horizon_days_from_row(r)
            ts = r.get("ts")
            entry_ts = ts if pd.notna(ts) else None
            exit_ts = (entry_ts + pd.Timedelta(days=h_days)) if entry_ts is not None else None

            size_eur = num(r.get("size_eur"))
            if size_eur is None and default_size > 0:
                size_eur = float(default_size)

            pnl_eur = expected_pnl_eur(size_eur, r.get("ret_exp"))

            # Header
            st.markdown(f"**{name}**")
            st.caption(f"{ex}:{sym} · model {r.get('model_id','—')} · horizon {h_days}d")

            # Row 1: Action + key metrics
            c1, c2, c3 = st.columns([1.0, 1.1, 1.1])
            with c1:
                st.markdown(
                    f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:{color};color:white;font-weight:700;'>"
                    f"{action_txt}</div>",
                    unsafe_allow_html=True
                )
            with c2:
                st.metric("EV (bps)", fmt(r.get("ev_bps"), 1))
            with c3:
                conf = r.get("confidence")
                st.metric("Confidence", f"{int(conf)}/100" if pd.notna(conf) else "—")

            # Row 2: Expected return + sizing
            c4, c5, c6 = st.columns(3)
            with c4:
                st.metric("Exp. return", fmt_pct(r.get("ret_exp"), 2))
            with c5:
                st.metric("Suggested size (€)", fmt(size_eur, 0))
            with c6:
                st.metric("Exp. P&L (€)", fmt(pnl_eur, 0))

            # Timeline (compact)
            if entry_ts is not None:
                st.caption(
                    f"Window: entry {entry_ts.strftime('%Y-%m-%d %H:%M')} → target {exit_ts.strftime('%Y-%m-%d %H:%M') if exit_ts is not None else '—'} (UTC)"
                )
            else:
                st.caption("Window: —")

            # Plan (short, professional)
            tlabel = trend_from_ret(r.get("ret_exp"))
            if action_txt == "BUY":
                st.write(f"Trend: {tlabel}. Plan: enter within the window; exit on SELL signal or at target date.")
                if pd.notna(r.get("sl_price")) or pd.notna(r.get("tp_price")):
                    st.caption(f"Risk levels: SL {fmt(r.get('sl_price'),2)} · TP {fmt(r.get('tp_price'),2)}")
            elif action_txt == "SELL":
                st.write(f"Trend: {tlabel}. Plan: avoid new entries; if holding, reduce/exit on next liquidity window.")
            else:
                st.write(f"Trend: {tlabel}. Plan: wait; reassess on next update or if confidence increases.")

            # Optional: technical explanation (kept away)
            with st.expander("Technical details"):
                st.code((r.get("explanation") or "").strip(), language="text")

st.divider()

# ----------------------------
# Ranking table
# ----------------------------
st.subheader("Ranking")

show_cols = [
    "asset_name", "exchange", "symbol", "action",
    "ev_bps", "confidence", "proba_up", "ret_exp",
    "sharpe", "max_dd", "hit_rate", "n_test",
    "model_id"
]
st.dataframe(df_rank[show_cols], use_container_width=True)

st.divider()

# ----------------------------
# Detail by asset (chart first)
# ----------------------------
st.subheader("Asset detail")

names = df_rank["asset_name"].tolist() if not df_rank.empty else df["asset_name"].tolist()
if not names:
    st.info("No assets under current filters.")
    st.stop()

sel = st.selectbox("Select an asset", options=names, index=0)
row = df[df["asset_name"] == sel].iloc[0]
ex = row["exchange"]
sym = row["symbol"]
h_days = horizon_days_from_row(row)

bars = load_bars(ex, sym, limit=900)
hist = load_signal_history(ex, sym, limit=250)

# Chart FIRST
if bars.empty:
    st.warning("No bars available for this asset.")
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

    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Summary BELOW
left, right = st.columns([1, 1], vertical_alignment="top")

with left:
    action_txt, color = action_style(row.get("action"))
    st.markdown(f"**{row.get('asset_name','—')}**")
    st.caption(f"{ex}:{sym} · horizon {h_days}d · model {row.get('model_id','—')}")

    st.markdown(
        f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:{color};color:white;font-weight:700;'>"
        f"{action_txt}</div>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write(f"EV: {fmt(row.get('ev_bps'),1)} bps ({bps_to_pct_str(row.get('ev_bps'))})")
    st.write(f"Expected return: {fmt_pct(row.get('ret_exp'),2)} ({trend_from_ret(row.get('ret_exp'))})")
    st.write(f"Confidence: {int(row.get('confidence'))}/100" if pd.notna(row.get("confidence")) else "Confidence: —")

with right:
    st.markdown("Backtest robustness (latest)")
    st.write(f"Sharpe: {fmt(row.get('sharpe'),2)}")
    st.write(f"Max drawdown: {fmt(row.get('max_dd'),2)}")
    st.write(f"Hit rate: {fmt(row.get('hit_rate'),2)}")
    st.write(f"n_test: {int(row.get('n_test')) if pd.notna(row.get('n_test')) else '—'}")
    st.caption(f"Trained at (UTC): {row.get('trained_at','—')}")

with st.expander("Technical details"):
    st.code((row.get("explanation") or "").strip(), language="text")

st.caption("This dashboard provides quantitative signals. It does not guarantee profits. Always apply risk management.")
