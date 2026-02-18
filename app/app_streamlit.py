import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

st.set_page_config(page_title="TR Quant Dashboard", layout="wide")

DATABASE_URL = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL"))
if not DATABASE_URL:
    st.error("Falta DATABASE_URL en secrets/env.")
    st.stop()

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

@st.cache_data(ttl=60)
def load_latest_signals():
    df = pd.read_sql(text("""
      select s.*
      from signals s
      join (
        select exchange, symbol, max(ts) as max_ts
        from signals
        group by exchange, symbol
      ) m on s.exchange=m.exchange and s.symbol=m.symbol and s.ts=m.max_ts
      order by coalesce(ev_bps,-1e9) desc
    """), engine)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

@st.cache_data(ttl=300)
def load_bars(exchange, symbol, limit=1400):
    df = pd.read_sql(text("""
      select ts, open, high, low, close, volume
      from bars_15m
      where exchange=:e and symbol=:s
      order by ts desc
      limit :n
    """), engine, params={"e": exchange, "s": symbol, "n": int(limit)})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").set_index("ts")

@st.cache_data(ttl=300)
def load_signals(exchange, symbol, limit=400):
    df = pd.read_sql(text("""
      select ts, action, ev_bps
      from signals
      where exchange=:e and symbol=:s
      order by ts desc
      limit :n
    """), engine, params={"e": exchange, "s": symbol, "n": int(limit)})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")

@st.cache_data(ttl=600)
def load_backtest(exchange, symbol):
    return pd.read_sql(text("select * from backtest_summary where exchange=:e and symbol=:s"), engine, params={"e": exchange, "s": symbol})

st.title("Panel cuantitativo (alineado a operar en Trade Republic)")
st.caption("Referencia XETRA (spreads ligados a XETRA cuando aplica). Señales: probabilidad + retorno esperado + EV + sizing.")

sig = load_latest_signals()
if sig.empty:
    st.warning("Aún no hay señales. Espera al primer run del workflow (cada 15 min).")
    st.stop()

def badge(a):
    return "🟢 BUY" if a=="BUY" else ("🔴 SELL" if a=="SELL" else "🟡 HOLD")

disp = sig.copy()
disp["Recomendación"] = disp["action"].map(badge)
disp["Prob ↑"] = disp["proba_up"].round(2)
disp["Ret exp %"] = (disp["ret_exp"]*100).round(2)
disp["EV (bps)"] = disp["ev_bps"].round(1)
disp["Tamaño €"] = disp["size_eur"].round(0)
disp["SL"] = disp["sl_price"].round(3)
disp["TP"] = disp["tp_price"].round(3)
disp = disp[["exchange","symbol","ts","Recomendación","Prob ↑","Ret exp %","EV (bps)","Tamaño €","SL","TP","explanation"]]

st.subheader("Resumen (6 acciones)")
st.dataframe(disp, use_container_width=True, hide_index=True)

left, right = st.columns([1,2])
with left:
    pick = st.selectbox("Detalle por acción", sig["symbol"].tolist(), index=0)
    ex = sig.loc[sig["symbol"]==pick, "exchange"].iloc[0]
    bt = load_backtest(ex, pick)
    st.subheader("Robustez (walk-forward)")
    if bt.empty:
        st.info("Backtest no disponible.")
    else:
        r = bt.iloc[0].to_dict()
        st.write({k: r[k] for k in ["trades","sharpe","sortino","max_drawdown","win_rate","profit_factor"]})

with right:
    bars = load_bars(ex, pick)
    if bars.empty:
        st.info("Sin datos todavía.")
    else:
        view = bars.tail(800)
        fig = go.Figure(data=[go.Candlestick(
            x=view.index, open=view["open"], high=view["high"], low=view["low"], close=view["close"], name="OHLC"
        )])
        s2 = load_signals(ex, pick)
        if not s2.empty:
            sp = s2[s2["ts"].between(view.index.min(), view.index.max())]
            buy = sp[sp["action"]=="BUY"]
            sell = sp[sp["action"]=="SELL"]
            if not buy.empty:
                fig.add_trace(go.Scatter(x=buy["ts"], y=bars.reindex(buy["ts"])["close"].values, mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10))
            if not sell.empty:
                fig.add_trace(go.Scatter(x=sell["ts"], y=bars.reindex(sell["ts"])["close"].values, mode="markers", name="SELL", marker_symbol="triangle-down", marker_size=10))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{pick} — {ex} — 15m")
        st.plotly_chart(fig, use_container_width=True)
