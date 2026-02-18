# Trading Monitor Cloud (TR-optimizado, legal, automático)

Sistema automatizado (cada 15 min) para 6 acciones:
- Ingesta + features + ML + señales + sizing con gestión de riesgo
- Backtest walk-forward (sin fugas de futuro) y métricas
- Interfaz web Streamlit (solo lectura)
- Sin conectarse a Trade Republic (legal)

**Optimización para TR:** Trade Republic ejecuta en **LS Exchange** y los spreads están ligados al mercado de referencia **XETRA** cuando aplica. Por eso usamos datos de **XETRA/Frankfurt** como referencia.

## Deploy recomendado (sin servidores)
- Supabase Postgres (persistencia)
- GitHub Actions (cron cada 15 min + semanal)
- Streamlit Community Cloud (UI)

## Secrets requeridos
- DATABASE_URL
- TWELVEDATA_API_KEY

## Pasos rápidos
1) Supabase: crea proyecto → copia DATABASE_URL
2) Twelve Data: crea API key
3) GitHub: sube este repo → añade Secrets en Actions
4) Supabase SQL: ejecuta sql/schema.sql
5) Activa Actions (verás runs cada 15 min)
6) Streamlit Cloud: despliega app/app_streamlit.py y añade secrets
