from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

FEATURES = ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]


def train_models(feat: pd.DataFrame, model_type: str, min_rows: int):
    """
    Entrena:
      - Clasificador: probabilidad de retorno positivo
      - Regresor: retorno esperado
    De forma robusta: si no hay datos/columnas suficientes, devuelve (None, None, n_rows)
    """
    # ✅ Guard: feat vacío
    if feat is None or feat.empty:
        return None, None, 0

    needed = set(FEATURES + ["ret_fwd"])
    # ✅ Guard: columnas faltantes
    if not needed.issubset(set(feat.columns)):
        return None, None, 0

    df = feat.dropna(subset=list(needed)).copy()
    n = len(df)
    if n < int(min_rows):
        return None, None, n

    X = df[FEATURES].values
    y_cls = (df["ret_fwd"] > 0).astype(int).values
    y_reg = df["ret_fwd"].values

    # Modelo robusto (default): HistGradientBoosting
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300)
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300)

    clf.fit(X, y_cls)
    reg.fit(X, y_reg)

    return clf, reg, n


def predict(clf, reg, feat: pd.DataFrame):
    """
    Devuelve (proba_subida, retorno_esperado) en el último punto con features válidas.
    Si no hay datos suficientes, devuelve (None, None).
    """
    if feat is None or feat.empty:
        return None, None

    if not set(FEATURES).issubset(set(feat.columns)):
        return None, None

    last = feat.dropna(subset=FEATURES).tail(1)
    if last.empty:
        return None, None

    X = last[FEATURES].values
    proba = float(clf.predict_proba(X)[0, 1]) if clf is not None else None
    ret = float(reg.predict(X)[0]) if reg is not None else None
    return proba, ret
