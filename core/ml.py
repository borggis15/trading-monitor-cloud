from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV


FEATURES = ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]


def _prep_xy(feat: pd.DataFrame):
    df = feat.dropna(subset=FEATURES + ["ret_fwd"]).copy()
    if df.empty:
        return None, None, None

    X = df[FEATURES].astype(float)
    y_up = (df["ret_fwd"].astype(float) > 0).astype(int)
    y_ret = df["ret_fwd"].astype(float)
    return df, X, y_up, y_ret


def train_models(
    feat: pd.DataFrame,
    model_type: str = "hgb",
    min_rows: int = 200,
):
    """
    Entrena:
      - clasificador (probabilidad subida)
      - regresor (retorno esperado)
    + Calibración de probas con holdout time-based (sigmoid) => más realista.
    """
    out = _prep_xy(feat)
    if out[0] is None:
        return None, None, 0
    df, X, y_up, y_ret = out

    if len(df) < min_rows:
        return None, None, int(len(df))

    # split time-based (no shuffle)
    split = int(len(df) * 0.8)
    if split < min_rows:
        split = min_rows

    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y_up.iloc[:split], y_up.iloc[split:]
    r_tr, r_va = y_ret.iloc[:split], y_ret.iloc[split:]

    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )
    reg = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )

    clf.fit(X_tr, y_tr)
    reg.fit(X_tr, r_tr)

    # Calibración SOLO si hay val suficiente
    clf_cal = clf
    if len(X_va) >= 50 and y_va.nunique() > 1:
        clf_cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        clf_cal.fit(X_va, y_va)

    return clf_cal, reg, int(len(df))


def predict(clf, reg, feat_latest: pd.DataFrame):
    """
    Devuelve:
      proba_up (float)
      ret_exp (float)
    """
    if clf is None or reg is None or feat_latest is None or feat_latest.empty:
        return None, None

    df = feat_latest.dropna(subset=FEATURES).copy()
    if df.empty:
        return None, None

    X = df[FEATURES].astype(float).tail(1)
    try:
        proba = float(clf.predict_proba(X)[0, 1])
    except Exception:
        proba = None

    try:
        ret_exp = float(reg.predict(X)[0])
    except Exception:
        ret_exp = None

    return proba, ret_exp
