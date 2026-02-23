from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV


FEATURES = ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]


def _prep_xy(df: pd.DataFrame):
    """
    Prepara X, y_cls, y_reg desde un DataFrame de features.
    Requiere: FEATURES y 'ret_fwd' en df.
    """
    need = FEATURES + ["ret_fwd"]
    if df is None or df.empty or any(c not in df.columns for c in need):
        return None, None, None

    d = df.dropna(subset=need).copy()
    if d.empty:
        return None, None, None

    X = d[FEATURES].astype(float)

    # Clasificación: "sube" si ret_fwd > 0
    y_cls = (d["ret_fwd"].astype(float) > 0.0).astype(int)

    # Regresión: retorno esperado
    y_reg = d["ret_fwd"].astype(float)

    return X, y_cls, y_reg


def train_models(
    feat: pd.DataFrame,
    model_type: str = "hgb",
    min_rows: int = 200,
    calibrate: bool = True,
    cal_cv: int = 3,
):
    """
    Entrena:
      - clf: clasificador prob(up)
      - reg: regresor retorno esperado

    NOTA sobre calibración:
    - scikit-learn reciente ya NO permite cv="prefit".
    - Usamos CalibratedClassifierCV con CV interno (cal_cv=3 por defecto).
    """
    X, y_cls, y_reg = _prep_xy(feat)
    if X is None:
        return None, None, 0

    n = len(X)
    if n < min_rows:
        # no entrenamos si no hay histórico suficiente
        return None, None, int(n)

    if model_type != "hgb":
        raise ValueError(f"Unsupported model_type={model_type}. Use 'hgb'.")

    # Base models
    base_clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.07,
        max_iter=300,
        random_state=42,
    )

    reg = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.07,
        max_iter=300,
        random_state=42,
    )

    # Entrenamos regresor
    reg.fit(X, y_reg)

    # Entrenamos clasificador (con calibración opcional)
    if calibrate:
        # ✅ Calibración robusta y compatible con sklearn moderno:
        # CalibratedClassifierCV entrena internamente con CV.
        clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=int(cal_cv),
        )
        clf.fit(X, y_cls)
    else:
        base_clf.fit(X, y_cls)
        clf = base_clf

    return clf, reg, int(n)


def predict(clf, reg, feat: pd.DataFrame):
    """
    Devuelve (proba_up, ret_exp) usando el último punto disponible del DataFrame feat.
    """
    if feat is None or feat.empty:
        return None, None

    last = feat.dropna(subset=FEATURES).tail(1)
    if last.empty:
        return None, None

    X_last = last[FEATURES].astype(float)

    proba = None
    ret_exp = None

    if clf is not None:
        try:
            # clf puede ser CalibratedClassifierCV o el clasificador base
            p = clf.predict_proba(X_last)[0, 1]
            proba = float(p)
        except Exception:
            proba = None

    if reg is not None:
        try:
            r = reg.predict(X_last)[0]
            ret_exp = float(r)
        except Exception:
            ret_exp = None

    # Sanidad básica
    if proba is not None:
        proba = max(0.0, min(1.0, proba))

    if ret_exp is not None and not np.isfinite(ret_exp):
        ret_exp = None

    return proba, ret_exp
