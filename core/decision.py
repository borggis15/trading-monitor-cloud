from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


FEATURES_DEFAULT = ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]


@dataclass
class Thresholds:
    buy_proba: float
    sell_proba: float
    buy_ev_bps: float
    sell_ev_bps: float


def _safe_float(x, default=None):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _prep_xy(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y_cls: 1 si ret_fwd > 0, 0 si <= 0
    y_reg: ret_fwd (float)
    """
    need = list(features) + ["ret_fwd"]
    d = df.dropna(subset=need).copy()
    X = d[features].to_numpy(dtype=float)
    y_reg = d["ret_fwd"].to_numpy(dtype=float)
    y_cls = (y_reg > 0).astype(int)
    return X, y_cls, y_reg


def _compute_thresholds_from_train(
    clf_cal: Any,
    reg: Any,
    X_train: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
) -> Thresholds:
    """
    Thresholds dinámicos:
    - buy_proba: cuantil alto de proba (p.ej. 70%)
    - sell_proba: cuantil bajo de proba (p.ej. 30%)
    - buy_ev_bps: cuantil alto de EV (p.ej. 60%)
    - sell_ev_bps: cuantil bajo de EV (p.ej. 40%)

    Estos números tienden a:
    - dar BUY cuando el propio histórico del activo sugiere señales “buenas”
    - reducir HOLDs sin volverlo loco
    """
    # Probabilidades calibradas en train
    proba = clf_cal.predict_proba(X_train)[:, 1]
    ret_exp = reg.predict(X_train)

    # Riesgo aproximado: usamos ATR relativo implícito en features no siempre disponible aquí;
    # para thresholds usamos EV basado solo en ret_exp - costes.
    # (En score final EV se calcula “bien” con risk_est).
    ev_bps = (ret_exp * 10000.0) - (fee_bps + slippage_bps)

    buy_p = float(np.quantile(proba, 0.70))
    sell_p = float(np.quantile(proba, 0.30))
    buy_ev = float(np.quantile(ev_bps, 0.60))
    sell_ev = float(np.quantile(ev_bps, 0.40))

    # clamps razonables
    buy_p = float(np.clip(buy_p, 0.55, 0.75))
    sell_p = float(np.clip(sell_p, 0.25, 0.48))

    # EV en bps: aseguramos un mínimo “decente” para BUY y un máximo “malo” para SELL
    buy_ev = max(buy_ev, 25.0)      # al menos +25 bps
    sell_ev = min(sell_ev, -25.0)   # al menos -25 bps

    return Thresholds(buy_proba=buy_p, sell_proba=sell_p, buy_ev_bps=buy_ev, sell_ev_bps=sell_ev)


def train_models(
    feat: pd.DataFrame,
    model_type: str = "hgb",
    min_rows: int = 240,
    features: Optional[list[str]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Tuple[Any, Any, int, Dict[str, Any]]:
    """
    Devuelve: (clf_calibrado, reg, train_rows, meta)

    meta incluye thresholds dinámicos:
      meta["thresholds"] = {buy_proba, sell_proba, buy_ev_bps, sell_ev_bps}
    """
    features = features or FEATURES_DEFAULT

    # Preparación
    X, y_cls, y_reg = _prep_xy(feat, features)
    n = int(len(X))
    if n < int(min_rows):
        raise ValueError(f"Not enough rows to train: {n} < min_rows={min_rows}")

    # Modelos base
    # HistGB funciona bien y es rápido.
    clf_base = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.06,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )

    reg = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.06,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )

    # ⚠️ TimeSeriesSplit para calibración sin leakage
    # Necesitamos al menos 3 splits, y tamaño suficiente.
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Fit base clf
    clf_base.fit(X, y_cls)

    # Calibración correcta (NO prefit)
    # En sklearn moderno, cv="prefit" ya no es válido como antes.
    clf_cal = CalibratedClassifierCV(
        estimator=clf_base,
        method="sigmoid",
        cv=tscv,
    )
    clf_cal.fit(X, y_cls)

    # Fit reg
    reg.fit(X, y_reg)

    # Thresholds dinámicos entrenados con el propio train
    thr = _compute_thresholds_from_train(clf_cal, reg, X, fee_bps=fee_bps, slippage_bps=slippage_bps)
    meta: Dict[str, Any] = {
        "features": features,
        "thresholds": {
            "buy_proba": thr.buy_proba,
            "sell_proba": thr.sell_proba,
            "buy_ev_bps": thr.buy_ev_bps,
            "sell_ev_bps": thr.sell_ev_bps,
        },
        "calibration": {"method": "sigmoid", "cv": "TimeSeriesSplit(3)"},
    }

    return clf_cal, reg, n, meta


def predict(clf, reg, feat: pd.DataFrame, features: Optional[list[str]] = None) -> tuple[Optional[float], Optional[float]]:
    """
    Predicción sobre la última fila de feat (o la que te pasen).
    """
    if feat is None or feat.empty:
        return None, None

    features = features or FEATURES_DEFAULT
    d = feat.dropna(subset=features).tail(1)
    if d.empty:
        return None, None

    X = d[features].to_numpy(dtype=float)

    proba = None
    ret_exp = None

    try:
        proba = float(clf.predict_proba(X)[0, 1])
    except Exception:
        proba = None

    try:
        ret_exp = float(reg.predict(X)[0])
    except Exception:
        ret_exp = None

    return proba, ret_exp
