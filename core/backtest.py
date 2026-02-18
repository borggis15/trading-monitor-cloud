from __future__ import annotations
import numpy as np
import pandas as pd
from .features import compute_features
from .ml import train_models, predict

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0

def sharpe(returns: pd.Series, bars_per_year: int = 252*26) -> float:
    mu = returns.mean()
    sd = returns.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((mu/sd) * np.sqrt(bars_per_year))

def sortino(returns: pd.Series, bars_per_year: int = 252*26) -> float:
    neg = returns[returns < 0]
    sd = neg.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((returns.mean()/sd) * np.sqrt(bars_per_year))

def walk_forward_metrics(bars: pd.DataFrame, cfg: dict) -> dict:
    if bars is None or len(bars) < 1500:
        return {"trades": 0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0}

    horizon = int(cfg["ml"]["horizon_bars"])
    fee_bps = float(cfg["backtest"]["fee_bps"])
    slip_bps = float(cfg["backtest"]["slippage_bps"])
    train_days = int(cfg["backtest"]["walk_forward"]["train_days"])
    test_days = int(cfg["backtest"]["walk_forward"]["test_days"])
    bars_per_day = 32
    train_n = train_days * bars_per_day
    test_n = test_days * bars_per_day

    feat_all = compute_features(bars, horizon_bars=horizon).dropna()

    equity = [1.0]
    trades = wins = 0
    pf_pos = pf_neg = 0.0

    i = train_n
    while i + test_n < len(feat_all):
        train = feat_all.iloc[i-train_n:i]
        test = feat_all.iloc[i:i+test_n]

        clf, reg, _ = train_models(train, model_type=cfg["ml"]["model_type"], min_rows=int(cfg["ml"]["min_train_rows"]))
        for ts, row in test.iterrows():
            window = feat_all.loc[:ts].tail(1)
            proba, ret_exp = predict(clf, reg, window)
            if proba is None or ret_exp is None:
                continue
            if proba > 0.58 and ret_exp > 0:
                realized = float(row["ret_fwd"])
                costs = (fee_bps + slip_bps) / 10000.0
                realized_net = realized - costs
                equity.append(equity[-1] * (1.0 + realized_net))
                trades += 1
                if realized_net > 0:
                    wins += 1
                    pf_pos += realized_net
                else:
                    pf_neg += abs(realized_net)
        i += test_n

    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    win_rate = (wins / trades) if trades else 0.0
    profit_factor = (pf_pos / pf_neg) if pf_neg > 0 else (pf_pos if pf_pos > 0 else 0.0)

    return {
        "trades": int(trades),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": max_drawdown(eq),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
    }
