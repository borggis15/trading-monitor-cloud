from __future__ import annotations

def size_from_atr(capital_eur: float, risk_pct: float, max_pos_pct: float, price: float, atr: float, sl_atr_mult: float):
    if price <= 0 or atr <= 0:
        return None, None
    risk_eur = capital_eur * (risk_pct/100.0)
    sl_dist = sl_atr_mult * atr
    qty = risk_eur / sl_dist
    size = min(qty * price, capital_eur * (max_pos_pct/100.0))
    sl = price - sl_dist
    return float(size), float(sl)

def ev_bps(ret_exp: float|None, risk_est: float|None, fee_bps: float, slippage_bps: float):
    if ret_exp is None:
        return None
    costs = (fee_bps + slippage_bps) / 10000.0
    penalty = 0.25 * (risk_est if risk_est is not None else 0.0)
    ev = ret_exp - costs - penalty
    return float(ev * 10000.0)
