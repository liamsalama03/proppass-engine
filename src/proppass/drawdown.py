# src/proppass/drawdown.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

DDType = Literal["STATIC", "TRUE_TRAIL", "BALANCE_TRAIL", "EOD_TRAIL"]

@dataclass(frozen=True)
class DrawdownState:
    remaining_total_dd: Optional[float]
    effective_risk_cap: Optional[float]
    remaining_profit: Optional[float]

def remaining_total_dd(dd_type: DDType, total_max_dd: Optional[float], realized_pnl: float) -> Optional[float]:
    if total_max_dd is None:
        return None
    if dd_type == "STATIC":
        rem = total_max_dd + realized_pnl
    else:
        rem = total_max_dd if realized_pnl >= 0 else (total_max_dd + realized_pnl)
    return max(0.0, rem)

def remaining_profit_to_target(profit_target: Optional[float], realized_pnl: float) -> Optional[float]:
    if profit_target is None:
        return None
    return max(0.0, profit_target - max(0.0, realized_pnl))

def effective_risk_cap(daily_max_loss: Optional[float], rem_total_dd: Optional[float]) -> Optional[float]:
    if daily_max_loss is None and rem_total_dd is None:
        return None
    if daily_max_loss is None:
        return rem_total_dd
    if rem_total_dd is None:
        return daily_max_loss
    return min(daily_max_loss, rem_total_dd)

