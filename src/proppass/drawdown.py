# src/proppass/drawdown.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

DDType = Literal["STATIC", "TRUE_TRAIL", "BALANCE_TRAIL", "EOD_TRAIL"]


@dataclass(frozen=True)
class DDResult:
    remaining_total_dd: Optional[float]
    remaining_profit: Optional[float]
    effective_risk_cap: Optional[float]  # min(daily, remaining_total_dd) if both exist


def calc_remaining_total_dd(
    dd_type: DDType,
    total_max_dd: Optional[float],
    realized_pnl: float,
) -> Optional[float]:
    """
    Mirrors your Sheets v1 realized-only DD logic:
      - STATIC: remaining = total_dd + realized_pnl
      - TRAILING (TRUE/BALANCE/EOD): if realized_pnl >= 0 => full DD remains
                                    else => total_dd + realized_pnl
    Clamp at >= 0.
    """
    if total_max_dd is None:
        return None

    if dd_type == "STATIC":
        rem = total_max_dd + realized_pnl
    else:
        rem = total_max_dd if realized_pnl >= 0 else (total_max_dd + realized_pnl)

    return max(0.0, rem)


def calc_remaining_profit(
    profit_target: Optional[float],
    realized_pnl: float,
) -> Optional[float]:
    if profit_target is None:
        return None
    # profit target is measured from start; profits below 0 don't increase "remaining"
    return max(0.0, profit_target - max(0.0, realized_pnl))


def calc_effective_risk_cap(
    daily_max_loss: Optional[float],
    remaining_total_dd: Optional[float],
) -> Optional[float]:
    """
    Effective cap used for contract sizing / max stop-outs:
      - If both exist => min(daily, remaining_total_dd)
      - If one exists => that one
      - If neither => None
    """
    if daily_max_loss is None and remaining_total_dd is None:
        return None
    if daily_max_loss is None:
        return remaining_total_dd
    if remaining_total_dd is None:
        return daily_max_loss
    return min(daily_max_loss, remaining_total_dd)


def calc_dd_bundle(
    dd_type: DDType,
    daily_max_loss: Optional[float],
    total_max_dd: Optional[float],
    profit_target: Optional[float],
    realized_pnl: float,
) -> DDResult:
    rem_dd = calc_remaining_total_dd(dd_type, total_max_dd, realized_pnl)
    rem_profit = calc_remaining_profit(profit_target, realized_pnl)
    cap = calc_effective_risk_cap(daily_max_loss, rem_dd)
    return DDResult(remaining_total_dd=rem_dd, remaining_profit=rem_profit, effective_risk_cap=cap)


def format_breach_message(value: Optional[float]) -> str:
    """
    Optional helper if you want text instead of blank.
    """
    if value is None:
        return "N/A"
    if value <= 0:
        return "Breaches limit"
    return f"{value:.0f}"
def update_high_water_mark(previous_hwm: float, current_equity: float) -> float:
    """
    High-water mark = max equity reached so far.
    If you don't track history yet, treat previous_hwm as starting balance.
    """
    return max(float(previous_hwm), float(current_equity))


def trailing_dd_line(hwm: float, max_drawdown: float) -> float:
    """
    Trailing threshold line = HWM - maxDD.
    Breach occurs if equity < this line.
    """
    return float(hwm) - float(max_drawdown)
from dataclasses import dataclass

@dataclass(frozen=True)
class TrailingState:
    hwm: float
    trailing_line: float

def compute_trailing_state(hwm: float, max_drawdown: float) -> TrailingState:
    """
    Returns the trailing DD threshold line for TRUE_TRAIL style:
      trailing_line = HWM - maxDD
    """
    hwm = float(hwm)
    max_drawdown = float(max_drawdown)
    return TrailingState(hwm=hwm, trailing_line=hwm - max_drawdown)
