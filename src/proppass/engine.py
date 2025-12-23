from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict
import math
import random

DDType = Literal["TRUE_TRAIL", "EOD_TRAIL", "BALANCE_TRAIL", "STATIC"]
Instrument = Literal["NQ", "MNQ"]
RiskMode = Literal["Safe", "Standard", "Aggressive"]


def tick_size_for(_instrument: Instrument) -> float:
    return 0.25


def tick_value_for(instrument: Instrument) -> float:
    return 5.0 if instrument == "NQ" else 0.5


def minis_to_contracts(instrument: Instrument, minis: int) -> int:
    # Your choice B: minis -> micros x10
    return minis * 10 if instrument == "MNQ" else minis


def risk_per_contract_dollars(instrument: Instrument, stop_points: float) -> float:
    ts = tick_size_for(instrument)
    tv = tick_value_for(instrument)
    ticks = stop_points / ts
    return ticks * tv


def remaining_total_dd(dd_type: DDType, total_max_dd: Optional[float], realized_pnl: float) -> Optional[float]:
    if total_max_dd is None:
        return None
    if dd_type == "STATIC":
        rem = total_max_dd + realized_pnl
    else:
        rem = total_max_dd if realized_pnl >= 0 else (total_max_dd + realized_pnl)
    return max(0.0, rem)


def remaining_profit(profit_target: Optional[float], realized_pnl: float) -> Optional[float]:
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


def contracts_for_mode(
    cap: Optional[float],
    risk_per_contract: float,
    firm_max_contracts: int,
    risk_frac_daily_dd: float,
) -> int:
    """
    Uses your RiskFracDailyDDPerTrade:
      risk_per_trade_target = cap * risk_frac
      contracts = floor(target / risk_per_contract)
    """
    if cap is None or cap <= 0 or risk_per_contract <= 0:
        return 0
    target_risk = cap * float(risk_frac_daily_dd)
    c = int(math.floor(target_risk / risk_per_contract))
    return max(0, min(c, firm_max_contracts))


def max_full_losses_before(cap: Optional[float], risk_per_trade: float) -> Optional[int]:
    if cap is None or cap <= 0 or risk_per_trade <= 0:
        return None
    return int(math.floor(cap / risk_per_trade))


def expected_edge_per_trade(win_rate: float, rr: float, risk_per_trade: float) -> Optional[float]:
    if risk_per_trade <= 0:
        return None
    p = min(max(win_rate, 0.0), 1.0)
    ev_r = (p * rr) - (1.0 - p)
    return ev_r * risk_per_trade


def estimated_trades_to_target(rem_profit: Optional[float], ev: Optional[float]) -> Optional[float]:
    if rem_profit is None:
        return None
    if ev is None or ev <= 0:
        return None
    return rem_profit / ev


def pass_probability_mc(
    win_rate: float,
    rr: float,
    risk_per_trade: float,
    rem_profit: Optional[float],
    rem_dd: Optional[float],
    n_sims: int = 2500,
    max_trades: int = 2000,
    seed: int = 7,
) -> Optional[float]:
    if rem_profit is None or rem_dd is None:
        return None
    if rem_profit <= 0:
        return 1.0
    if rem_dd <= 0:
        return 0.0
    if risk_per_trade <= 0:
        return None

    p = min(max(win_rate, 0.0), 1.0)
    rng = random.Random(seed)

    wins = 0
    for _ in range(n_sims):
        profit = 0.0
        dd_used = 0.0

        for _t in range(max_trades):
            if profit >= rem_profit:
                wins += 1
                break
            if dd_used >= rem_dd:
                break

            if rng.random() < p:
                profit += rr * risk_per_trade
            else:
                dd_used += risk_per_trade

    return wins / n_sims


def pass_label(prob: Optional[float]) -> Optional[str]:
    if prob is None:
        return None
    if prob < 0.33:
        return f"Low ({prob:.0%})"
    if prob < 0.66:
        return f"Moderate ({prob:.0%})"
    if prob < 0.80:
        return f"Moderate-High ({prob:.0%})"
    return f"High ({prob:.0%})"


@dataclass(frozen=True)
class EngineOutputs:
    tick_value: float
    risk_per_contract: float
    firm_max_contracts: int

    rem_total_dd: Optional[float]
    rem_profit: Optional[float]
    cap: Optional[float]

    contracts: Dict[RiskMode, int]
    active_contracts: int
    risk_per_trade: float

    max_losses_daily: Optional[int]
    max_losses_total: Optional[int]

    ev_per_trade: Optional[float]
    est_trades: Optional[float]

    pass_prob: Optional[float]
    pass_text: Optional[str]


def compute(
    *,
    dd_type: DDType,
    daily_max_loss: Optional[float],
    total_max_dd: Optional[float],
    profit_target: Optional[float],
    firm_max_contracts_minis: int,
    instrument: Instrument,
    win_rate: float,
    rr: float,
    stop_points: float,
    realized_pnl: float,
    risk_fracs: Dict[RiskMode, float],
    selected_mode: RiskMode,
    sims: int = 2500,
) -> EngineOutputs:

    tv = tick_value_for(instrument)
    rpc = risk_per_contract_dollars(instrument, stop_points)

    firm_max = minis_to_contracts(instrument, firm_max_contracts_minis)

    rem_dd = remaining_total_dd(dd_type, total_max_dd, realized_pnl)
    rem_p = remaining_profit(profit_target, realized_pnl)
    cap = effective_risk_cap(daily_max_loss, rem_dd)

    contracts = {
        "Safe": contracts_for_mode(cap, rpc, firm_max, risk_fracs["Safe"]),
        "Standard": contracts_for_mode(cap, rpc, firm_max, risk_fracs["Standard"]),
        "Aggressive": contracts_for_mode(cap, rpc, firm_max, risk_fracs["Aggressive"]),
    }

    active = contracts[selected_mode]
    risk_per_trade = rpc * active

    max_daily = max_full_losses_before(cap, risk_per_trade)
    max_total = max_full_losses_before(rem_dd, risk_per_trade)

    ev = expected_edge_per_trade(win_rate, rr, risk_per_trade)
    est = estimated_trades_to_target(rem_p, ev)

    prob = pass_probability_mc(win_rate, rr, risk_per_trade, rem_p, rem_dd, n_sims=sims)
    text = pass_label(prob)

    return EngineOutputs(
        tick_value=tv,
        risk_per_contract=rpc,
        firm_max_contracts=firm_max,
        rem_total_dd=rem_dd,
        rem_profit=rem_p,
        cap=cap,
        contracts=contracts,
        active_contracts=active,
        risk_per_trade=risk_per_trade,
        max_losses_daily=max_daily,
        max_losses_total=max_total,
        ev_per_trade=ev,
        est_trades=est,
        pass_prob=prob,
        pass_text=text,
    )
