from dataclasses import dataclass
from enum import Enum

class DrawdownType(str, Enum):
    EOD_TRAILING = "eod_trailing"
    INTRADAY_TRAILING = "intraday"
    STATIC = "static"

@dataclass(frozen=True)
class EvalRules:
    drawdown_type: DrawdownType
    max_drawdown: float
    profit_target: float
    starting_balance: float
