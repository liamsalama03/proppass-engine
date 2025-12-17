from dataclasses import dataclass

@dataclass(frozen=True)
class TrailingState:
    high_water_mark: float
    dd_floor: float

def compute_trailing_state(high_water_mark: float, max_drawdown: float) -> TrailingState:
    if max_drawdown <= 0:
        raise ValueError("max_drawdown must be > 0")

    return TrailingState(
        high_water_mark=high_water_mark,
        dd_floor=high_water_mark - max_drawdown,
    )

def update_high_water_mark(prev_hwm: float, equity: float) -> float:
    return max(prev_hwm, equity)

def buffer_to_floor(equity: float, dd_floor: float) -> float:
    return equity - dd_floor
