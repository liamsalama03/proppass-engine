from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

DDType = Literal["TRUE_TRAIL", "EOD_TRAIL", "BALANCE_TRAIL", "STATIC"]


@dataclass(frozen=True)
class FirmRule:
    firm: str
    account_size: str
    daily_max_loss: Optional[float]     # None means N/A
    total_max_dd: Optional[float]       # None means N/A
    profit_target: Optional[float]      # None means N/A
    firm_max_contracts_minis: Optional[int]  # stored as MINIS by design
    dd_type: DDType


def load_config(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # normalize headers
    df.columns = [c.strip() for c in df.columns]

    # normalize N/A
    df = df.replace({"N/A": pd.NA, "NA": pd.NA, "": pd.NA})

    # numeric columns
    for col in ["DailyMaxLoss", "TotalMaxDD", "ProfitTarget", "RiskFracDailyDDPerTrade"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "FirmMaxContracts" in df.columns:
        df["FirmMaxContracts"] = pd.to_numeric(df["FirmMaxContracts"], errors="coerce").astype("Int64")

    # string columns
    for col in ["Firm", "AccountSize", "DDType", "Mode"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # validate dd types
    allowed = {"TRUE_TRAIL", "EOD_TRAIL", "BALANCE_TRAIL", "STATIC"}
    bad = df.loc[~df["DDType"].isin(allowed), "DDType"].dropna().unique().tolist()
    if bad:
        raise ValueError(f"Invalid DDType(s) found: {bad}. Allowed: {sorted(allowed)}")

    return df


def get_firms(df: pd.DataFrame) -> list[str]:
    return sorted(df["Firm"].dropna().unique().tolist())


def get_accounts_for_firm(df: pd.DataFrame, firm: str) -> list[str]:
    sub = df[df["Firm"] == firm]
    return sub["AccountSize"].dropna().unique().tolist()


def get_risk_profiles_for_firm(df: pd.DataFrame, firm: str) -> pd.DataFrame:
    """
    Returns rows that have Mode + RiskFracDailyDDPerTrade for this firm.
    Some firms may only have these set for one account size (your data is fine).
    """
    sub = df[(df["Firm"] == firm) & df["Mode"].notna() & df["RiskFracDailyDDPerTrade"].notna()].copy()
    if sub.empty:
        # default fallback profile if none provided
        return pd.DataFrame(
            [{"Mode": "Safe", "RiskFracDailyDDPerTrade": 0.10},
             {"Mode": "Standard", "RiskFracDailyDDPerTrade": 0.15},
             {"Mode": "Aggressive", "RiskFracDailyDDPerTrade": 0.25}]
        )
    return sub[["Mode", "RiskFracDailyDDPerTrade"]].drop_duplicates().reset_index(drop=True)


def get_rule(df: pd.DataFrame, firm: str, account_size: str) -> FirmRule:
    sub = df[(df["Firm"] == firm) & (df["AccountSize"] == account_size)]
    if sub.empty:
        raise KeyError(f"No config row found for firm={firm!r}, account_size={account_size!r}")

    row = sub.iloc[0]

    daily = None if pd.isna(row.get("DailyMaxLoss")) else float(row["DailyMaxLoss"])
    total = None if pd.isna(row.get("TotalMaxDD")) else float(row["TotalMaxDD"])
    target = None if pd.isna(row.get("ProfitTarget")) else float(row["ProfitTarget"])
    max_minis = None if pd.isna(row.get("FirmMaxContracts")) else int(row["FirmMaxContracts"])

    return FirmRule(
        firm=str(row["Firm"]),
        account_size=str(row["AccountSize"]),
        daily_max_loss=daily,
        total_max_dd=total,
        profit_target=target,
        firm_max_contracts_minis=max_minis,
        dd_type=str(row["DDType"]),  # type: ignore
    )
