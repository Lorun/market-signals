from __future__ import annotations

import pandas as pd

from indicators import rolling_pct_rank, rolling_z


def credit_metrics(credit: pd.Series, p: dict) -> pd.DataFrame:
    """IG OAS (або проксі BAA10Y у бектесті), б.п., і z-score за рік — по днях FRED."""
    c = credit.dropna() * 100
    return pd.DataFrame({"credit_bp": c, "credit_z": rolling_z(c, p["z_window"])})


def vix_metrics(vix: pd.Series, p: dict) -> pd.DataFrame:
    v = vix.dropna()
    return pd.DataFrame({"vix": v, "vix_pct_1y": rolling_pct_rank(v, p["z_window"])})


def stress_track(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """§2: три умови Стресу окремими прапорцями. `df` уже вирівняний (rates + credit + vix)."""
    out = pd.DataFrame(index=df.index)
    out["stress_rates"] = (df["y10_chg_bp"] >= s["stress_y10_chg_bp"]) & (df["rv_pct_1y"] >= s["stress_vol_pct"])
    out["stress_credit"] = df["credit_z"] >= s["stress_credit_z"]
    out["stress_vix"] = df["vix_pct_1y"] >= s["stress_vix_pct"]
    out["stress"] = out[["stress_rates", "stress_credit", "stress_vix"]].any(axis=1)
    return out
