from __future__ import annotations

import pandas as pd

from indicators import rolling_pct_rank


def rates_track(y2: pd.Series, y10: pd.Series, y30: pd.Series, p: dict) -> pd.DataFrame:
    """Ставки по днях FRED (до вирівнювання на календар NYSE)."""
    n = p["chg_window"]
    y2, y10, y30 = y2.dropna(), y10.dropna(), y30.dropna()
    rv = (y10.diff() * 100).rolling(p["vol_window"]).std()     # реалізована вол. 10Y, б.п./день
    df = pd.DataFrame({
        "y2": y2, "y10": y10, "y30": y30,
        "y10_chg_bp": (y10 - y10.shift(n)) * 100,
        "rv10y_bp": rv,
        "rv_pct_1y": rolling_pct_rank(rv, p["z_window"]),
        "y30_pct_5y": rolling_pct_rank(y30, p["pct_window_5y"], p["pct_min_periods"]),
    })
    df["s2s10_bp"] = (df["y10"] - df["y2"]) * 100
    df["s10s30_bp"] = (df["y30"] - df["y10"]) * 100
    return df
