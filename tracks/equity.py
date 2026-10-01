from __future__ import annotations

import pandas as pd

from indicators import streak


def equity_track(px: pd.Series, p: dict) -> pd.DataFrame:
    """Акції (SSAC.L / ACWI у бектесті) vs SMA200 — по днях."""
    sma = px.rolling(p["sma_slow"]).mean()
    above = (px > sma) & sma.notna()
    return pd.DataFrame({
        "eq_px": px,
        "eq_sma200": sma,
        "eq_dist_pct": (px / sma - 1) * 100,
        "eq_above": above,
        "eq_above_streak": streak(above),
    })
