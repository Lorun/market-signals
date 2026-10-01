from __future__ import annotations

import pandas as pd

from indicators import rsi, divergence_series, streak


def core_track(tlt: pd.Series, p: dict) -> pd.DataFrame:
    """Сигнальний ряд ядра (TLT, скоригований) — по днях."""
    sma_f, sma_s = tlt.rolling(p["sma_fast"]).mean(), tlt.rolling(p["sma_slow"]).mean()
    rs = rsi(tlt, p["rsi_len"])
    above = (tlt > sma_s) & sma_s.notna()
    return pd.DataFrame({
        "core_px": tlt,
        "core_sma50": sma_f,
        "core_sma200": sma_s,
        "core_dist_pct": (tlt / sma_s - 1) * 100,         # TLT vs SMA200, %
        "core_sma_gap_pct": (sma_f / sma_s - 1) * 100,    # SMA50 vs SMA200, %
        "core_above": above,
        "core_above_streak": streak(above),
        "core_downtrend": (tlt < sma_s) & (sma_f < sma_s),
        "core_rsi": rs,
        "core_divergence": divergence_series(tlt, rs, p["divergence_lookback"],
                                             p["divergence_recent"], p["divergence_rsi_margin"]),
    })
