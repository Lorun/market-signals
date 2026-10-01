from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ["tlt_vs_sma200", "sma50_vs_sma200", "y10_chg", "rates_vol", "dbmf_vs_sma100"]


def _bucket(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """0 якщо x < lo, 1 якщо lo ≤ x < hi, 2 якщо x ≥ hi; NaN → 0."""
    return pd.Series(np.select([x >= hi, x >= lo], [2, 1], 0), index=x.index).where(x.notna(), 0).astype(int)


def score_to_weight(score: int, table: list) -> float:
    w = table[0][1]
    for mn, wt in table:
        if score >= mn:
            w = wt
    return float(w)


def cta_track(df: pd.DataFrame, cta_px: pd.Series, p: dict, s: dict) -> pd.DataFrame:
    """§5: скоринг CTA 0–10 по днях. `df` — вирівняні колонки core_* і ставок; cta_px — на тому ж індексі."""
    n, below = p["chg_window"], -df["core_dist_pct"]
    gap = -df["core_sma_gap_pct"]
    sma = cta_px.rolling(p["sma_cta"]).mean()
    cta_dist = (cta_px / sma - 1) * 100
    out = pd.DataFrame(index=df.index)
    # компоненти 1/2: "вище" → 0; "нижче < поріг" → 1; "нижче ≥ поріг" → 2
    out["cta_c_tlt_vs_sma200"] = _bucket(below.where(below > 0, -1), 0, s["cta_tlt_below_pct"])
    out["cta_c_sma50_vs_sma200"] = _bucket(gap.where(gap > 0, -1), 0, s["cta_sma_gap_pct"])
    out["cta_c_y10_chg"] = _bucket(df["y10_chg_bp"], *s["cta_y10_bp"])
    out["cta_c_rates_vol"] = _bucket(df["rv_pct_1y"], *s["cta_vol_pct"])
    out["cta_c_dbmf_vs_sma100"] = _bucket(cta_dist.where(cta_dist > 0, -1), 0, s["cta_dbmf_above_pct"])
    out["cta_score"] = out[[f"cta_c_{c}" for c in COMPONENTS]].sum(axis=1).astype(int)
    out["cta_score_w"] = out["cta_score"].map(lambda x: score_to_weight(x, s["cta_score_weights"]))
    out["cta_px"] = cta_px
    out["cta_sma100"] = sma
    out["cta_dist_pct"] = cta_dist
    out["cta_ret20_pct"] = (cta_px / cta_px.shift(n) - 1) * 100
    out["core_ret20_pct"] = (df["core_px"] / df["core_px"].shift(n) - 1) * 100
    out["cta_corr60"] = cta_px.pct_change().rolling(p["corr_window"]).corr(df["core_px"].pct_change())
    # зрізання: за 20 днів і DBMF, і TLT у мінусі, а кореляція за 60 днів > 0 — хедж не хеджує
    out["cta_cut"] = (out["cta_ret20_pct"] < 0) & (out["core_ret20_pct"] < 0) & (out["cta_corr60"] > 0)
    return out
