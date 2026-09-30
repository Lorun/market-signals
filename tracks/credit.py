from __future__ import annotations

import pandas as pd

from indicators import zscore, pct_rank, trend3, r


def credit_track(oas_ig: pd.Series, oas_bbb: pd.Series, p) -> dict:
    n = p["chg_window"]
    ig = oas_ig * 100  # bp
    bbb = oas_bbb.reindex(ig.index).ffill() * 100
    gap = bbb - ig
    sma_f, sma_s = ig.rolling(p["sma_fast"]).mean(), ig.rolling(p["sma_slow"]).mean()
    chg = ig.iloc[-1] - ig.iloc[-1 - n]
    gap_chg = gap.iloc[-1] - gap.iloc[-1 - n]
    z = zscore(ig, p["z_window"])
    # прискорення: зміна за останні n днів проти попередніх n днів
    prev_chg = ig.iloc[-1 - n] - ig.iloc[-1 - 2 * n]
    accel = chg - prev_chg

    states = {
        "oas_trend": trend3(chg, p["oas_chg_bp"], "widening", "tightening"),
        "oas_vs_sma50": "above" if ig.iloc[-1] > sma_f.iloc[-1] else "below",
        "oas_sma_cross": "widening_regime" if sma_f.iloc[-1] > sma_s.iloc[-1] else "tightening_regime",
        "bbb_decompression": trend3(gap_chg, p["bbb_gap_chg_bp"], "decompressing", "compressing"),
        "oas_extreme": "wide" if z >= p["z_threshold"] else "tight" if z <= -p["z_threshold"] else "normal",
        "oas_acceleration": trend3(accel, p["oas_chg_bp"] / 2, "accelerating_wider", "accelerating_tighter"),
    }
    # + = розширення спреду = на користь TLT
    votes = {
        "oas_trend": {"widening": 1, "tightening": -1},
        "oas_vs_sma50": {"above": 1, "below": -1},
        "oas_sma_cross": {"widening_regime": 1, "tightening_regime": -1},
        "bbb_decompression": {"decompressing": 1, "compressing": -1},
        "oas_acceleration": {"accelerating_wider": 1, "accelerating_tighter": -1},
    }
    score = sum(votes[k].get(states[k], 0) for k in votes)
    metrics = {
        "oas_ig_bp": r(ig.iloc[-1], 1), "oas_ig_chg_bp": r(chg, 1),
        "oas_ig_sma50": r(sma_f.iloc[-1], 1), "oas_ig_sma200": r(sma_s.iloc[-1], 1),
        "oas_ig_z_1y": r(z), "oas_ig_pct_1y": r(pct_rank(ig, p["z_window"]), 0),
        "oas_bbb_bp": r(bbb.iloc[-1], 1), "bbb_ig_gap_bp": r(gap.iloc[-1], 1),
        "bbb_ig_gap_chg_bp": r(gap_chg, 1), "oas_accel_bp": r(accel, 1),
    }
    return {"score": score, "max_score": len(votes), "states": states, "metrics": metrics}
