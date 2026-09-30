from __future__ import annotations

import pandas as pd

from indicators import rsi, macd_hist, divergence, pct_rank, trend3, r


def rates_track(ratio: pd.Series, y2, y10, y30, p) -> dict:
    n = p["chg_window"]
    sma_f, sma_s = ratio.rolling(p["sma_fast"]).mean(), ratio.rolling(p["sma_slow"]).mean()
    rs = rsi(ratio, p["rsi_len"])
    mh = macd_hist(ratio)

    rng = ratio.iloc[-p["range_len"] - p["range_skip"]:-p["range_skip"]]
    last = ratio.iloc[-1]
    range_state = "above" if last > rng.max() else "below" if last < rng.min() else "inside"

    y10_chg = (y10.iloc[-1] - y10.iloc[-1 - n]) * 100
    s2s10 = (y10 - y2) * 100
    s10s30 = (y30 - y10) * 100
    c1 = s2s10.iloc[-1] - s2s10.iloc[-1 - n]
    c2 = s10s30.iloc[-1] - s10s30.iloc[-1 - n]

    rv = (y10.diff() * 100).rolling(p["vol_window"]).std()
    rv_pct = pct_rank(rv, p["z_window"])
    vol_state = "high" if rv_pct >= p["vol_high_pct"] else "low" if rv_pct <= p["vol_low_pct"] else "normal"

    states = {
        "ratio_vs_sma200": "above" if last > sma_s.iloc[-1] else "below",
        "sma_cross": "golden" if sma_f.iloc[-1] > sma_s.iloc[-1] else "death",
        "macd": "positive" if mh.iloc[-1] > 0 else "negative",
        "rsi_zone": "overbought" if rs.iloc[-1] >= 70 else "oversold" if rs.iloc[-1] <= 30 else "neutral",
        "rsi_divergence": divergence(ratio, rs, p["divergence_lookback"],
                                     p["divergence_recent"], p["divergence_rsi_margin"]),
        "range_breakout": range_state,
        "y10_trend": trend3(y10_chg, p["yield_chg_bp"], "rising", "falling"),
        "curve_2s10s": trend3(c1, p["curve_chg_bp"], "steepening", "flattening"),
        "curve_10s30s": trend3(c2, p["curve_chg_bp"], "steepening", "flattening"),
        "rates_vol": vol_state,
    }
    # голоси: + = на користь TLT
    votes = {
        "ratio_vs_sma200": {"above": 1, "below": -1},
        "sma_cross": {"golden": 1, "death": -1},
        "macd": {"positive": 1, "negative": -1},
        "rsi_divergence": {"bullish": 1, "bearish": -1},
        "range_breakout": {"above": 1, "below": -1},
        "y10_trend": {"falling": 1, "rising": -1},
        "curve_10s30s": {"flattening": 1, "steepening": -1},
    }
    score = sum(votes[k].get(states[k], 0) for k in votes)
    metrics = {
        "ratio": r(last, 5), "sma50": r(sma_f.iloc[-1], 5), "sma200": r(sma_s.iloc[-1], 5),
        "ratio_chg_20d_pct": r((last / ratio.iloc[-1 - n] - 1) * 100),
        "rsi": r(rs.iloc[-1], 1), "macd_hist": r(mh.iloc[-1], 6),
        "range_high": r(rng.max(), 5), "range_low": r(rng.min(), 5),
        "y2": r(y2.iloc[-1]), "y10": r(y10.iloc[-1]), "y30": r(y30.iloc[-1]),
        "y10_chg_bp": r(y10_chg, 1),
        "s2s10_bp": r(s2s10.iloc[-1], 1), "s2s10_chg_bp": r(c1, 1),
        "s10s30_bp": r(s10s30.iloc[-1], 1), "s10s30_chg_bp": r(c2, 1),
        "rv10y_bp": r(rv.iloc[-1], 2), "rv10y_pct_1y": r(rv_pct, 0),
    }
    return {"score": score, "max_score": len(votes), "states": states, "metrics": metrics}
