"""High Yield (XUHA) track.

Знак скорингу: "+" = сприятливо для XUHA (risk-on), "−" = несприятливо.
Це ПРОТИЛЕЖНА конвенція відносно треків TLT/SUOA (там + = на користь TLT).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import rsi, macd_hist, divergence, zscore, pct_rank, trend3, r


def high_yield_track(
    xuha: pd.Series,
    suoa: pd.Series,
    hy_oas: pd.Series,
    hy_yield: pd.Series,
    hy_bb: pd.Series,
    hy_ccc: pd.Series,
    oas_ig: pd.Series,
    vix: pd.Series,
    nfci: pd.Series,
    sloos: pd.Series,
    oil: pd.Series,
    p: dict,
) -> dict:
    n = p["chg_window"]

    # ── XUHA price indicators ──────────────────────────────────────
    # Forward-fill sparse NaN gaps (LSE vs NYSE/SIX calendar mismatches)
    xuha_px = xuha.ffill()
    sma50 = xuha_px.rolling(p["sma_fast"]).mean()
    sma200 = xuha_px.rolling(p["sma_slow"]).mean()
    rs = rsi(xuha_px, p["rsi_len"])
    mh = macd_hist(xuha_px)
    xuha_last = xuha_px.iloc[-1]

    # 52-week high for drawdown (z_window ≈ 252 trading days)
    high_52w = xuha_px.rolling(p["z_window"]).max().iloc[-1]
    dd_pct = (xuha_last / high_52w - 1) * 100 if high_52w > 0 else 0.0

    # range breakout (same as rates_track)
    rng = xuha_px.iloc[-p["range_len"] - p["range_skip"]:-p["range_skip"]]

    # ── HY/IG ratio (XUHA/SUOA on shared trading days) ────────────
    common_px = pd.concat([xuha_px.rename("xuha"), suoa.ffill().rename("suoa")], axis=1).dropna()
    ratio_hy = (common_px["xuha"] / common_px["suoa"]).rename("ratio_hy")
    ratio_sma200 = ratio_hy.rolling(p["sma_slow"]).mean()

    # ── Spread series in basis points ─────────────────────────────
    hy_bp = hy_oas.ffill(limit=5) * 100
    bb_bp = hy_bb.reindex(hy_bp.index).ffill(limit=5) * 100
    ccc_bp = hy_ccc.reindex(hy_bp.index).ffill(limit=5) * 100
    ig_bp = oas_ig.reindex(hy_bp.index).ffill(limit=5) * 100
    hy_yield_s = hy_yield.reindex(hy_bp.index).ffill(limit=5)

    # Align all daily series to XUHA price dates
    def align(s: pd.Series) -> pd.Series:
        return s.reindex(xuha.index).ffill(limit=5)

    hy_bp_a = align(hy_bp)
    bb_bp_a = align(bb_bp)
    ccc_bp_a = align(ccc_bp)
    ig_bp_a = align(ig_bp)
    hy_yield_a = align(hy_yield_s)
    vix_a = align(vix.ffill(limit=5))
    oil_a = align(oil.ffill(limit=5))

    hy_chg = hy_bp_a.iloc[-1] - hy_bp_a.iloc[-1 - n]
    ccc_bb = ccc_bp_a - bb_bp_a
    ccc_bb_chg = ccc_bb.iloc[-1] - ccc_bb.iloc[-1 - n]

    hy_sma50 = hy_bp_a.rolling(p["sma_fast"]).mean()
    hy_sma200 = hy_bp_a.rolling(p["sma_slow"]).mean()

    # ── VIX percentile ────────────────────────────────────────────
    vix_pct = pct_rank(vix_a.dropna(), p["z_window"])

    # ── NFCI (weekly — use index positions, not business days) ────
    nfci_clean = nfci.dropna()
    nfci_last = float(nfci_clean.iloc[-1])
    lookback = min(5, len(nfci_clean))
    nfci_4w_ago = float(nfci_clean.iloc[-lookback])
    nfci_chg = nfci_last - nfci_4w_ago

    # ── SLOOS (quarterly) ─────────────────────────────────────────
    sloos_clean = sloos.dropna()
    sloos_last = float(sloos_clean.iloc[-1])
    sloos_prev = float(sloos_clean.iloc[-2]) if len(sloos_clean) >= 2 else sloos_last
    sloos_chg = sloos_last - sloos_prev

    # ── Oil 20-day % change ───────────────────────────────────────
    oil_chg_pct = (oil_a.iloc[-1] / oil_a.iloc[-1 - n] - 1) * 100 if oil_a.iloc[-1 - n] > 0 else 0.0

    # ── HY OAS z-score ────────────────────────────────────────────
    hy_z = zscore(hy_bp_a.dropna(), p["z_window"])
    hy_pct = pct_rank(hy_bp_a.dropna(), p["z_window"])

    # ══ VOTING STATES ════════════════════════════════════════════
    sv = {}  # states with votes

    sv["price_vs_sma200"] = "above" if xuha_last > sma200.iloc[-1] else "below"

    sv["sma_cross"] = "golden" if sma50.iloc[-1] > sma200.iloc[-1] else "death"

    sv["macd"] = "positive" if mh.iloc[-1] > 0 else "negative"

    sv["rsi_divergence"] = divergence(
        xuha_px, rs, p["divergence_lookback"], p["divergence_recent"], p["divergence_rsi_margin"]
    )

    sv["hy_vs_ig_ratio"] = "above" if ratio_hy.iloc[-1] > ratio_sma200.iloc[-1] else "below"

    # tightening = OAS fell = good for XUHA → +1; negate chg so trend3 maps correctly
    sv["hy_oas_trend"] = trend3(-hy_chg, p["hy_oas_chg_bp"], "tightening", "widening")

    sv["hy_oas_regime"] = (
        "tightening_regime" if hy_sma50.iloc[-1] <= hy_sma200.iloc[-1] else "widening_regime"
    )

    # compressing = CCC-BB gap fell = good for XUHA → +1; negate change
    sv["ccc_bb_decompression"] = trend3(-ccc_bb_chg, p["hy_oas_chg_bp"], "compressing", "decompressing")

    if vix_pct <= p["vix_low_pct"]:
        sv["vix_regime"] = "low"
    elif vix_pct >= p["vix_high_pct"]:
        sv["vix_regime"] = "elevated"
    else:
        sv["vix_regime"] = "normal"

    if nfci_last < 0 and nfci_chg < -0.1:
        sv["financial_conditions"] = "easing"
    elif nfci_last > 0 or nfci_chg > 0.1:
        sv["financial_conditions"] = "tightening"
    else:
        sv["financial_conditions"] = "stable"

    votes = {
        "price_vs_sma200":    {"above": 1, "below": -1},
        "sma_cross":          {"golden": 1, "death": -1},
        "macd":               {"positive": 1, "negative": -1},
        "rsi_divergence":     {"bullish": 1, "bearish": -1},
        "hy_vs_ig_ratio":     {"above": 1, "below": -1},
        "hy_oas_trend":       {"tightening": 1, "widening": -1},
        "hy_oas_regime":      {"tightening_regime": 1, "widening_regime": -1},
        "ccc_bb_decompression": {"compressing": 1, "decompressing": -1},
        "vix_regime":         {"low": 1, "elevated": -1},
        "financial_conditions": {"easing": 1, "tightening": -1},
    }
    score = sum(votes[k].get(sv[k], 0) for k in votes)

    # ══ CONTEXT STATES (no votes) ═════════════════════════════════
    ctx = {}

    ctx["drawdown"] = "significant" if dd_pct < -5 else "mild" if dd_pct < -2 else "none"

    ctx["hy_oas_extreme"] = (
        "wide" if hy_z >= p["z_threshold"] else "tight" if hy_z <= -p["z_threshold"] else "normal"
    )

    if sloos_chg > 1.0:
        ctx["lending_standards"] = "tightening"
    elif sloos_chg < -1.0:
        ctx["lending_standards"] = "easing"
    else:
        ctx["lending_standards"] = "stable"

    ctx["oil_trend"] = trend3(oil_chg_pct, p["oil_chg_pct"], "rising", "falling")

    ctx["range_breakout"] = (
        "above" if xuha_last > rng.max() else "below" if xuha_last < rng.min() else "inside"
    )

    # ══ METRICS ══════════════════════════════════════════════════
    metrics = {
        "xuha": r(xuha_last),
        "sma50": r(sma50.iloc[-1]),
        "sma200": r(sma200.iloc[-1]),
        "rsi": r(rs.iloc[-1], 1),
        "drawdown_pct": r(dd_pct),
        "xuha_suoa_ratio": r(ratio_hy.iloc[-1], 5),
        "hy_oas_bp": r(hy_bp_a.iloc[-1], 1),
        "hy_oas_chg_bp": r(hy_chg, 1),
        "hy_oas_z_1y": r(hy_z),
        "hy_oas_pct_1y": r(hy_pct, 0),
        "hy_bb_bp": r(bb_bp_a.iloc[-1], 1),
        "hy_ccc_bp": r(ccc_bp_a.iloc[-1], 1),
        "ccc_bb_bp": r(ccc_bb.iloc[-1], 1),
        "hy_ig_gap_bp": r(hy_bp_a.iloc[-1] - ig_bp_a.iloc[-1], 1),
        "hy_yield_pct": r(hy_yield_a.iloc[-1]),
        "vix": r(vix_a.iloc[-1], 1),
        "vix_pct_1y": r(vix_pct, 0),
        "nfci": r(nfci_last, 2),
        "nfci_4w_chg": r(nfci_chg, 2),
        "sloos_last": r(sloos_last, 1),
        "oil": r(oil_a.iloc[-1], 1),
        "oil_chg_20d_pct": r(oil_chg_pct),
        "xuha_date": str(xuha.dropna().index[-1].date()),
        "hy_oas_date": str(hy_oas.index[-1].date()),
        "nfci_date": str(nfci_clean.index[-1].date()),
        "sloos_date": str(sloos_clean.index[-1].date()),
    }

    return {
        "score": score,
        "max_score": len(votes),
        "states": sv,
        "context": ctx,
        "metrics": metrics,
    }


def hy_label(score: int, mx: int) -> str:
    if score >= mx * 0.5:
        return "сприятливо для XUHA"
    if score > 0:
        return "помірно сприятливо для XUHA"
    if score <= -mx * 0.5:
        return "несприятливо для XUHA"
    if score < 0:
        return "помірно несприятливо для XUHA"
    return "нейтрально"
