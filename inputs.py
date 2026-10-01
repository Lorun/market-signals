"""Збирає всі треки в один DataFrame по днях (календар NYSE) — вхід рушія.

Спільний для живого звіту (signals.py) і бектесту (backtest.py)."""
from __future__ import annotations

import pandas as pd

from sources import align
from tracks import core_track, rates_track, stress_track, credit_metrics, vix_metrics, cta_track, equity_track

BLOCKS = ("core", "equity", "cta", "cash")


def exec_days(index: pd.DatetimeIndex, live: bool) -> pd.Series:
    """Останній торговий день ISO-тижня. Для останнього рядка в живому режимі майбутнє невідоме —
    тоді день виконання = п'ятниця."""
    iso = pd.Series([d.isocalendar()[:2] for d in index], index=index)
    nxt = iso.shift(-1)
    out = pd.Series([a != b for a, b in zip(iso, nxt)], index=index)
    if live and len(index):
        out.iloc[-1] = index[-1].weekday() == 4
    return out


def build_inputs(blocks: pd.DataFrame, fred: dict[str, pd.Series], cfg: dict,
                 credit_key: str = "oas_ig", live: bool = True) -> pd.DataFrame:
    """blocks — індекси блоків (core/cta/equity/cash) за сирими датами бірж;
    fred — сирі ряди FRED; credit_key — ряд для умови Стресу IG OAS (у бектесті — baa10y)."""
    p, s = cfg["params"], cfg["strategy"]
    lim = p["ffill_limit"]
    core = blocks["core"].dropna()
    idx = core.index

    px = align(blocks.drop(columns="core"), idx, lim)
    px["core"] = core

    rates = align(rates_track(fred["y2"], fred["y10"], fred["y30"], p), idx, lim)
    credit = align(credit_metrics(fred[credit_key], p), idx, lim)
    vix = align(vix_metrics(fred["vix"], p), idx, lim)

    df = pd.concat([core_track(core, p), rates, credit, vix], axis=1)
    df = pd.concat([df, stress_track(df, s)], axis=1)
    df = pd.concat([df, cta_track(df, px["cta"], p, s), equity_track(px["equity"].dropna(), p).reindex(idx)], axis=1)
    df["eq_above_streak"] = df["eq_above_streak"].fillna(0).astype(int)
    df["eq_above"] = df["eq_above"].eq(True)

    for b in BLOCKS:
        df[f"ret_{b}"] = px[b].pct_change()
        df[f"px_{b}"] = px[b]
    if "cta_exec" in px:
        n = p["chg_window"]
        ex = px["cta_exec"]
        df["cta_exec_gap_pct"] = ((ex / ex.shift(n)) / (px["cta"] / px["cta"].shift(n)) - 1) * 100
    df["is_exec_day"] = exec_days(idx, live)
    return df


def first_valid_date(df: pd.DataFrame) -> pd.Timestamp:
    """Перший день, коли є все потрібне рушію (SMA200 ядра, ставки, ціни й дохідності всіх блоків)."""
    need = ["core_sma200", "y10_chg_bp", "rv_pct_1y", "cta_sma100"] + [f"px_{b}" for b in BLOCKS]
    ok = df[need].notna().all(axis=1)
    return ok[ok].index[0]
