"""Тести векторних треків і склейки рядів."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indicators import divergence, divergence_series, pct_rank, rolling_pct_rank, rsi, streak  # noqa: E402
from inputs import build_inputs, exec_days  # noqa: E402
from sources import build_block_prices, chain_returns, demo_data, rate_to_returns  # noqa: E402
from tracks.cta import _bucket, score_to_weight  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_divergence_series_matches_scalar():
    prices, _ = demo_data(CFG)
    px = prices["core"].iloc[:800]
    rs = rsi(px, 14)
    vec = divergence_series(px, rs, 60, 5, 3)
    for t in range(100, len(px), 7):
        assert vec.iloc[t] == divergence(px.iloc[:t + 1], rs.iloc[:t + 1], 60, 5, 3), t


def test_rolling_pct_rank_matches_scalar():
    s = pd.Series(np.random.default_rng(1).normal(size=400))
    roll = rolling_pct_rank(s, 252)
    assert roll.iloc[-1] == pytest.approx(pct_rank(s, 252))
    assert roll.iloc[300] == pytest.approx(pct_rank(s.iloc[:301], 252))


def test_streak():
    s = pd.Series([True, True, False, True, True, True])
    assert streak(s).tolist() == [1, 2, 0, 1, 2, 3]


def test_chain_returns_continuous():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(np.linspace(10, 19, 10), idx)        # старий фонд
    b = pd.Series(np.linspace(500, 590, 10), idx)      # новий фонд, інший рівень
    out = chain_returns([(a, idx[0]), (b, idx[5])])
    assert out.iloc[0] == pytest.approx(100 * (11 / 10))
    r = out.pct_change().dropna()
    assert r.loc[idx[4]] == pytest.approx(a.iloc[4] / a.iloc[3] - 1)
    assert r.loc[idx[6]] == pytest.approx(b.iloc[6] / b.iloc[5] - 1)
    assert r.loc[idx[5]] == pytest.approx(b.iloc[5] / b.iloc[4] - 1)   # стик — уже за новим рядом


def test_rate_to_returns():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08"])  # п'ятниця → понеділок, 3 дні
    r = rate_to_returns(pd.Series([3.6, 3.6], idx))
    assert r.iloc[0] == pytest.approx(0.036 * 3 / 360)


@pytest.mark.parametrize("x,expected", [(-1, 0), (0, 1), (2.99, 1), (3, 2), (np.nan, 0)])
def test_bucket(x, expected):
    assert _bucket(pd.Series([x]), 0, 3).iloc[0] == expected


@pytest.mark.parametrize("score,w", [(0, 5), (3, 5), (4, 10), (5, 10), (6, 15), (7, 15), (8, 20), (9, 20), (10, 25)])
def test_cta_score_table(score, w):
    assert score_to_weight(score, CFG["strategy"]["cta_score_weights"]) == w


def test_exec_days_last_trading_day_of_week():
    idx = pd.to_datetime(["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28",   # Good Friday 29.03 вихідний
                          "2024-04-01", "2024-04-05"])
    assert exec_days(idx, live=False).tolist() == [False, False, False, True, False, True]


def test_build_inputs_demo():
    prices, fred = demo_data(CFG)
    df = build_inputs(build_block_prices(prices, fred, CFG, backtest=False), fred, CFG)
    last = df.iloc[-1]
    assert 0 <= last["cta_score"] <= 10
    assert last["cta_score"] == sum(last[f"cta_c_{c}"] for c in
                                    ["tlt_vs_sma200", "sma50_vs_sma200", "y10_chg", "rates_vol", "dbmf_vs_sma100"])
    for col in ("stress", "stress_rates", "stress_credit", "stress_vix", "core_above_streak", "eq_above_streak"):
        assert col in df


def test_lag_series_next_business_day():
    from sources import lag_series
    idx = pd.to_datetime(["2026-10-01", "2026-10-02"])          # чт, пт
    out = lag_series(pd.Series([1.0, 2.0], idx), 1)
    assert out.index.tolist() == pd.to_datetime(["2026-10-02", "2026-10-05"]).tolist()   # пт → пн


def test_build_inputs_applies_fred_lag_only_to_listed_series():
    prices, fred = demo_data(CFG)
    blocks = build_block_prices(prices, fred, CFG, backtest=False)
    df = build_inputs(blocks, fred, CFG)
    d = df.index[-5]
    prev = df.index[df.index < d][-1]
    assert df.loc[d, "credit_bp"] == pytest.approx(fred["oas_ig"].loc[prev] * 100)   # IG OAS — вчорашній
    assert df.loc[d, "vix"] == pytest.approx(fred["vix"].loc[d])                     # VIX — того ж дня
    assert df.loc[d, "y10"] == pytest.approx(fred["y10"].loc[d])                     # 10Y — того ж дня


def test_lag_series_weekend_observation_no_duplicates():
    from sources import lag_series
    idx = pd.to_datetime(["2026-10-02", "2026-10-03"])          # пт, сб
    out = lag_series(pd.Series([1.0, 2.0], idx), 1)
    assert out.index.is_unique and out.loc["2026-10-05"] == 2.0
