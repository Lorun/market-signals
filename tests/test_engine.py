"""Тести рушія правил на синтетичних днях (docs/strategy.md §2–§7)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (DayInputs, EngineState, StrategyConfig, base_state, portfolio_state,  # noqa: E402
                    step, update_regime)

CFG = StrategyConfig()
MON = pd.Timestamp("2024-01-08")          # понеділок
ZERO = {"core": 0.0, "equity": 0.0, "cta": 0.0, "cash": 0.0}


def day(date, **kw) -> DayInputs:
    date = pd.Timestamp(date)
    base = dict(date=date, is_exec_day=date.weekday() == 4, ret=dict(ZERO), core_px=100.0)
    base.update(kw)
    return DayInputs(**base)


def fri(n: int) -> pd.Timestamp:
    """n-та п'ятниця від MON (0 = перша)."""
    return MON + pd.Timedelta(days=4 + 7 * n)


def state(**kw) -> EngineState:
    s = base_state(CFG, day(MON))
    s = dataclasses.replace(s, date=MON)
    if "weights" in kw:
        w = dict(kw.pop("weights"))
        w["cash"] = 100 - w["core"] - w["equity"] - w["cta"]
        kw["weights"] = w
    return dataclasses.replace(s, **kw)


def acts_by_block(actions):
    return {a.block: a for a in actions}


# ───────────── сходинки ─────────────

@pytest.mark.parametrize("px,downtrend,expected", [
    (93.0, True, 50), (86.0, True, 40), (80.0, True, 30),
    (80.0, False, 40),      # −20% без тренду вниз → лише сходинка −14%
    (94.0, True, None),     # −6% — ще не сходинка
])
def test_ladder_levels(px, downtrend, expected):
    s = state(regime="rates_up")
    s2, acts = step(s, day(fri(0), core_px=px, downtrend=downtrend), CFG)
    a = acts_by_block(acts)
    if expected is None:
        assert "core" not in a
    else:
        assert a["core"].to_weight == pytest.approx(expected)
        assert "сходинка" in a["core"].reason


def test_ladder_only_in_rates_up_or_stress():
    s2, acts = step(state(regime="base"), day(fri(0), core_px=85.0, downtrend=True), CFG)
    assert "core" not in acts_by_block(acts)


def test_ladder_waits_for_exec_day_unless_stress():
    wed = fri(0) - pd.Timedelta(days=2)
    _, acts = step(state(regime="rates_up"), day(wed, core_px=92.0, downtrend=True), CFG)
    assert not acts
    _, acts = step(state(regime="stress"), day(wed, core_px=92.0, downtrend=True, stress=True), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(50)


# ───────────── пауза «схоже на дно» ─────────────

BOTTOM = dict(y30_pct_5y=95.0, y10_chg_bp=3.0)     # 2 з 3 ознак


def test_pause_defers_but_does_not_cancel():
    s = state(regime="rates_up")
    s, acts = step(s, day(fri(0), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    assert not acts and s.pause_step == 1 and s.pause_until is not None
    s, acts = step(s, day(fri(2), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    assert not acts                                   # пауза триває
    s, acts = step(s, day(fri(5), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(50)   # > 4 тижнів — продаж


def test_pause_ends_when_signs_vanish():
    s = state(regime="rates_up")
    s, _ = step(s, day(fri(0), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    s, acts = step(s, day(fri(1), core_px=92.0, downtrend=True), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(50)


def test_pause_ends_when_fall_continues():
    s = state(regime="rates_up")
    s, _ = step(s, day(fri(0), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    s, acts = step(s, day(fri(1), core_px=85.0, downtrend=True, **BOTTOM), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(40)


def test_each_step_paused_once():
    s = state(regime="rates_up")
    s, _ = step(s, day(fri(0), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    s, _ = step(s, day(fri(1), core_px=95.0, downtrend=True, **BOTTOM), CFG)   # відскок, сходинки немає
    s, acts = step(s, day(fri(6), core_px=92.0, downtrend=True, **BOTTOM), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(50)          # повторно не паузиться


# ───────────── повернення в ядро і скидання піку ─────────────

def _after_ladder() -> EngineState:
    return state(regime="base", weights={"core": 30, "equity": 15, "cta": 5},
                 targets={"core": 30.0, "equity": 15.0, "cta": 5.0}, ladder_step=3)


def test_return_not_more_often_than_two_weeks():
    s = _after_ladder()
    up = dict(core_above_streak=12, core_px=90.0)
    s, acts = step(s, day(fri(0), **up), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(40)
    s, acts = step(s, day(fri(1), **up), CFG)
    assert "core" not in acts_by_block(acts)
    s, acts = step(s, day(fri(2), **up), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(50)


def test_return_requires_streak():
    _, acts = step(_after_ladder(), day(fri(0), core_above_streak=9, core_px=90.0), CFG)
    assert "core" not in acts_by_block(acts)


def test_peak_reset_after_full_recovery():
    s = state(regime="base", weights={"core": 50, "equity": 15, "cta": 5},
              targets={"core": 50.0, "equity": 15.0, "cta": 5.0}, ladder_step=1,
              paused_steps=frozenset({1}))
    s, acts = step(s, day(fri(0), core_above_streak=15, core_px=95.0), CFG)
    assert acts_by_block(acts)["core"].to_weight == pytest.approx(60)
    assert s.core_peak == pytest.approx(95.0)
    assert s.ladder_step == 0 and s.paused_steps == frozenset()


# ───────────── режими ─────────────

@pytest.mark.parametrize("n", [3, 5, 10])
def test_regime_confirmation(n):
    cfg = dataclasses.replace(CFG, confirm_days=n)
    s = state(regime="base")
    for i in range(n):
        d = day(MON + pd.Timedelta(days=i), stress=True)
        s = dataclasses.replace(s, **dict(zip(("regime", "candidate", "candidate_days"),
                                              update_regime(s, d, cfg))))
        assert s.regime == ("stress" if i == n - 1 else "base")


def test_regime_candidate_resets_on_interruption():
    s = state(regime="base", candidate="stress", candidate_days=4)
    r, c, k = update_regime(s, day(MON, downtrend=True), CFG)
    assert (r, c, k) == ("base", "rates_up", 1)


def test_stress_priority():
    r, c, k = update_regime(state(regime="base"), day(MON, stress=True, downtrend=True),
                            dataclasses.replace(CFG, confirm_days=1))
    assert r == "stress"


def test_stress_blocks_return_and_equity_extension():
    s = dataclasses.replace(_after_ladder(), regime="stress", ladder_step=0,
                            targets={"core": 30.0, "equity": 15.0, "cta": 5.0})
    _, acts = step(s, day(fri(0), stress=True, core_above_streak=30, eq_above_streak=30, core_px=90.0), CFG)
    a = acts_by_block(acts)
    assert "core" not in a and "equity" not in a


# ───────────── акції ─────────────

def test_equity_extension_in_tranches():
    s = state(regime="base")
    s, acts = step(s, day(fri(0), eq_above_streak=6), CFG)
    assert acts_by_block(acts)["equity"].to_weight == pytest.approx(20)
    s, acts = step(s, day(fri(1), eq_above_streak=11), CFG)
    assert acts_by_block(acts)["equity"].to_weight == pytest.approx(25)
    s, acts = step(s, day(fri(2), eq_above_streak=16), CFG)
    assert "equity" not in acts_by_block(acts)


def test_equity_base_never_sold_by_trend():
    s = state(regime="rates_up")
    _, acts = step(s, day(fri(0), eq_above_streak=0, downtrend=True), CFG)
    assert "equity" not in acts_by_block(acts)
    s = state(regime="base", weights={"core": 60, "equity": 25, "cta": 5},
              targets={"core": 60.0, "equity": 25.0, "cta": 5.0}, equity_extra=10.0)
    s2, acts = step(s, day(fri(0), eq_above_streak=0), CFG)
    a = acts_by_block(acts)["equity"]
    assert a.to_weight == pytest.approx(15) and "SSAC нижче SMA200" in a.reason
    assert s2.equity_extra == 0


def test_equity_extension_blocked_during_ladder():
    s = state(regime="rates_up", ladder_step=1, targets={"core": 50.0, "equity": 15.0, "cta": 5.0},
              weights={"core": 50, "equity": 15, "cta": 5})
    _, acts = step(s, day(fri(0), eq_above_streak=20), CFG)
    assert "equity" not in acts_by_block(acts)


# ───────────── кеш ─────────────

def test_cash_minimum_has_priority():
    # кеш 18%; бажані покупки: акції +5 (база), CTA +5 → кеш 8% < 10% → урізаються спершу акції
    s = state(regime="rates_up", weights={"core": 62, "equity": 10, "cta": 10},
              targets={"core": 60.0, "equity": 15.0, "cta": 10.0},
              settled={"core": True, "equity": False, "cta": True})
    s2, acts = step(s, day(fri(0), cta_score=10, cta_score_w=25.0), CFG)
    a = acts_by_block(acts)
    assert a["cta"].to_weight == pytest.approx(15)
    assert a["equity"].to_weight == pytest.approx(13)
    assert s2.weights["cash"] >= CFG.cash_min - 1e-9


def test_cash_minimum_blocks_cta_increase():
    s = state(regime="rates_up", weights={"core": 60, "equity": 15, "cta": 14},
              targets={"core": 60.0, "equity": 15.0, "cta": 15.0})
    s2, acts = step(s, day(fri(0), cta_score=10, cta_score_w=25.0), CFG)
    assert "cta" not in acts_by_block(acts)          # кеш 11% → можна лише +1 < 2 п.п. → угоди немає
    assert s2.weights["cash"] >= CFG.cash_min - 1e-9


# ───────────── CTA ─────────────

def test_cta_up_in_tranches_down_at_once():
    s = state(regime="rates_up")
    s, acts = step(s, day(fri(0), cta_score=8, cta_score_w=20.0), CFG)
    assert acts_by_block(acts)["cta"].to_weight == pytest.approx(10)
    s, acts = step(s, day(fri(1), cta_score=8, cta_score_w=20.0), CFG)
    assert acts_by_block(acts)["cta"].to_weight == pytest.approx(15)
    s, acts = step(s, day(fri(2), cta_score=2, cta_score_w=5.0), CFG)
    a = acts_by_block(acts)["cta"]
    assert a.to_weight == pytest.approx(5) and "скоринг CTA 2 → 5%" in a.reason


def test_cta_cut_rule():
    s = state(regime="rates_up", weights={"core": 50, "equity": 15, "cta": 15},
              targets={"core": 50.0, "equity": 15.0, "cta": 15.0})
    _, acts = step(s, day(fri(0), cta_score=7, cta_score_w=15.0, cta_cut=True), CFG)
    a = acts_by_block(acts)["cta"]
    assert a.to_weight == pytest.approx(5) and "зрізання" in a.reason


# ───────────── загальні ─────────────

def test_changes_below_2pp_ignored():
    s = state(regime="base", weights={"core": 61.5, "equity": 15, "cta": 5},
              settled={"core": False, "equity": True, "cta": True})
    s2, acts = step(s, day(fri(0)), CFG)
    assert not acts and s2.settled["core"]


def test_weights_drift_with_returns():
    s = state(regime="base")
    s2, _ = step(s, day(MON + pd.Timedelta(days=1), ret={"core": 0.10, "equity": 0.0, "cta": 0.0, "cash": 0.0}), CFG)
    assert s2.weights["core"] == pytest.approx(66 / 106 * 100)
    assert sum(s2.weights.values()) == pytest.approx(100)


def test_start_transition_from_portfolio():
    s = portfolio_state(CFG, {"core": 70, "cash": 30, "cta": 0, "equity": 0}, 100.0, MON)
    s, acts = step(s, day(fri(0), cta_score=5, cta_score_w=10.0), CFG)
    a = acts_by_block(acts)
    assert a["core"].to_weight == pytest.approx(60)        # захисне — одразу
    assert a["cta"].to_weight == pytest.approx(5)          # мінімум CTA — одразу
    assert a["equity"].to_weight == pytest.approx(5)       # ризик — по 5 п.п.
    s, acts = step(s, day(fri(1), cta_score=5, cta_score_w=10.0), CFG)
    a = acts_by_block(acts)
    assert a["equity"].to_weight == pytest.approx(10) and a["cta"].to_weight == pytest.approx(10)
