"""Рушій правил стратегії (docs/strategy.md §2–§7).

Чиста функція `step(state, day, cfg) -> (state, actions)`, яку проганяють по історії день за днем.
Без файлу стану, без `today()`: дата приходить у `DayInputs`. Живий звіт = прогін від portfolio.yaml
до сьогодні; бектест = той самий прогін з 2008/2010 від базового стану.

Ваги — у % портфеля. Між діями ваги дрейфують з дохідностями блоків.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import pandas as pd

BLOCKS = ("core", "equity", "cta", "cash")
TRADED = ("core", "equity", "cta")

REGIME_UA = {
    "stress": "🔴 Стрес",
    "rates_up": "🟠 Ставки ростуть стійко",
    "base": "🟢 Ставки падають / стабільні",
}
BLOCK_UA = {"core": "Ядро", "equity": "Акції", "cta": "CTA", "cash": "Кеш"}


# ───────────────────────────── типи ─────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    base: dict = field(default_factory=lambda: {"core": 60, "equity": 15, "cta": 5, "cash": 20})
    cash_min: float = 10
    confirm_days: int = 5
    ladder: tuple = ((7, 50), (14, 40), (20, 30))
    core_floor: float = 30
    stress_ladder_below_sma200: bool = True
    pause_weeks: int = 4
    bottom_y30_pct: float = 90
    bottom_y10_flat_bp: float = 10
    bottom_min_signs: int = 2
    return_streak: int = 10
    return_step: float = 10
    return_min_days: int = 14
    equity_max: float = 25
    equity_streak: int = 5
    equity_tranche: float = 5
    equity_rebalance_band: float = 5
    cta_min: float = 5
    cta_tranche: float = 5
    min_change: float = 2

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "ladder" in kw:
            kw["ladder"] = tuple(tuple(x) for x in kw["ladder"])
        return cls(**kw)


@dataclass(frozen=True)
class DayInputs:
    date: pd.Timestamp
    is_exec_day: bool
    ret: dict                      # денні дохідності блоків (частки), cash включно
    core_px: float
    stress: bool = False
    downtrend: bool = False        # TLT < SMA200 і SMA50 < SMA200
    core_above: bool = False       # TLT > SMA200
    core_above_streak: int = 0
    y30_pct_5y: float = float("nan")
    y10_chg_bp: float = float("nan")
    core_divergence: str = "none"
    cta_score: int = 0
    cta_score_w: float = 5
    cta_cut: bool = False
    eq_above_streak: int = 0

    @classmethod
    def from_row(cls, date, row: pd.Series) -> "DayInputs":
        def num(k, default=0.0):
            v = row.get(k, default)
            return default if v is None or (isinstance(v, float) and math.isnan(v)) else v
        return cls(
            date=pd.Timestamp(date),
            is_exec_day=bool(row["is_exec_day"]),
            ret={b: float(num(f"ret_{b}")) for b in BLOCKS},
            core_px=float(row["core_px"]),
            stress=bool(num("stress", False)),
            downtrend=bool(num("core_downtrend", False)),
            core_above=bool(num("core_above", False)),
            core_above_streak=int(num("core_above_streak", 0)),
            y30_pct_5y=float(num("y30_pct_5y", float("nan"))),
            y10_chg_bp=float(num("y10_chg_bp", float("nan"))),
            core_divergence=str(num("core_divergence", "none")),
            cta_score=int(num("cta_score", 0)),
            cta_score_w=float(num("cta_score_w", 5)),
            cta_cut=bool(num("cta_cut", False)),
            eq_above_streak=int(num("eq_above_streak", 0)),
        )


@dataclass(frozen=True)
class Action:
    date: pd.Timestamp
    block: str
    from_weight: float
    to_weight: float
    reason: str

    def as_dict(self) -> dict:
        return {"date": str(self.date.date()), "block": self.block,
                "from_weight": round(self.from_weight, 1), "to_weight": round(self.to_weight, 1),
                "reason": self.reason}


@dataclass(frozen=True)
class EngineState:
    date: pd.Timestamp | None
    weights: dict                  # поточні модельні ваги (core, equity, cta, cash)
    targets: dict                  # цільові ваги (core, equity, cta); cash = залишок
    settled: dict                  # чи блок уже на цілі (False → є незавершений рух до цілі)
    core_peak: float
    ladder_step: int = 0           # скільки сходинок продажу виконано в поточному циклі
    pause_until: pd.Timestamp | None = None
    pause_step: int | None = None
    paused_steps: frozenset = frozenset()
    last_core_return: pd.Timestamp | None = None
    regime: str = "base"
    candidate: str | None = None
    candidate_days: int = 0
    equity_extra: float = 0.0      # додаткова частка акцій понад базу
    rebalance_year: int = 0        # рік останнього річного ребалансу бази акцій

    def core_drawdown(self, px: float) -> float:
        return (px / self.core_peak - 1) * 100 if self.core_peak else 0.0

    def target_weights(self) -> dict:
        t = {b: self.targets[b] for b in TRADED}
        t["cash"] = 100 - sum(t.values())
        return t


# ───────────────────────────── стартові стани ─────────────────────────────

def base_state(cfg: StrategyConfig, day: DayInputs) -> EngineState:
    """Базовий стан (§1): ядро 60 · акції 15 · CTA 5 · кеш 20, пік = поточна ціна."""
    w = {b: float(cfg.base[b]) for b in BLOCKS}
    return EngineState(date=None, weights=w, targets={b: w[b] for b in TRADED},
                       settled={b: True for b in TRADED}, core_peak=day.core_px,
                       regime=raw_regime(day), rebalance_year=day.date.year)


def portfolio_state(cfg: StrategyConfig, weights: dict, core_peak: float, as_of: pd.Timestamp,
                    regime: str = "base") -> EngineState:
    """Стартовий стан з portfolio.yaml (§7): усі блоки «не на цілі» → рушій сам веде перехід."""
    w = {b: float(weights.get(b, 0)) for b in TRADED}
    w["cash"] = 100 - sum(w.values())
    return EngineState(date=None, weights=w,
                       targets={"core": float(cfg.base["core"]), "equity": float(cfg.base["equity"]),
                                "cta": float(cfg.cta_min)},
                       settled={b: False for b in TRADED}, core_peak=float(core_peak),
                       regime=regime, rebalance_year=as_of.year)


# ───────────────────────────── правила ─────────────────────────────

def raw_regime(day: DayInputs) -> str:
    """§2: пріоритет Стрес > Ставки ростуть > Базовий (без підтвердження)."""
    if day.stress:
        return "stress"
    if day.downtrend:
        return "rates_up"
    return "base"


def update_regime(state: EngineState, day: DayInputs, cfg: StrategyConfig) -> tuple[str, str | None, int]:
    raw = raw_regime(day)
    if raw == state.regime:
        return state.regime, None, 0
    days = state.candidate_days + 1 if raw == state.candidate else 1
    if days >= cfg.confirm_days:
        return raw, None, 0
    return state.regime, raw, days


def bottom_signs(day: DayInputs, cfg: StrategyConfig) -> list[str]:
    """§3 пауза «схоже на дно»: які з трьох ознак виконуються."""
    out = []
    if not math.isnan(day.y30_pct_5y) and day.y30_pct_5y >= cfg.bottom_y30_pct:
        out.append("30Y ≥ 90-го перцентиля за 5 років")
    if not math.isnan(day.y10_chg_bp) and abs(day.y10_chg_bp) <= cfg.bottom_y10_flat_bp:
        out.append("10Y за 20 днів у межах ±10 б.п.")
    if day.core_divergence == "bullish":
        out.append("бичача RSI-дивергенція TLT")
    return out


def ladder_target_step(dd_pct: float, downtrend: bool, cfg: StrategyConfig) -> int:
    """Найглибша сходинка, умову якої виконано (dd_pct — падіння від піку, додатне число)."""
    k = 0
    for i, (lvl, _) in enumerate(cfg.ladder, start=1):
        last = i == len(cfg.ladder)
        if dd_pct >= lvl - 1e-9 and (downtrend or not last):
            k = i
    return k


def _drift(w: dict, ret: dict) -> dict:
    grown = {b: w[b] * (1 + (ret.get(b) or 0.0)) for b in BLOCKS}
    tot = sum(grown.values())
    return {b: grown[b] / tot * 100 for b in BLOCKS} if tot > 0 else dict(w)


def step(state: EngineState, day: DayInputs, cfg: StrategyConfig) -> tuple[EngineState, list[Action]]:
    """Один торговий день. Порядок (PORTFOLIO_TASK, етап 3):
    0) дрейф ваг; 1) режим; 2) ядро; 3) CTA; 4) акції; 5) кеш ≥ мінімуму; 6) виконання."""
    w = _drift(state.weights, day.ret) if state.date is not None else dict(state.weights)
    regime, candidate, cdays = update_regime(state, day, cfg)
    stress = regime == "stress"
    can_exec = day.is_exec_day
    targets, settled = dict(state.targets), dict(state.settled)
    reasons: dict[str, list[str]] = {b: [] for b in TRADED}
    defensive_sell: set[str] = set()       # продажі, дозволені в Стресі не в день виконання

    s = dict(core_peak=max(state.core_peak, day.core_px), ladder_step=state.ladder_step,
             pause_until=state.pause_until, pause_step=state.pause_step, paused_steps=state.paused_steps,
             last_core_return=state.last_core_return, equity_extra=state.equity_extra,
             rebalance_year=state.rebalance_year)
    base_core = float(cfg.base["core"])

    # ── 2. Ядро: сходинки продажу → пауза → повернення → скидання піку ──
    dd = -(day.core_px / s["core_peak"] - 1) * 100            # падіння від піку, додатне
    ladder_on = regime == "rates_up" or (stress and not (cfg.stress_ladder_below_sma200 and day.core_above))
    if ladder_on and (can_exec or stress):
        k = ladder_target_step(dd, day.downtrend, cfg)
        if k > s["ladder_step"]:
            nxt = s["ladder_step"] + 1
            signs = bottom_signs(day, cfg)
            enough = len(signs) >= cfg.bottom_min_signs
            pause_active = s["pause_until"] is not None and day.date <= s["pause_until"]
            if pause_active and enough and k == s["pause_step"]:
                pass                                            # пауза триває
            elif enough and k == nxt and nxt not in s["paused_steps"]:
                s["pause_until"] = day.date + pd.Timedelta(weeks=cfg.pause_weeks)
                s["pause_step"] = nxt
                s["paused_steps"] = s["paused_steps"] | {nxt}
            else:
                lvl, to = cfg.ladder[k - 1]
                s["ladder_step"], s["pause_until"], s["pause_step"] = k, None, None
                if to < targets["core"]:
                    targets["core"] = float(to)
                    settled["core"] = False
                    defensive_sell.add("core")
                    reasons["core"].append(f"сходинка −{lvl}%: ядро −{dd:.1f}% від піку {s['core_peak']:.2f}")

    if (can_exec and not stress and targets["core"] < base_core
            and day.core_above_streak >= cfg.return_streak
            and (s["last_core_return"] is None
                 or (day.date - s["last_core_return"]).days >= cfg.return_min_days)):
        targets["core"] = min(targets["core"] + cfg.return_step, base_core)
        settled["core"] = False
        s["last_core_return"] = day.date
        reasons["core"].append(f"повернення в ядро: TLT вище SMA200 {day.core_above_streak} днів поспіль")
        if targets["core"] >= base_core:              # ядро відновлене → новий цикл, пік = поточна ціна
            s.update(core_peak=day.core_px, ladder_step=0, pause_until=None, pause_step=None,
                     paused_steps=frozenset())
            reasons["core"].append("ядро відновлене — пік циклу скинуто")
        else:
            s["ladder_step"] = sum(1 for _, lw in cfg.ladder if lw >= targets["core"])

    gap = w["core"] - targets["core"]
    if settled["core"] and (gap >= cfg.equity_rebalance_band
                            or (-gap >= cfg.equity_rebalance_band and regime == "base")):
        settled["core"] = False
        reasons["core"].append(f"ребаланс ядра до цілі {targets['core']:.0f}%")
    elif not settled["core"] and not reasons["core"]:
        reasons["core"].append(f"ядро до цілі {targets['core']:.0f}%")

    # ── 3. CTA: ціль зі скорингу, зрізання; вгору транш, вниз одразу ──
    desired = float(cfg.cta_min) if day.cta_cut else max(float(cfg.cta_min), day.cta_score_w)
    if desired != targets["cta"]:
        if day.cta_cut:
            reasons["cta"].append(f"зрізання CTA до {cfg.cta_min:.0f}%: DBMF і TLT у мінусі за 20 днів, "
                                  "кореляція за 60 днів > 0")
        else:
            reasons["cta"].append(f"скоринг CTA {day.cta_score} → {desired:.0f}%")
        targets["cta"], settled["cta"] = desired, False
    elif settled["cta"] and abs(w["cta"] - targets["cta"]) >= cfg.equity_rebalance_band:
        settled["cta"] = False
        reasons["cta"].append(f"ребаланс CTA до цілі {targets['cta']:.0f}%")
    elif not settled["cta"] and not reasons["cta"]:
        reasons["cta"].append(f"CTA до цілі {targets['cta']:.0f}% (скоринг {day.cta_score})")

    # ── 4. Акції: база 15% завжди; розширення до 25% траншами; вихід додаткових ──
    base_eq = float(cfg.base["equity"])
    ext_fail = []
    if day.eq_above_streak < cfg.equity_streak:
        ext_fail.append("ISAC нижче SMA200" if day.eq_above_streak == 0
                        else f"ISAC вище SMA200 лише {day.eq_above_streak} дн.")
    if stress:
        ext_fail.append("режим Стрес")
    if s["ladder_step"] > 0:
        ext_fail.append("ядро в сходинках продажу")
    if s["equity_extra"] > 0 and ext_fail and (can_exec or stress):
        s["equity_extra"] = 0.0
        targets["equity"], settled["equity"] = base_eq, False
        defensive_sell.add("equity")
        reasons["equity"].append("вихід із додаткових акцій: " + ", ".join(ext_fail))
    if can_exec and day.date.year != s["rebalance_year"]:
        s["rebalance_year"] = day.date.year
        if abs(w["equity"] - targets["equity"]) >= cfg.min_change:
            settled["equity"] = False
            reasons["equity"].append(f"річний ребаланс бази акцій до {targets['equity']:.0f}%")
    if settled["equity"] and abs(w["equity"] - targets["equity"]) > cfg.equity_rebalance_band:
        settled["equity"] = False
        reasons["equity"].append(f"акції відхилились від цілі {targets['equity']:.0f}% більш ніж на "
                                 f"{cfg.equity_rebalance_band:.0f} п.п.")

    # ── 6 (частина). Угоди ядра й CTA, щоб перевірити кеш для розширення акцій ──
    deltas = {b: 0.0 for b in TRADED}
    tranche = {"core": None, "cta": cfg.cta_tranche, "equity": cfg.equity_tranche}

    def plan(b: str) -> None:
        if settled[b]:
            return
        g = targets[b] - w[b]
        if abs(g) < cfg.min_change:                    # зміни < 2 п.п. ігноруються
            settled[b] = True
            return
        if g < 0 and (can_exec or (stress and b in defensive_sell) or (stress and b == "core")):
            deltas[b] = g
        elif g > 0 and can_exec and not (b == "core" and stress):
            d = g if tranche[b] is None else min(g, tranche[b])
            if b == "cta" and w["cta"] < cfg.cta_min:   # мінімальні 5% CTA — захисні, одразу
                d = max(d, min(g, cfg.cta_min - w["cta"]))
            deltas[b] = d

    plan("core")
    plan("cta")

    if (can_exec and not ext_fail and s["equity_extra"] < cfg.equity_max - base_eq
            and w["equity"] >= targets["equity"] - cfg.min_change):
        cash_after = 100 - sum(w[b] + deltas[b] for b in TRADED) - cfg.equity_tranche
        if cash_after >= cfg.cash_min:
            s["equity_extra"] += cfg.equity_tranche
            targets["equity"], settled["equity"] = base_eq + s["equity_extra"], False
            reasons["equity"].append(f"ISAC вище SMA200 {day.eq_above_streak} днів — розширення "
                                     f"акцій до {targets['equity']:.0f}%")
    if not settled["equity"] and not reasons["equity"]:
        reasons["equity"].append(f"база акцій {targets['equity']:.0f}%")
    plan("equity")

    # ── 5. Мінімум кешу 10% має пріоритет ──
    def cash_left() -> float:
        return 100 - sum(w[b] + deltas[b] for b in TRADED)

    short = cfg.cash_min - cash_left()
    if short > 1e-9:
        for b in ("equity", "cta", "core"):            # спершу урізаємо покупки
            if deltas[b] > 0 and short > 1e-9:
                cut = min(deltas[b], short)
                deltas[b] -= cut
                short -= cut
                reasons[b].append(f"покупку урізано: мінімум кешу {cfg.cash_min:.0f}%")
                if b == "equity":
                    s["equity_extra"] = max(0.0, s["equity_extra"] - cut)
                    targets["equity"] = base_eq + s["equity_extra"]
        if short > 1e-9 and (can_exec or stress):      # кеш просів дрейфом: продаємо додаткові акції, потім CTA
            for b, floor in (("equity", base_eq), ("cta", float(cfg.cta_min))):
                room = max(0.0, w[b] + deltas[b] - floor)
                cut = min(room, short)
                if cut > 1e-9:
                    deltas[b] -= cut
                    short -= cut
                    reasons[b].append(f"продаж: мінімум кешу {cfg.cash_min:.0f}%")
                    if b == "equity":
                        s["equity_extra"] = 0.0
                        targets["equity"] = base_eq

    # ── 6. Виконання ──
    actions = []
    new_w = dict(w)
    for b in TRADED:
        d = deltas[b]
        if abs(d) < cfg.min_change:
            continue
        new_w[b] = w[b] + d
        actions.append(Action(day.date, b, w[b], new_w[b], "; ".join(reasons[b]) or "ребаланс"))
    new_w["cash"] = 100 - sum(new_w[b] for b in TRADED)
    for b in TRADED:
        if abs(new_w[b] - targets[b]) < cfg.min_change:
            settled[b] = True

    new = dataclasses.replace(
        state, date=day.date, weights=new_w, targets=targets, settled=settled,
        regime=regime, candidate=candidate, candidate_days=cdays, **s)
    return new, actions


# ───────────────────────────── прогін ─────────────────────────────

def warm_regime(state: EngineState, days: list[DayInputs], cfg: StrategyConfig) -> EngineState:
    """Розігрів режиму на днях до старту (лише крок 1), щоб на старті був підтверджений режим."""
    if days:
        state = dataclasses.replace(state, regime=raw_regime(days[0]), candidate=None, candidate_days=0)
    for d in days[1:]:
        r, c, n = update_regime(state, d, cfg)
        state = dataclasses.replace(state, regime=r, candidate=c, candidate_days=n)
    return state


def record(state: EngineState, day: DayInputs) -> dict:
    return {"date": day.date, "regime": state.regime, "candidate": state.candidate,
            "candidate_days": state.candidate_days,
            **{f"w_{b}": state.weights[b] for b in BLOCKS},
            **{f"t_{b}": v for b, v in state.target_weights().items()},
            "core_peak": state.core_peak, "core_dd": state.core_drawdown(day.core_px),
            "ladder_step": state.ladder_step,
            "paused": state.pause_until is not None and day.date <= state.pause_until}


def run(df: pd.DataFrame, state: EngineState, cfg: StrategyConfig, start=None, warmup_days: int = 60
        ) -> tuple[pd.DataFrame, list[Action], EngineState, EngineState | None]:
    """Проганяє `step` по рядках df від `start`. Повертає (журнал по днях, дії, фінальний стан,
    стан перед останнім днем — для `preview`)."""
    start = pd.Timestamp(start) if start is not None else df.index[0]
    pre = [DayInputs.from_row(d, r) for d, r in df[df.index < start].tail(warmup_days).iterrows()]
    state = warm_regime(state, pre, cfg)
    rows, actions, prev = [], [], None
    for d, r in df[df.index >= start].iterrows():
        day = DayInputs.from_row(d, r)
        prev = state
        state, acts = step(state, day, cfg)
        actions += acts
        rows.append(record(state, day))
    return pd.DataFrame(rows).set_index("date"), actions, state, prev


def preview(prev: EngineState, day: DayInputs, cfg: StrategyConfig) -> tuple[EngineState, list[Action]]:
    """Що рушій зробить у найближчий день виконання з поточними даними («угоди цього тижня»)."""
    return step(prev, dataclasses.replace(day, is_exec_day=True), cfg)
