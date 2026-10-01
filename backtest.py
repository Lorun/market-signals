#!/usr/bin/env python3
"""
Бектест стратегії на тому самому рушії (`engine.step`), що й живий звіт.

Прогони: з 2008 (CTA: RYMFX→AQMIX→DBMF) і з 2010 (CTA: AQMIX→DBMF); акції ACWI→ISAC.L,
кеш DTB3→IB01.L, умова Стресу по кредиту — BAA10Y (IG OAS на FRED лише ~3 роки).
Старт із базового стану. Сигнал на close дня t, угода на close t, дохідність з t+1.

Бенчмарки: 60/15/25 (TLT / акції / кеш) з річним ребалансом; 100% TLT.

Вихід: data/backtest/equity_curve.csv, actions.csv, summary.md

Запуск:  FRED_API_KEY=... python backtest.py [--fee 0.001]
         python backtest.py --demo   (синтетичні дані, без мережі)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine import BLOCKS, REGIME_UA, DayInputs, StrategyConfig, base_state, run
from inputs import build_inputs, first_valid_date
from sources import build_block_prices, demo_data, load_market

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "backtest"
PERIODS = {"2013 (taper tantrum)": "2013", "2020": "2020", "2022": "2022", "2023": "2023"}
DD_LIMIT = -18.0


# ───────────────────────────── прогін ─────────────────────────────

def simulate(df: pd.DataFrame, scfg: StrategyConfig, start, fee: float):
    """Повертає (журнал по днях з вартістю портфеля, дії)."""
    start = max(pd.Timestamp(start), first_valid_date(df))
    row0 = df.loc[start:].iloc[0]
    state = base_state(scfg, DayInputs.from_row(row0.name, row0))
    log, actions, _, _ = run(df, state, scfg, start=row0.name)
    rets = df.loc[log.index, [f"ret_{b}" for b in BLOCKS]].fillna(0.0).to_numpy()
    w_prev = log[[f"w_{b}" for b in BLOCKS]].shift(1).to_numpy() / 100
    gross = np.nansum(w_prev * rets, axis=1)
    gross[0] = 0.0
    turnover = pd.Series(0.0, index=log.index)
    for a in actions:
        turnover[a.date] += abs(a.to_weight - a.from_weight) / 100
    # кеш — протилежна сторона кожної угоди, тож оборот рахуємо лише по торгованих блоках
    log["strategy"] = 100 * np.cumprod(1 + gross - fee * turnover.to_numpy())
    log["turnover"] = turnover
    return log, actions


def static_benchmark(df: pd.DataFrame, weights: dict, index: pd.DatetimeIndex, fee: float) -> pd.Series:
    """Статичний портфель з ребалансом у перший торговий день року."""
    w = {b: weights.get(b, 0) / 100 for b in BLOCKS}
    cur, vals, v = dict(w), [], 100.0
    rets = df.loc[index, [f"ret_{b}" for b in BLOCKS]].fillna(0.0)
    prev_year = index[0].year
    for i, (d, r) in enumerate(rets.iterrows()):
        if i:
            g = {b: cur[b] * (1 + r[f"ret_{b}"]) for b in BLOCKS}
            tot = sum(g.values())
            v *= tot
            cur = {b: g[b] / tot for b in BLOCKS}
            if d.year != prev_year:
                v *= 1 - fee * sum(abs(cur[b] - w[b]) for b in BLOCKS if b != "cash")
                cur, prev_year = dict(w), d.year
        vals.append(v)
    return pd.Series(vals, index=index)


# ───────────────────────────── метрики ─────────────────────────────

def max_drawdown(v: pd.Series) -> dict:
    peak = v.cummax()
    dd = v / peak - 1
    trough = dd.idxmin()
    pk = v.loc[:trough].idxmax()
    rec = v.loc[trough:][v.loc[trough:] >= v.loc[pk]]
    return {"max_dd": dd.min() * 100, "peak": pk.date(), "trough": trough.date(),
            "recovery": rec.index[0].date() if len(rec) else None}


def metrics(v: pd.Series) -> dict:
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    r = v.pct_change().dropna()
    # дохідність календарного року: від останнього значення попереднього року
    ye = v.groupby(v.index.year).last()
    yr_ret = ye / ye.shift(1).fillna(v.iloc[0]) - 1
    m = {"cagr": ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1) * 100,
         "vol": r.std() * np.sqrt(252) * 100,
         "worst_year": f"{yr_ret.idxmin()} ({yr_ret.min() * 100:+.1f}%)"}
    m.update(max_drawdown(v))
    return m


def period_stats(v: pd.Series, year: str) -> tuple[float, float]:
    """(дохідність за рік %, максимальна просадка всередині року %)."""
    prev = v[v.index < pd.Timestamp(f"{year}-01-01")]
    s = v[v.index.year == int(year)]
    if s.empty:
        return float("nan"), float("nan")
    base = prev.iloc[-1] if len(prev) else s.iloc[0]
    s2 = pd.concat([pd.Series([base], index=[s.index[0] - pd.Timedelta(days=1)]), s])
    return (s.iloc[-1] / base - 1) * 100, (s2 / s2.cummax() - 1).min() * 100


# ───────────────────────────── звіт ─────────────────────────────

def fmt_row(name: str, m: dict, extra: str = "") -> str:
    return (f"| {name} | {m['cagr']:.2f}% | {m['vol']:.2f}% | **{m['max_dd']:.1f}%** "
            f"({m['peak']} → {m['trough']}, відновлення {m['recovery'] or 'ще ні'}) | {m['worst_year']} |{extra}")


def section(label: str, log: pd.DataFrame, actions: list, bench: dict[str, pd.Series], fee: float) -> list[str]:
    v = log["strategy"]
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    ms = metrics(v)
    ok = "✅ так" if ms["max_dd"] >= DD_LIMIT else "❌ ні"
    out = [f"## Прогін {label}: {v.index[0].date()} → {v.index[-1].date()}", "",
           f"**Чи вкладається максимальна просадка в {DD_LIMIT:.0f}%: {ok}** — {ms['max_dd']:.1f}%, "
           f"пік {ms['peak']} → дно {ms['trough']}, відновлення {ms['recovery'] or 'ще ні'}.", "",
           "| Портфель | CAGR | Волатильність | Макс. просадка | Найгірший рік | Угод/рік |",
           "|---|---|---|---|---|---|",
           fmt_row("Стратегія", ms, f" {len(actions) / yrs:.1f} |")]
    for name, b in bench.items():
        out.append(fmt_row(name, metrics(b), " 1.0 |" if "60/15/25" in name else " 0 |"))
    out += ["", f"Комісія/спред: {fee * 100:.2f}% від обороту.", "",
            "**Частка часу в режимах:** " + " · ".join(
                f"{REGIME_UA[k]} {log['regime'].eq(k).mean() * 100:.0f}%" for k in ("base", "rates_up", "stress")),
            "",
            "**Середні частки:** " + " · ".join(
                f"{n} {log[f'w_{b}'].mean():.1f}%" for b, n in
                (("core", "ядро"), ("cta", "CTA"), ("equity", "акції"), ("cash", "кеш"))), "",
            "**Окремі періоди** (дохідність за рік / макс. просадка всередині року):", "",
            "| Період | Стратегія | " + " | ".join(bench) + " |",
            "|---|---|" + "---|" * len(bench)]
    for name, y in PERIODS.items():
        cells = [period_stats(v, y)] + [period_stats(b, y) for b in bench.values()]
        out.append(f"| {name} | " + " | ".join("—" if np.isnan(r) else f"{r:+.1f}% / {d:.1f}%" for r, d in cells) + " |")
    return out + [""]


# ───────────────────────────── main ─────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=None, help="комісія/спред від обороту (0.001 = 0,1%)")
    ap.add_argument("--demo", action="store_true", help="синтетичні дані, без мережі")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    fee = args.fee if args.fee is not None else cfg["backtest"]["fee"]
    scfg = StrategyConfig.from_dict(cfg["strategy"])

    if args.demo:
        prices, fred = demo_data(cfg)
        starts = {"demo": prices.index[0]}
        variants = {"demo": cfg}
    else:
        key = os.getenv("FRED_API_KEY")
        if not key:
            print("FRED_API_KEY не задано", file=sys.stderr)
            return 1
        print("Завантаження даних…")
        prices, fred = load_market(cfg, "2002-01-01", key, backtest=True)
        cfg_2010 = copy.deepcopy(cfg)
        cfg_2010["proxies"]["cta"] = [s for s in cfg["proxies"]["cta"] if s.get("ticker") != "RYMFX"]
        starts = dict(zip(("2008", "2010"), cfg["backtest"]["starts"]))
        variants = {"2008": cfg, "2010": cfg_2010}

    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Бектест стратегії", "",
          "Рушій — той самий `engine.step()`, що й у щоденному звіті; старт із базового стану "
          "(ядро 60 · акції 15 · CTA 5 · кеш 20). Проксі: CTA RYMFX (2007–2010) → AQMIX (2010–2019) → DBMF; "
          "акції ACWI → ISAC.L (з 2011-10); кеш DTB3 → IB01.L (з 2019-02); умова Стресу по кредиту — "
          "z-score BAA10Y замість IG OAS (ICE OAS на FRED доступний лише з 2023). "
          "Проксі CTA — інші менеджери: результат показує характер захисту, а не точні цифри.", ""]
    curves, all_actions, main_df, main_start = [], [], None, None
    for label, c in variants.items():
        blocks = build_block_prices(prices, fred, c, backtest=not args.demo)
        df = build_inputs(blocks, fred, c, credit_key="baa10y", live=False)
        print(f"Прогін {label}…")
        log, actions = simulate(df, scfg, starts[label], fee)
        tlt_r = df.loc[log.index, "ret_core"].fillna(0)
        tlt_r.iloc[0] = 0.0
        bench = {"60/15/25 (річний ребаланс)": static_benchmark(df, cfg["backtest"]["benchmark"], log.index, fee),
                 "100% TLT": 100 * (1 + tlt_r).cumprod()}
        md += section(label, log, actions, bench, fee)
        curve = log[["regime", "strategy"] + [f"w_{b}" for b in BLOCKS] + ["core_dd", "ladder_step"]].copy()
        curve["bench_60_15_25"] = bench["60/15/25 (річний ребаланс)"]
        curve["tlt"] = bench["100% TLT"]
        curve.insert(0, "run", label)
        curves.append(curve.round(4))
        all_actions += [{"run": label, **a.as_dict()} for a in actions]
        if main_df is None:
            main_df, main_start = df, starts[label]

    # ── чутливість (на першому прогоні) ──
    md += ["## Чутливість (прогін " + next(iter(variants)) + ")", "",
           "| Варіант | CAGR | Макс. просадка | Угод/рік |", "|---|---|---|---|"]
    lad = scfg.ladder
    variants_s = [("базові пороги, підтвердження 5 дн.", scfg)]
    for sh in (-2, 2):
        variants_s.append((f"сходинки {'+' if sh > 0 else '−'}2 п.п. "
                           f"({', '.join(f'−{l + sh}%' for l, _ in lad)})",
                           dataclasses.replace(scfg, ladder=tuple((l + sh, w) for l, w in lad))))
    for n in (3, 10):
        variants_s.append((f"підтвердження режиму {n} дн.", dataclasses.replace(scfg, confirm_days=n)))
    variants_s.append(("сходинки в Стресі навіть при TLT вище SMA200 (старе правило)",
                       dataclasses.replace(scfg, stress_ladder_below_sma200=not scfg.stress_ladder_below_sma200)))
    for name, sc in variants_s:
        log, actions = simulate(main_df, sc, main_start, fee)
        m = metrics(log["strategy"])
        yrs = (log.index[-1] - log.index[0]).days / 365.25
        md.append(f"| {name} | {m['cagr']:.2f}% | {m['max_dd']:.1f}% | {len(actions) / yrs:.1f} |")

    pd.concat(curves).to_csv(OUT / "equity_curve.csv", index_label="date")
    pd.DataFrame(all_actions).to_csv(OUT / "actions.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n→ {OUT}/equity_curve.csv, actions.csv, summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
