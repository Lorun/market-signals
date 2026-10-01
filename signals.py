#!/usr/bin/env python3
"""
Портфельний монітор (docs/strategy.md): ядро TLT · акції ISAC · CTA DBMF · кеш.

Треки (векторно, по днях): core, rates, stress, cta, equity; macro — контекст.
Рушій правил (engine.py) проганяється від стартового стану з portfolio.yaml до сьогодні
і показує режим, цільову модельну алокацію та угоди цього тижня.

Виходи:
  data/signals.json — режим, ваги, цілі, угоди тижня, стани/метрики треків, алерти
  data/history.csv  — щоденний знімок ключових чисел
Сповіщення: Telegram при зміні режиму, сходинки, цільової ваги або появі угод на тиждень.

Запуск:  python signals.py           (потрібен FRED_API_KEY)
         python signals.py --demo    (синтетичні дані, без мережі)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from engine import (BLOCK_UA, REGIME_UA, DayInputs, StrategyConfig, bottom_signs, portfolio_state,
                    preview, run)
from indicators import r
from inputs import build_inputs
from sources import build_block_prices, demo_data, load_market, unit_jumps
from tracks import CTA_COMPONENTS, macro_track

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SIGNALS = DATA / "signals.json"
HISTORY = DATA / "history.csv"

CTA_UA = {
    "tlt_vs_sma200": "TLT vs SMA200",
    "sma50_vs_sma200": "SMA50 vs SMA200",
    "y10_chg": "10Y за 20д",
    "rates_vol": "вол. ставок",
    "dbmf_vs_sma100": "DBMF vs SMA100",
}


# ───────────────────────────── перевірки даних ─────────────────────────────

def data_warnings(prices: pd.DataFrame, fred: dict, df: pd.DataFrame, cfg: dict, today: pd.Timestamp) -> list[str]:
    p, out = cfg["params"], []
    checks = [(cfg["tickers"][k], prices[k], p["stale_days"]) for k in cfg["tickers"]]
    checks += [(cfg["fred"][k], fred[k], p["stale_days"]) for k in ("y2", "y10", "y30", "tbill", "oas_ig", "vix")]
    checks += [(cfg["fred"]["nfci"], fred["nfci"], p["stale_days_weekly"])]
    checks += [(cfg["fred"][k], fred[k], p["stale_days_monthly"]) for k in ("cpi", "payrolls", "unrate")]
    for name, s, max_age in checks:
        s = s.dropna()
        if s.empty:
            out.append(f"{name}: немає даних")
            continue
        if (today - s.index[-1]).days > max_age:
            out.append(f"{name}: дані застарілі ({s.index[-1].date()})")
    for k in ("equity", "cash", "cta_exec"):
        jumps = unit_jumps(prices[k].dropna().iloc[-500:])
        if jumps:
            out.append(f"{cfg['tickers'][k]}: стрибок ціни >50% за день ({', '.join(jumps[-3:])}) — "
                       "можлива зміна одиниць котирування")
    gap = df["cta_exec_gap_pct"].dropna()
    if len(gap) and abs(gap.iloc[-1]) > p["cta_exec_gap_pct"]:
        out.append(f"DBMF.PA і DBMF розійшлись за 20 днів на {gap.iloc[-1]:+.1f}% (поріг {p['cta_exec_gap_pct']}%)")
    return out


# ───────────────────────────── стани треків ─────────────────────────────

def track_snapshot(row: pd.Series, fred: dict, cfg: dict) -> dict:
    yn = lambda b: "yes" if bool(b) else "no"
    hy, nfci = fred["hy_oas"].dropna(), fred["nfci"].dropna()
    return {
        "core": {
            "states": {"vs_sma200": "above" if row["core_above"] else "below",
                       "sma_cross": "golden" if row["core_sma50"] > row["core_sma200"] else "death",
                       "downtrend": yn(row["core_downtrend"]),
                       "rsi_divergence": row["core_divergence"]},
            "metrics": {"tlt": r(row["core_px"]), "sma50": r(row["core_sma50"]), "sma200": r(row["core_sma200"]),
                        "dist_sma200_pct": r(row["core_dist_pct"]), "sma50_vs_sma200_pct": r(row["core_sma_gap_pct"]),
                        "above_sma200_days": int(row["core_above_streak"]), "rsi": r(row["core_rsi"], 1)},
        },
        "rates": {
            "metrics": {"y2": r(row["y2"]), "y10": r(row["y10"]), "y30": r(row["y30"]),
                        "y10_chg_20d_bp": r(row["y10_chg_bp"], 1), "rv10y_bp": r(row["rv10y_bp"]),
                        "rv10y_pct_1y": r(row["rv_pct_1y"], 0), "y30_pct_5y": r(row["y30_pct_5y"], 0),
                        "s2s10_bp": r(row["s2s10_bp"], 1), "s10s30_bp": r(row["s10s30_bp"], 1)},
        },
        "stress": {
            "states": {k: yn(row[k]) for k in ("stress", "stress_rates", "stress_credit", "stress_vix")},
            "metrics": {"oas_ig_bp": r(row["credit_bp"], 1), "oas_ig_z_1y": r(row["credit_z"]),
                        "vix": r(row["vix"], 1), "vix_pct_1y": r(row["vix_pct_1y"], 0),
                        "hy_oas_bp": r(hy.iloc[-1] * 100, 0) if len(hy) else None,
                        "nfci": r(nfci.iloc[-1]) if len(nfci) else None},
        },
        "cta": {
            "metrics": {"dbmf": r(row["cta_px"]), "sma100": r(row["cta_sma100"]),
                        "dist_sma100_pct": r(row["cta_dist_pct"]), "dbmf_ret20_pct": r(row["cta_ret20_pct"]),
                        "tlt_ret20_pct": r(row["core_ret20_pct"]), "corr60": r(row["cta_corr60"]),
                        "cut": yn(row["cta_cut"]), "dbmf_pa_gap_20d_pct": r(row.get("cta_exec_gap_pct"))},
        },
        "equity": {
            "states": {"vs_sma200": "above" if row["eq_above"] else "below"},
            "metrics": {"ssac": r(row["eq_px"]), "sma200": r(row["eq_sma200"]),
                        "dist_sma200_pct": r(row["eq_dist_pct"]), "above_sma200_days": int(row["eq_above_streak"])},
        },
        "macro": macro_track(fred["cpi"], fred["payrolls"], fred["unrate"]),
    }


# ───────────────────────────── алерти / Telegram ─────────────────────────────

def diff_signals(prev: dict | None, cur: dict) -> list[dict]:
    if not prev:
        return []
    out = []
    if prev.get("regime", {}).get("current") != cur["regime"]["current"]:
        out.append({"what": "режим", "from": prev.get("regime", {}).get("current"), "to": cur["regime"]["current"]})
    if prev.get("core", {}).get("ladder_step") != cur["core"]["ladder_step"]:
        out.append({"what": "сходинка", "from": prev.get("core", {}).get("ladder_step"),
                    "to": cur["core"]["ladder_step"]})
    for b, v in cur["target"].items():
        old = prev.get("target", {}).get(b)
        if old is None or abs(old - v) >= 0.5:
            out.append({"what": f"ціль {BLOCK_UA[b]}", "from": old, "to": v})
    if cur["actions_this_week"] and cur["actions_this_week"] != prev.get("actions_this_week"):
        out.append({"what": "угоди тижня", "from": len(prev.get("actions_this_week") or []),
                    "to": len(cur["actions_this_week"])})
    return out


def telegram(text: str) -> None:
    import requests
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[telegram] токен/chat_id не задано — пропускаю")
        return
    resp = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                         json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                               "disable_web_page_preview": True}, timeout=20)
    if not resp.ok:
        print(f"[telegram] помилка {resp.status_code}: {resp.text}", file=sys.stderr)


def fmt_w(w: dict) -> str:
    return " · ".join(f"{BLOCK_UA[b]} {w[b]:.0f}%" for b in ("core", "equity", "cta", "cash"))


def format_message(sig: dict) -> str:
    R, C, T = sig["regime"], sig["core"], sig["tracks"]
    lines = [f"<b>Портфель · {sig['as_of']}</b>", f"Режим: <b>{REGIME_UA[R['current']]}</b>"]
    if R["candidate"]:
        lines.append(f"  кандидат: {REGIME_UA[R['candidate']]} ({R['candidate_days']}/{R['confirm_days']} днів)")
    lines += ["", f"Зараз:  {fmt_w(sig['portfolio'])}", f"Ціль:   {fmt_w(sig['target'])}"]
    if sig["actions_this_week"]:
        if sig["signal_final"]:
            lines.append(f"\n<b>Угоди</b> (сигнал за закриттям {sig['signal_day']} → виконати {sig['execute_on']}):")
        else:
            lines.append(f"\n<b>Угоди</b> (попередньо за даними {sig['as_of']}; остаточний сигнал — закриття "
                         f"{sig['signal_day']} → виконати {sig['execute_on']}):")
        for a in sig["actions_this_week"]:
            tag = (f"виконати {a['when']}" + (", Стрес — не чекаючи кінця тижня" if a["urgent"] else "")
                   + (", попередньо" if a["preliminary"] else ""))
            lines.append(f"• {BLOCK_UA[a['block']]}: {a['from_weight']:.0f}% → <b>{a['to_weight']:.0f}%</b> "
                         f"[{tag}] — {a['reason']}")
    else:
        lines.append("\nУгод цього тижня немає.")
    pause = f", пауза до {C['pause_until']}" if C["paused"] else ""
    lines += [
        "",
        f"Ядро: пік {C['peak']}, просадка {C['drawdown_pct']:+.1f}%, сходинка {C['ladder_step']}/3"
        f"{pause}; до наступної: {C['next_step_pct']}",
        f"CTA-скоринг {sig['cta_score']['total']}/10 → {sig['cta_score']['weight']:.0f}%  ("
        + ", ".join(f"{CTA_UA[k]} {v}" for k, v in sig["cta_score"]["components"].items()) + ")",
        f"TLT {T['core']['metrics']['dist_sma200_pct']:+.1f}% від SMA200 · "
        f"10Y {T['rates']['metrics']['y10']}% ({T['rates']['metrics']['y10_chg_20d_bp']:+} б.п. за 20д) · "
        f"IG OAS z {T['stress']['metrics']['oas_ig_z_1y']} · VIX {T['stress']['metrics']['vix']}",
        f"ISAC {T['equity']['metrics']['dist_sma200_pct']:+.1f}% від SMA200",
    ]
    if sig["warnings"]:
        lines.append("\n⚠️ " + "; ".join(sig["warnings"]))
    return "\n".join(lines)


# ───────────────────────────── main ─────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="синтетичні дані, без мережі")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pf = yaml.safe_load((ROOT / "portfolio.yaml").read_text(encoding="utf-8"))
    scfg = StrategyConfig.from_dict(cfg["strategy"])
    today = pd.Timestamp.today().normalize()

    if args.demo:
        prices, fred = demo_data(cfg)
    else:
        key = os.getenv("FRED_API_KEY")
        if not key:
            print("FRED_API_KEY не задано", file=sys.stderr)
            return 1
        as_of = pd.Timestamp(pf["as_of"])
        start = (min(today, as_of) - pd.Timedelta(days=cfg["history_days"])).date().isoformat()
        prices, fred = load_market(cfg, start, key)

    blocks = build_block_prices(prices, fred, cfg, backtest=False)
    df = build_inputs(blocks, fred, cfg, credit_key="oas_ig", live=True)
    df = df[df.index <= today]

    # демо: старт за ~3 місяці до кінця, щоб рушій встиг зробити кілька кроків
    as_of = df.index[-60] if args.demo else pd.Timestamp(pf["as_of"])
    start = df.index[df.index >= as_of][0] if (df.index >= as_of).any() else df.index[-1]
    peak = pf.get("core_peak", "auto")
    if peak in (None, "auto"):
        peak = float(df.loc[:start, "core_px"].tail(252).max())
    state0 = portfolio_state(scfg, pf["weights"], float(peak), start)
    log, actions, state, prev = run(df, state0, scfg, start=start)

    last_date = df.index[-1]
    day = DayInputs.from_row(last_date, df.iloc[-1])
    # Сигнал рахується після закриття LSE/Euronext → угода виконується наступного торгового дня
    # (сигнал у п'ятницю → угода в понеділок; захисна дія в Стресі — наступного дня після сигналу).
    nxt_bday = lambda d: pd.Timestamp(d) + pd.offsets.BDay(1)
    week_start = last_date - pd.Timedelta(days=last_date.weekday())
    exec_flag = df["is_exec_day"]
    done = [a for a in actions if nxt_bday(a.date) >= week_start]     # виконання припадає на цей тиждень
    week_actions = [{**a.as_dict(), "signal_date": str(a.date.date()), "when": str(nxt_bday(a.date).date()),
                     "urgent": not bool(exec_flag.get(a.date, False)), "preliminary": False} for a in done]
    if day.is_exec_day:
        target_state, signal_day = state, last_date
    else:
        target_state, planned = preview(prev, day, scfg)
        signal_day = last_date + pd.offsets.Week(weekday=4)
        # прогноз: що рушій зробить за закриттям дня сигналу, якщо дані не зміняться (дата — дата даних)
        week_actions += [{**a.as_dict(), "signal_date": str(last_date.date()),
                          "when": str(nxt_bday(signal_day).date()), "urgent": False, "preliminary": True}
                         for a in planned
                         if not any(x.block == a.block for x in done if x.date == last_date)]

    # «Зараз» — ваги до угод цього тижня (рушій міг уже виконати захисні дії в Стресі)
    current = {b: state.weights[b] - sum(a.to_weight - a.from_weight for a in done if a.block == b)
               for b in ("core", "equity", "cta")}
    current["cash"] = 100 - sum(current.values())

    row = df.iloc[-1]
    dd = state.core_drawdown(day.core_px)
    nxt = scfg.ladder[state.ladder_step] if state.ladder_step < len(scfg.ladder) else None
    warnings = data_warnings(prices, fred, df, cfg, today) if not args.demo else []
    if not args.demo and start > pd.Timestamp(pf["as_of"]):
        warnings.append(f"as_of {pf['as_of']} поза торговим календарем — старт з {start.date()}")

    sig = {
        "as_of": str(last_date.date()),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "portfolio_as_of": str(start.date()),
        "regime": {"current": state.regime, "label": REGIME_UA[state.regime],
                   "candidate": state.candidate, "candidate_days": state.candidate_days,
                   "confirm_days": scfg.confirm_days},
        "portfolio": {b: round(v, 1) for b, v in current.items()},
        "model_weights": {b: round(v, 1) for b, v in state.weights.items()},
        "target": {b: round(v, 1) for b, v in target_state.target_weights().items()},
        "actions_this_week": week_actions,
        "signal_day": str(signal_day.date()),           # день тижневого сигналу (може бути в майбутньому)
        "signal_final": bool(day.is_exec_day),          # False → угоди тижня попередні, за даними as_of
        "execute_on": str(nxt_bday(signal_day).date()),
        "core": {"peak": r(state.core_peak), "drawdown_pct": r(dd), "ladder_step": state.ladder_step,
                 "next_step_pct": f"−{nxt[0]}% → ядро {nxt[1]}%" if nxt else "—",
                 "paused": state.pause_until is not None and last_date <= state.pause_until,
                 "pause_until": str(state.pause_until.date()) if state.pause_until is not None else None,
                 "paused_steps": sorted(state.paused_steps),
                 "bottom_signs": bottom_signs(day, scfg)},
        "cta_score": {"total": int(row["cta_score"]), "weight": float(row["cta_score_w"]),
                      "cut": bool(row["cta_cut"]),
                      "components": {c: int(row[f"cta_c_{c}"]) for c in CTA_COMPONENTS}},
        "tracks": track_snapshot(row, fred, cfg),
        "actions_since_start": [a.as_dict() for a in actions],
        "warnings": warnings,
    }

    prev_sig = json.loads(SIGNALS.read_text(encoding="utf-8")) if SIGNALS.exists() else None
    alerts = diff_signals(prev_sig, sig)
    sig["alerts"] = alerts
    alert_log = (prev_sig or {}).get("alert_log", []) + [{"date": sig["as_of"], **a} for a in alerts]
    sig["alert_log"] = alert_log[-60:]

    DATA.mkdir(exist_ok=True)
    if not args.demo:
        SIGNALS.write_text(json.dumps(sig, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        hrow = {"date": sig["as_of"], "regime": state.regime,
                **{f"w_{b}": sig["model_weights"][b] for b in ("core", "equity", "cta", "cash")},
                "core_dd_pct": r(dd), "cta_score": sig["cta_score"]["total"],
                "y10": r(row["y10"]), "oas_ig_bp": r(row["credit_bp"], 1), "vix": r(row["vix"], 1)}
        hist = pd.DataFrame([hrow])
        if HISTORY.exists():
            old = pd.read_csv(HISTORY)
            hist = pd.concat([old[old["date"] != hrow["date"]], hist], ignore_index=True)
        hist.sort_values("date").to_csv(HISTORY, index=False)

    msg = format_message(sig)
    print(msg)
    if not args.no_notify and (alerts or cfg.get("telegram", {}).get("daily_summary") or prev_sig is None):
        telegram(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
