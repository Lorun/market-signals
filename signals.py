#!/usr/bin/env python3
"""
TLT / SUOA signal monitor.

Три незалежні треки:
  * RATES       — рух ставок і технічка ratio TLT/SUOA (дюраційний трек)
  * CREDIT      — кредитний спред (OAS) напряму з FRED, не залежить від дюрації
  * MACRO       — контекст (інфляція, ринок праці), без голосів у скорингу
  * HIGH_YIELD  — XUHA ETF: спреди, VIX, NFCI, технічка (+ = сприятливо для XUHA)

Знак скорингу RATES/CREDIT: "+" = на користь TLT, "−" = на користь SUOA.
Знак скорингу HIGH_YIELD:   "+" = сприятливо для XUHA (risk-on), "−" = несприятливо.

Виходи:
  data/signals.json — поточний стан усіх індикаторів + алерти (зміни стану)
  data/history.csv  — щоденний знімок ключових чисел
Сповіщення: Telegram, коли змінюється стан хоча б одного індикатора.

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

from sources import fetch_fred, fetch_prices, demo_data
from tracks import rates_track, credit_track, macro_track, label, high_yield_track, hy_label

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SIGNALS = DATA / "signals.json"
HISTORY = DATA / "history.csv"


# ───────────────────────────── alerts ─────────────────────────────

def diff_states(prev: dict | None, cur: dict) -> list[dict]:
    if not prev:
        return []
    alerts = []
    label_fn = {"rates": label, "credit": label, "high_yield": hy_label}
    for track in ("rates", "credit", "macro", "high_yield"):
        old = prev.get(track, {}).get("states", {})
        for k, v in cur[track]["states"].items():
            if k in old and old[k] != v:
                alerts.append({"track": track, "indicator": k, "from": old[k], "to": v})
        if track != "macro":
            fn = label_fn.get(track, label)
            o, n = prev.get(track, {}).get("score"), cur[track]["score"]
            if o is not None and fn(o, cur[track]["max_score"]) != fn(n, cur[track]["max_score"]):
                alerts.append({"track": track, "indicator": "score_regime",
                               "from": f"{o} ({fn(o, cur[track]['max_score'])})",
                               "to": f"{n} ({fn(n, cur[track]['max_score'])})"})
    return alerts


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


TRACK_UA = {
    "rates": "Ставки",
    "credit": "Кредит",
    "macro": "Макро",
    "high_yield": "Висока дохідність",
}


def format_message(sig: dict, alerts: list[dict]) -> str:
    R, C, HY = sig["rates"], sig["credit"], sig["high_yield"]
    lines = [f"<b>TLT/SUOA · {sig['as_of']}</b>"]
    if alerts:
        lines.append("\n<b>Зміни стану:</b>")
        for a in alerts:
            lines.append(f"• {TRACK_UA[a['track']]} / {a['indicator']}: {a['from']} → <b>{a['to']}</b>")
    lines += [
        "",
        f"<b>Ставки</b>: {R['score']:+d}/{R['max_score']} — {R['label']}",
        f"  ratio {R['metrics']['ratio']} ({R['metrics']['ratio_chg_20d_pct']:+}% 20д), "
        f"RSI {R['metrics']['rsi']}, {R['states']['sma_cross']}",
        f"  10Y {R['metrics']['y10']}% ({R['metrics']['y10_chg_bp']:+} б.п.), "
        f"10s30s {R['metrics']['s10s30_bp']} б.п., вол. {R['states']['rates_vol']}",
        f"<b>Кредит</b>: {C['score']:+d}/{C['max_score']} — {C['label']}",
        f"  IG OAS {C['metrics']['oas_ig_bp']} б.п. ({C['metrics']['oas_ig_chg_bp']:+} за 20д), "
        f"z {C['metrics']['oas_ig_z_1y']}, BBB−IG {C['metrics']['bbb_ig_gap_bp']}",
        f"<b>Висока дохідність (XUHA)</b>: {HY['score']:+d}/{HY['max_score']} — {HY['label']}",
        f"  XUHA {HY['metrics']['xuha']} vs SMA200 {HY['metrics']['sma200']} ({HY['states']['price_vs_sma200']})",
        f"  HY OAS {HY['metrics']['hy_oas_bp']} б.п. ({HY['metrics']['hy_oas_chg_bp']:+} за 20д), "
        f"CCC−BB {HY['metrics']['ccc_bb_bp']} б.п.",
        f"  VIX {HY['metrics']['vix']} ({HY['metrics']['vix_pct_1y']:.0f}-й перцентиль), "
        f"NFCI {HY['metrics']['nfci']} ({HY['metrics']['nfci_4w_chg']:+.2f} за 4 тижні)",
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
    p, f = cfg["params"], cfg["fred"]
    start = (dt.date.today() - dt.timedelta(days=cfg["history_days"])).isoformat()

    if args.demo:
        prices, fred = demo_data(cfg)
    else:
        key = os.getenv("FRED_API_KEY")
        if not key:
            print("FRED_API_KEY не задано", file=sys.stderr)
            return 1
        fred = {sid: fetch_fred(sid, start, key) for sid in f.values()}
        prices = fetch_prices(cfg["tickers"], start)

    warnings = []
    today = pd.Timestamp.today().normalize()

    stale_checks = [
        ("TLT",    prices["long"].dropna(),       p["stale_days"]),
        ("SUOA",   prices["corp"].dropna(),        p["stale_days"]),
        ("XUHA",   prices["hy"].dropna(),          p["stale_days"]),
        ("OAS",    fred[f["oas_ig"]],              p["stale_days"]),
        ("10Y",    fred[f["y10"]],                 p["stale_days"]),
        ("HY OAS", fred[f["hy_oas"]],              p["stale_days"]),
        ("NFCI",   fred[f["nfci"]],                p["stale_days_weekly"]),
        ("SLOOS",  fred[f["sloos"]],               p["stale_days_quarterly"]),
    ]
    for name, s, max_age in stale_checks:
        age = (today - s.dropna().index[-1]).days
        if age > max_age:
            warnings.append(f"{name}: дані застарілі ({s.dropna().index[-1].date()})")

    # TLT (NYSE) і SUOA (LSE) мають різні торгові дні — беремо лише спільні
    px = prices[["long", "corp"]].dropna()
    ratio = (px["long"] / px["corp"]).rename("ratio")

    daily = pd.DataFrame({k: fred[f[k]] for k in ("y2", "y10", "y30", "oas_ig", "oas_bbb")})
    daily = daily.sort_index().ffill(limit=5).dropna()

    need = p["sma_slow"] + p["range_len"]
    if len(ratio) < need or len(daily) < p["z_window"] + 2 * p["chg_window"]:
        print(f"Замало історії: ratio={len(ratio)}, fred={len(daily)} — збільште history_days",
              file=sys.stderr)
        return 1

    sig = {
        "as_of": str(max(ratio.index[-1], daily.index[-1]).date()),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "convention": "score > 0 = на користь TLT (ratio TLT/SUOA росте); < 0 = на користь SUOA",
        "hy_convention": "score > 0 = сприятливо для XUHA (risk-on); < 0 = несприятливо",
        "rates": rates_track(ratio, daily["y2"], daily["y10"], daily["y30"], p),
        "credit": credit_track(daily["oas_ig"], daily["oas_bbb"], p),
        "macro": macro_track(fred[f["cpi"]], fred[f["payrolls"]], fred[f["unrate"]]),
        "high_yield": high_yield_track(
            xuha=prices["hy"],
            suoa=prices["corp"],
            hy_oas=fred[f["hy_oas"]],
            hy_yield=fred[f["hy_yield"]],
            hy_bb=fred[f["hy_bb"]],
            hy_ccc=fred[f["hy_ccc"]],
            oas_ig=fred[f["oas_ig"]],
            vix=fred[f["vix"]],
            nfci=fred[f["nfci"]],
            sloos=fred[f["sloos"]],
            oil=fred[f["oil"]],
            p=p,
        ),
        "warnings": warnings,
    }
    for t in ("rates", "credit"):
        sig[t]["label"] = label(sig[t]["score"], sig[t]["max_score"])
    sig["high_yield"]["label"] = hy_label(sig["high_yield"]["score"], sig["high_yield"]["max_score"])
    sig["rates"]["metrics"]["ratio_date"] = str(ratio.index[-1].date())
    sig["credit"]["metrics"]["oas_date"] = str(daily.index[-1].date())

    prev = json.loads(SIGNALS.read_text(encoding="utf-8")) if SIGNALS.exists() else None
    alerts = diff_states(prev, sig)
    sig["alerts"] = alerts
    # зберігаємо останні зміни стану, щоб тижневий бриф бачив їх навіть після "тихих" днів
    log = (prev or {}).get("alert_log", [])
    log += [{"date": sig["as_of"], **a} for a in alerts]
    sig["alert_log"] = log[-60:]

    DATA.mkdir(exist_ok=True)
    SIGNALS.write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "date": sig["as_of"],
        "ratio": sig["rates"]["metrics"]["ratio"],
        "rsi": sig["rates"]["metrics"]["rsi"],
        "y10": sig["rates"]["metrics"]["y10"],
        "s10s30_bp": sig["rates"]["metrics"]["s10s30_bp"],
        "rv10y_bp": sig["rates"]["metrics"]["rv10y_bp"],
        "oas_ig_bp": sig["credit"]["metrics"]["oas_ig_bp"],
        "bbb_ig_gap_bp": sig["credit"]["metrics"]["bbb_ig_gap_bp"],
        "rates_score": sig["rates"]["score"],
        "credit_score": sig["credit"]["score"],
        "xuha": sig["high_yield"]["metrics"]["xuha"],
        "hy_oas_bp": sig["high_yield"]["metrics"]["hy_oas_bp"],
        "ccc_bb_bp": sig["high_yield"]["metrics"]["ccc_bb_bp"],
        "vix": sig["high_yield"]["metrics"]["vix"],
        "hy_score": sig["high_yield"]["score"],
    }
    hist = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame(columns=list(row))
    hist = pd.concat([hist[hist["date"] != row["date"]], pd.DataFrame([row])], ignore_index=True)
    hist.sort_values("date").to_csv(HISTORY, index=False)

    msg = format_message(sig, alerts)
    print(msg)
    if not args.no_notify and (alerts or cfg.get("telegram", {}).get("daily_summary") or prev is None):
        telegram(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
