#!/usr/bin/env python3
"""
TLT / SUOA signal monitor.

Два незалежні треки:
  * RATES  — рух ставок і технічка ratio TLT/SUOA (дюраційний трек)
  * CREDIT — кредитний спред (OAS) напряму з FRED, не залежить від дюрації
  + MACRO  — контекст (інфляція, ринок праці), без голосів у скорингу

Знак скорингу: "+" = на користь TLT (ratio TLT/SUOA росте), "−" = на користь SUOA.

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

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SIGNALS = DATA / "signals.json"
HISTORY = DATA / "history.csv"


# ───────────────────────────── data ─────────────────────────────

def fetch_fred(series_id: str, start: str, api_key: str) -> pd.Series:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": api_key,
                "file_type": "json", "observation_start": start},
        timeout=30,
    )
    r.raise_for_status()
    vals = {pd.Timestamp(o["date"]): float(o["value"])
            for o in r.json()["observations"] if o["value"] not in (".", "")}
    return pd.Series(vals, name=series_id, dtype=float).sort_index()


def fetch_prices(tickers: dict, start: str) -> pd.DataFrame:
    import yfinance as yf
    out = {}
    for role, t in tickers.items():
        df = yf.download(t, start=start, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError(f"Немає цін для {t} — перевірте тікер у config.yaml")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        out[role] = close.astype(float)
    return pd.DataFrame(out)


def demo_data(cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Синтетичні ряди для перевірки логіки без мережі."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
    y10 = 4.2 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    y30 = y10 + 0.4 + np.cumsum(rng.normal(0, 0.01, len(idx)))
    y2 = y10 - 0.3 + np.cumsum(rng.normal(0, 0.02, len(idx)))
    oas = np.clip(0.9 + np.cumsum(rng.normal(0, 0.01, len(idx))), 0.5, 3)
    bbb = oas + 0.3 + np.cumsum(rng.normal(0, 0.004, len(idx)))
    tlt = 90 * np.exp(-16 * (y30 - y30[0]) / 100 + np.cumsum(rng.normal(0, 0.002, len(idx))))
    corp = 5 * np.exp(-6.5 * ((y10 + oas) - (y10[0] + oas[0])) / 100)
    prices = pd.DataFrame({"long": tlt, "corp": corp}, index=idx)
    midx = pd.date_range(end=idx[-1].replace(day=1), periods=30, freq="MS")
    f = cfg["fred"]
    fred = {
        f["y2"]: pd.Series(y2, idx), f["y10"]: pd.Series(y10, idx), f["y30"]: pd.Series(y30, idx),
        f["oas_ig"]: pd.Series(oas, idx), f["oas_bbb"]: pd.Series(bbb, idx),
        f["cpi"]: pd.Series(300 * np.exp(np.cumsum(rng.normal(0.0025, 0.001, 30))), midx),
        f["payrolls"]: pd.Series(158000 + np.cumsum(rng.normal(150, 80, 30)), midx),
        f["unrate"]: pd.Series(4.1 + np.cumsum(rng.normal(0, 0.05, 30)), midx),
    }
    return prices, fred


# ─────────────────────────── indicators ───────────────────────────

def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd_hist(s: pd.Series) -> pd.Series:
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return m - m.ewm(span=9, adjust=False).mean()


def divergence(price: pd.Series, osc: pd.Series, lookback: int, recent: int, margin: float) -> str:
    """bearish: ціна зробила новий max за lookback в останні `recent` днів, а RSI — ні.
       bullish: дзеркально для min."""
    for i in range(1, recent + 1):
        p_win = price.iloc[-lookback - i:-i]
        o_win = osc.iloc[-lookback - i:-i]
        p, o = price.iloc[-i], osc.iloc[-i]
        if len(p_win) < lookback or np.isnan(o):
            continue
        if p > p_win.max() and o < o_win.max() - margin:
            return "bearish"
        if p < p_win.min() and o > o_win.min() + margin:
            return "bullish"
    return "none"


def zscore(s: pd.Series, n: int) -> float:
    w = s.iloc[-n:]
    sd = w.std()
    return float((w.iloc[-1] - w.mean()) / sd) if sd > 0 else 0.0


def pct_rank(s: pd.Series, n: int) -> float:
    w = s.dropna().iloc[-n:]
    return float((w < w.iloc[-1]).mean() * 100)


def trend3(chg: float, thr: float, up: str, down: str) -> str:
    return up if chg > thr else down if chg < -thr else "flat"


def r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


# ───────────────────────────── tracks ─────────────────────────────

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


def macro_track(cpi: pd.Series, payrolls: pd.Series, unrate: pd.Series) -> dict:
    yoy = (cpi / cpi.shift(12) - 1) * 100
    nfp = payrolls.diff()  # тис. робочих місць за місяць
    nfp3 = nfp.rolling(3).mean()
    states = {
        "inflation": "accelerating" if yoy.iloc[-1] > yoy.iloc[-4] + 0.1
                     else "decelerating" if yoy.iloc[-1] < yoy.iloc[-4] - 0.1 else "stable",
        "labor": "weakening" if (nfp3.iloc[-1] < nfp3.iloc[-4]) and (unrate.iloc[-1] > unrate.iloc[-4])
                 else "strengthening" if (nfp3.iloc[-1] > nfp3.iloc[-4]) and (unrate.iloc[-1] <= unrate.iloc[-4])
                 else "mixed",
    }
    metrics = {
        "cpi_yoy": r(yoy.iloc[-1]), "cpi_yoy_3m_ago": r(yoy.iloc[-4]),
        "cpi_date": str(cpi.index[-1].date()),
        "nfp_last_k": r(nfp.iloc[-1], 0), "nfp_3m_avg_k": r(nfp3.iloc[-1], 0),
        "unrate": r(unrate.iloc[-1], 1), "unrate_3m_ago": r(unrate.iloc[-4], 1),
        "labor_date": str(payrolls.index[-1].date()),
    }
    return {"states": states, "metrics": metrics}


def label(score: int, mx: int) -> str:
    if score >= mx * 0.5:
        return "сильно на користь TLT"
    if score > 0:
        return "помірно на користь TLT"
    if score <= -mx * 0.5:
        return "сильно на користь SUOA"
    if score < 0:
        return "помірно на користь SUOA"
    return "нейтрально"


# ───────────────────────────── alerts ─────────────────────────────

def diff_states(prev: dict | None, cur: dict) -> list[dict]:
    if not prev:
        return []
    alerts = []
    for track in ("rates", "credit", "macro"):
        old = prev.get(track, {}).get("states", {})
        for k, v in cur[track]["states"].items():
            if k in old and old[k] != v:
                alerts.append({"track": track, "indicator": k, "from": old[k], "to": v})
        if track != "macro":
            o, n = prev.get(track, {}).get("score"), cur[track]["score"]
            if o is not None and label(o, cur[track]["max_score"]) != label(n, cur[track]["max_score"]):
                alerts.append({"track": track, "indicator": "score_regime",
                               "from": f"{o} ({label(o, cur[track]['max_score'])})",
                               "to": f"{n} ({label(n, cur[track]['max_score'])})"})
    return alerts


def telegram(text: str) -> None:
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[telegram] токен/chat_id не задано — пропускаю")
        return
    resp = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                         json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                               "disable_web_page_preview": True}, timeout=20)
    if not resp.ok:
        print(f"[telegram] помилка {resp.status_code}: {resp.text}", file=sys.stderr)


TRACK_UA = {"rates": "Ставки", "credit": "Кредит", "macro": "Макро"}


def format_message(sig: dict, alerts: list[dict]) -> str:
    R, C = sig["rates"], sig["credit"]
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
    for name, s in [("TLT", prices["long"].dropna()), ("SUOA", prices["corp"].dropna()),
                    ("OAS", fred[f["oas_ig"]]), ("10Y", fred[f["y10"]])]:
        age = (today - s.index[-1]).days
        if age > p["stale_days"]:
            warnings.append(f"{name}: дані застарілі ({s.index[-1].date()})")

    # TLT (NYSE) і SUOA (LSE) мають різні торгові дні — беремо лише спільні
    px = prices.dropna()
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
        "rates": rates_track(ratio, daily["y2"], daily["y10"], daily["y30"], p),
        "credit": credit_track(daily["oas_ig"], daily["oas_bbb"], p),
        "macro": macro_track(fred[f["cpi"]], fred[f["payrolls"]], fred[f["unrate"]]),
        "warnings": warnings,
    }
    for t in ("rates", "credit"):
        sig[t]["label"] = label(sig[t]["score"], sig[t]["max_score"])
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

    row = {"date": sig["as_of"], "ratio": sig["rates"]["metrics"]["ratio"],
           "rsi": sig["rates"]["metrics"]["rsi"], "y10": sig["rates"]["metrics"]["y10"],
           "s10s30_bp": sig["rates"]["metrics"]["s10s30_bp"],
           "rv10y_bp": sig["rates"]["metrics"]["rv10y_bp"],
           "oas_ig_bp": sig["credit"]["metrics"]["oas_ig_bp"],
           "bbb_ig_gap_bp": sig["credit"]["metrics"]["bbb_ig_gap_bp"],
           "rates_score": sig["rates"]["score"], "credit_score": sig["credit"]["score"]}
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
