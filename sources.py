"""Data fetching, series chaining and synthetic demo data generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests


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


def _download(t: str, start: str) -> tuple[pd.Series, list[str]]:
    """Скоригована денна ціна закриття. Yahoo інколи вже має рядок за останній день, але з порожньою
    ціною (буває для LSE/Euronext кілька годин після закриття) — тоді беремо останню погодинну ціну
    того дня. Повертає (ряд, дати, заповнені з погодинних даних)."""
    import yfinance as yf
    df = yf.download(t, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Немає цін для {t} — перевірте тікер у config.yaml")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.astype(float)
    filled = []
    missing = close.index[close.isna()]
    missing = missing[missing >= close.index[-1] - pd.Timedelta(days=7)]
    if len(missing):
        h = yf.Ticker(t).history(period="5d", interval="1h", auto_adjust=True)["Close"].dropna()
        if len(h):
            h.index = h.index.tz_localize(None)
            for d in missing:
                day = h[h.index.normalize() == d]
                if len(day):
                    close[d] = float(day.iloc[-1])
                    filled.append(str(d.date()))
    return close.dropna(), filled


def fetch_prices(tickers: dict, start: str) -> pd.DataFrame:
    """Ціни з Yahoo. У `.attrs["intraday_filled"]` — які тикери й дні взято з погодинних даних."""
    out, filled = {}, {}
    for role, t in tickers.items():
        out[role], f = _download(t, start)
        if f:
            filled[t] = f
    df = pd.DataFrame(out)
    df.attrs["intraday_filled"] = filled
    return df


# ───────────────────────────── склейка рядів ─────────────────────────────

def rate_to_returns(rate_pct: pd.Series) -> pd.Series:
    """Річна ставка T-bill (%) → щоденна дохідність: ставка попереднього дня × календарні дні / 360."""
    s = rate_pct.dropna()
    days = s.index.to_series().diff().dt.days
    return (s.shift(1) / 100 * days / 360).dropna()


def chain_returns(segments: list[tuple[pd.Series, str | pd.Timestamp]], kind: str = "price") -> pd.Series:
    """Склеює ряди за щоденною дохідністю в один безперервний індекс (старт = 100).

    segments: [(ряд, дата_початку), ...] у хронологічному порядку; сегмент діє
    від своєї дати до дати наступного. kind="price" — ціни, "return" — уже дохідності."""
    parts = []
    for i, (s, frm) in enumerate(segments):
        rets = s.dropna().pct_change().dropna() if kind == "price" else s.dropna()
        frm = pd.Timestamp(frm)
        to = pd.Timestamp(segments[i + 1][1]) if i + 1 < len(segments) else None
        rets = rets[rets.index >= frm]
        if to is not None:
            rets = rets[rets.index < to]
        parts.append(rets)
    rets = pd.concat(parts).sort_index()
    rets = rets[~rets.index.duplicated(keep="last")]
    return 100 * (1 + rets).cumprod()


def align(df: pd.DataFrame | pd.Series, index: pd.DatetimeIndex, limit: int = 5):
    """Вирівнювання на торговий календар NYSE (індекс TLT): різні біржі / FRED → ffill(limit)."""
    full = df.index.union(index)
    return df.reindex(full).ffill(limit=limit).reindex(index)


def lag_series(s: pd.Series, days: int) -> pd.Series:
    """Значення за день D діє з D+days робочих днів (дані FRED публікуються наступного робочого дня).
    Спостереження у вихідні (є в ICE OAS) і п'ятниця потрапляють на той самий понеділок — лишаємо найсвіжіше."""
    s = s.dropna()
    if not days:
        return s
    out = pd.Series(s.to_numpy(), index=s.index + pd.offsets.BDay(days), name=s.name)
    return out[~out.index.duplicated(keep="last")]


def unit_jumps(s: pd.Series, thr: float = 0.5) -> list[str]:
    """Дати, де денна зміна |r| > thr — найімовірніше зміна одиниць котирування (GBp/USD), а не ринок."""
    r = s.dropna().pct_change().abs()
    return [str(d.date()) for d in r[r > thr].index]


def clean_returns(r: pd.Series, thr: float = 0.5) -> pd.Series:
    """Обнуляє денні дохідності |r| > thr — для ETF це глюк котирування, а не ринок (див. unit_jumps)."""
    return r.where(r.abs() <= thr, 0.0)


def load_market(cfg: dict, start: str, fred_key: str | None, *, backtest: bool = False
                ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Тягне всі ряди. Повертає (prices, fred), де prices має колонки
    core, cta, cta_exec, equity, cash (+ проксі для бектесту) за датами бірж, без вирівнювання."""
    f = cfg["fred"]
    fred = {k: fetch_fred(sid, start, fred_key) for k, sid in f.items()}
    ys = fetch_prices(cfg["yahoo_series"], start)       # ^TNX, ^TYX, ^VIX — ті самі одиниці, що й у FRED
    fred.update({k: ys[k].dropna() for k in cfg["yahoo_series"]})
    tickers = dict(cfg["tickers"])
    if backtest:
        for block, segs in cfg["proxies"].items():
            for seg in segs:
                if "ticker" in seg:
                    tickers[f"px_{seg['ticker']}"] = seg["ticker"]
    prices = fetch_prices(tickers, start)
    prices.attrs["intraday_filled"].update(ys.attrs.get("intraday_filled", {}))
    return prices, fred


def build_block_prices(prices: pd.DataFrame, fred: dict[str, pd.Series], cfg: dict,
                       backtest: bool) -> pd.DataFrame:
    """Індекси блоків (core/cta/equity/cash) за сирими датами.
    Живий режим: ряди як є; кеш до появи IB01 — з DTB3. Бектест: склейка проксі з config.proxies."""
    out = {"core": prices["core"].dropna()}
    if not backtest:
        out["cta"] = prices["cta"].dropna()
        out["equity"] = prices["equity"].dropna()
        out["cta_exec"] = prices["cta_exec"].dropna()
        ib01 = clean_returns(prices["cash"].dropna().pct_change().dropna())
        cash_segs = [(rate_to_returns(fred["tbill"]), "1900-01-01"), (ib01, ib01.index[0])]
        out["cash"] = chain_returns(cash_segs, kind="return")
        return pd.concat(out, axis=1)
    for block, segs in cfg["proxies"].items():
        parts = []
        for seg in segs:
            if "fred" in seg:
                key = next(k for k, v in cfg["fred"].items() if v == seg["fred"])
                parts.append((rate_to_returns(fred[key]), seg["from"]))
            else:
                parts.append((clean_returns(prices[f"px_{seg['ticker']}"].dropna().pct_change().dropna()),
                              seg["from"]))
        out[block] = chain_returns(parts, kind="return")
    return pd.concat(out, axis=1)


# ───────────────────────────── demo ─────────────────────────────

def demo_data(cfg: dict, periods: int = 2200) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Синтетичні ряди для перевірки логіки без мережі (~8,5 років робочих днів)."""
    rng = np.random.default_rng(42)
    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today, periods=periods)
    n = len(idx)
    # 10Y: спокійний період, потім стійкий ріст ставок в останні ~9 місяців (щоб рушій мав що робити)
    drift = np.where(np.arange(n) > n - 190, 0.006, 0.0)
    y10 = 4.0 + np.cumsum(rng.normal(0, 0.03, n) + drift)
    y30 = y10 + 0.35 + np.cumsum(rng.normal(0, 0.008, n))
    y2 = y10 - 0.3 + np.cumsum(rng.normal(0, 0.02, n))
    tbill = np.clip(y2 - 0.2, 0.05, None)
    oas = np.clip(1.1 + np.cumsum(rng.normal(0, 0.01, n)), 0.6, 3)
    bbb = oas + 0.3 + np.cumsum(rng.normal(0, 0.004, n))
    baa = oas + 0.8
    hy = np.clip(3.5 + np.cumsum(rng.normal(0, 0.03, n)), 2.5, 8)
    vix = np.clip(17 + np.cumsum(rng.normal(0, 0.4, n)), 10, 60)
    tlt = 100 * np.exp(-17 * (y30 - y30[0]) / 100 + np.cumsum(rng.normal(0.0001, 0.003, n)))
    eq = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n)))
    cta_r = rng.normal(0.0001, 0.006, n) + 0.25 * np.r_[0, np.diff(y10)] / 100 * 17
    cta = 30 * np.exp(np.cumsum(cta_r))
    cta_exec = 4.2 * cta * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    cash = 100 * np.cumprod(1 + tbill / 100 / 252)

    midx = pd.date_range(end=idx[-1].replace(day=1), periods=100, freq="MS")
    widx = pd.date_range(end=today, periods=n // 5, freq="W-FRI")
    nfci = np.clip(np.cumsum(rng.normal(0, 0.05, len(widx))) * 0.3 - 0.4, -1, 1.5)
    D = lambda a: pd.Series(a, idx)
    fred = {
        "y2": D(y2), "y10": D(y10), "y30": D(y30), "y10_fred": D(y10), "tbill": D(tbill),
        "oas_ig": D(oas)[-780:], "oas_bbb": D(bbb)[-780:], "baa10y": D(baa),
        "hy_oas": D(hy)[-780:], "vix": D(vix),
        "nfci": pd.Series(nfci, widx),
        "cpi": pd.Series(250 * np.exp(np.cumsum(rng.normal(0.0025, 0.001, 100))), midx),
        "payrolls": pd.Series(150000 + np.cumsum(rng.normal(150, 80, 100)), midx),
        "unrate": pd.Series(4.0 + np.cumsum(rng.normal(0, 0.05, 100)), midx),
    }
    prices = pd.DataFrame({"core": tlt, "cta": cta, "cta_exec": cta_exec, "equity": eq,
                           "cash": cash}, index=idx)
    return prices, fred
