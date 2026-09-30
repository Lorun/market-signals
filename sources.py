"""Data fetching and synthetic demo data generation."""
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
    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=today, periods=600)
    y10 = 4.2 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    y30 = y10 + 0.4 + np.cumsum(rng.normal(0, 0.01, len(idx)))
    y2 = y10 - 0.3 + np.cumsum(rng.normal(0, 0.02, len(idx)))
    oas = np.clip(0.9 + np.cumsum(rng.normal(0, 0.01, len(idx))), 0.5, 3)
    bbb = oas + 0.3 + np.cumsum(rng.normal(0, 0.004, len(idx)))
    tlt = 90 * np.exp(-16 * (y30 - y30[0]) / 100 + np.cumsum(rng.normal(0, 0.002, len(idx))))
    corp = 5 * np.exp(-6.5 * ((y10 + oas) - (y10[0] + oas[0])) / 100)
    midx = pd.date_range(end=idx[-1].replace(day=1), periods=30, freq="MS")
    f = cfg["fred"]

    # Keep original monthly series FIRST to preserve RNG state for regression
    cpi_raw = pd.Series(300 * np.exp(np.cumsum(rng.normal(0.0025, 0.001, 30))), midx)
    payrolls_raw = pd.Series(158000 + np.cumsum(rng.normal(150, 80, 30)), midx)
    unrate_raw = pd.Series(4.1 + np.cumsum(rng.normal(0, 0.05, 30)), midx)

    # HY bond series (daily) — generated after monthly to preserve existing RNG sequence
    hy_oas_raw = np.clip(3.5 + np.cumsum(rng.normal(0, 0.03, len(idx))), 2.5, 6.0)
    hy_bb_raw = np.clip(hy_oas_raw - 0.5 + np.cumsum(rng.normal(0, 0.01, len(idx))), 1.5, 5.5)
    hy_ccc_raw = np.clip(hy_oas_raw + 4.0 + np.cumsum(rng.normal(0, 0.05, len(idx))), 5.0, 14.0)
    hy_yield_raw = np.clip(hy_oas_raw + 4.5 + np.cumsum(rng.normal(0, 0.01, len(idx))), 5.0, 15.0)

    # VIX (daily, mean-reverting)
    vix_raw = np.clip(18 + np.cumsum(rng.normal(0, 0.3, len(idx))), 10, 45)

    # WTI oil (daily)
    oil_raw = np.clip(80 + np.cumsum(rng.normal(0, 0.8, len(idx))), 40, 120)

    # XUHA price (daily, HY bond ETF ~10.5 USD)
    xuha_raw = 10.5 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, len(idx))))

    # NFCI (weekly, Wednesdays, ~85 weeks)
    widx = pd.bdate_range(end=today, periods=600, freq="W-WED")[-85:]
    nfci_raw = np.cumsum(rng.normal(0, 0.05, len(widx)))
    nfci_raw = np.clip(nfci_raw - nfci_raw.mean(), -1.5, 1.5)

    # SLOOS (quarterly, ~12 quarters)
    qidx = pd.date_range(end=today, periods=12, freq="QS")
    sloos_raw = np.clip(10 + np.cumsum(rng.normal(0, 3, 12)), -20, 60)

    fred = {
        f["y2"]: pd.Series(y2, idx), f["y10"]: pd.Series(y10, idx), f["y30"]: pd.Series(y30, idx),
        f["oas_ig"]: pd.Series(oas, idx), f["oas_bbb"]: pd.Series(bbb, idx),
        f["cpi"]: cpi_raw,
        f["payrolls"]: payrolls_raw,
        f["unrate"]: unrate_raw,
        f["hy_oas"]: pd.Series(hy_oas_raw, idx),
        f["hy_yield"]: pd.Series(hy_yield_raw, idx),
        f["hy_bb"]: pd.Series(hy_bb_raw, idx),
        f["hy_ccc"]: pd.Series(hy_ccc_raw, idx),
        f["vix"]: pd.Series(vix_raw, idx),
        f["nfci"]: pd.Series(nfci_raw, widx),
        f["sloos"]: pd.Series(sloos_raw, qidx),
        f["oil"]: pd.Series(oil_raw, idx),
    }
    prices = pd.DataFrame({"long": tlt, "corp": corp, "hy": xuha_raw}, index=idx)
    return prices, fred
