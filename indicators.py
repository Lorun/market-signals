"""Pure technical indicator functions — no I/O, no config dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
