from __future__ import annotations

import pandas as pd

from indicators import r


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
