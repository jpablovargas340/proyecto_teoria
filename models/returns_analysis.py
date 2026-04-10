from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def stylized_facts(returns: pd.Series):
    r = returns.dropna()
    kurt = r.kurtosis()
    sq = r ** 2
    acf_sq = sq.autocorr(lag=1)
    acf_ret = r.autocorr(lag=1)
    leverage = sq.corr(r.shift(1))
    return {
        "heavy_tails": bool(kurt > 3),
        "volatility_clustering": bool(acf_sq > 0.1),
        "market_efficiency_proxy": bool(abs(acf_ret) < 0.1),
        "leverage_effect": bool(leverage < 0),
    }

def full_returns_report(log_returns: pd.Series):
    r = log_returns.dropna()
    simple_r = np.exp(r) - 1

    jb_stat, jb_p = stats.jarque_bera(r)
    sample = r.iloc[:5000] if len(r) > 5000 else r
    sw_stat, sw_p = stats.shapiro(sample)

    facts = stylized_facts(r)
    interpretation = (
        "Los rendimientos exhiben "
        + ("colas pesadas, " if facts["heavy_tails"] else "colas no extremas, ")
        + ("agrupamiento de volatilidad, " if facts["volatility_clustering"] else "poco clustering, ")
        + ("y evidencia de efecto apalancamiento." if facts["leverage_effect"] else "y poco efecto apalancamiento.")
    )

    return {
        "returns": r,
        "simple_returns": simple_r,
        "simple_annual_mean": float(simple_r.mean() * 252),
        "log_annual_mean": float(r.mean() * 252),
        "annual_vol": float(r.std() * np.sqrt(252)),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_p),
        "sw_stat": float(sw_stat),
        "sw_pvalue": float(sw_p),
        "facts": facts,
        "interpretation": interpretation,
    }
