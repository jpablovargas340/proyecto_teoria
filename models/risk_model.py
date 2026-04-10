from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def kupiec_test(returns: pd.Series, var_threshold: float, confidence: float):
    violations = int((returns < var_threshold).sum())
    n = len(returns)
    p = 1 - confidence
    expected = n * p
    if n == 0 or violations in (0, n):
        return {"expected_violations": expected, "actual_violations": violations, "p_value": 1.0}
    pi_hat = violations / n
    lr = -2 * np.log((((1-p)**(n-violations)) * (p**violations)) / (((1-pi_hat)**(n-violations)) * (pi_hat**violations)))
    p_value = 1 - stats.chi2.cdf(lr, 1)
    return {"expected_violations": expected, "actual_violations": violations, "p_value": float(p_value)}

def risk_report(returns: pd.Series, portfolio_value: float, confidence: float):
    r = returns.dropna()
    alpha = 1 - confidence
    mu = r.mean()
    sigma = r.std()

    var_hist = np.percentile(r, alpha * 100)
    z = stats.norm.ppf(alpha)
    var_param = mu + z * sigma
    sims = np.random.default_rng(42).normal(mu, sigma, 10000)
    var_mc = np.percentile(sims, alpha * 100)
    cvar = r[r <= var_hist].mean() if len(r[r <= var_hist]) else var_hist

    annual_scale = np.sqrt(252)
    table = pd.DataFrame([
        {"Método": "Histórico", "VaR %": float(var_hist), "VaR anual %": float(var_hist * annual_scale), "VaR $": abs(float(var_hist * portfolio_value))},
        {"Método": "Paramétrico", "VaR %": float(var_param), "VaR anual %": float(var_param * annual_scale), "VaR $": abs(float(var_param * portfolio_value))},
        {"Método": "Monte Carlo", "VaR %": float(var_mc), "VaR anual %": float(var_mc * annual_scale), "VaR $": abs(float(var_mc * portfolio_value))},
        {"Método": "CVaR", "VaR %": float(cvar), "VaR anual %": float(cvar * annual_scale), "VaR $": abs(float(cvar * portfolio_value))},
    ])

    kupiec = kupiec_test(r, var_hist, confidence)
    interpretation = f"El VaR histórico captura la pérdida umbral observada al {confidence:.0%}; el VaR paramétrico depende de normalidad; Monte Carlo aporta una aproximación simulada; y el CVaR estima la pérdida promedio más allá del VaR."
    return {
        "table": table,
        "var_hist": float(var_hist),
        "var_param": float(var_param),
        "var_mc": float(var_mc),
        "cvar": float(cvar),
        "kupiec": kupiec,
        "interpretation": interpretation,
    }
