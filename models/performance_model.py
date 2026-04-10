from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def max_drawdown(returns: pd.Series):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = cumulative / peak - 1
    return float(dd.min()) if len(dd) else 0.0

def sharpe_ratio(returns: pd.Series, rf: float):
    if len(returns) == 0:
        return 0.0
    vol = returns.std() * np.sqrt(252)
    if vol == 0:
        return 0.0
    return float((returns.mean() * 252 - rf) / vol)

def annual_return(returns: pd.Series):
    return float(returns.mean() * 252) if len(returns) else 0.0

def annual_vol(returns: pd.Series):
    return float(returns.std() * np.sqrt(252)) if len(returns) else 0.0

def benchmark_report(portfolio_returns: pd.Series, benchmark_returns: pd.Series, rf: float):
    if benchmark_returns is None or len(benchmark_returns) == 0:
        return {
            "alpha": 0.0, "tracking_error": 0.0, "information_ratio": 0.0,
            "portfolio_sharpe": sharpe_ratio(portfolio_returns, rf), "benchmark_sharpe": 0.0,
            "portfolio_cum_return": float((1 + portfolio_returns).prod() - 1) if len(portfolio_returns) else 0.0,
            "benchmark_cum_return": 0.0, "portfolio_annual_return": annual_return(portfolio_returns),
            "benchmark_annual_return": 0.0, "portfolio_vol": annual_vol(portfolio_returns),
            "benchmark_vol": 0.0, "max_drawdown": max_drawdown(portfolio_returns),
            "interpretation": "No se pudo calcular benchmark de mercado. El resto de métricas del portafolio sigue disponible."
        }

    idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(idx) < 30:
        return {
            "alpha": 0.0, "tracking_error": 0.0, "information_ratio": 0.0,
            "portfolio_sharpe": sharpe_ratio(portfolio_returns, rf), "benchmark_sharpe": sharpe_ratio(benchmark_returns, rf),
            "portfolio_cum_return": float((1 + portfolio_returns).prod() - 1), "benchmark_cum_return": float((1 + benchmark_returns).prod() - 1),
            "portfolio_annual_return": annual_return(portfolio_returns), "benchmark_annual_return": annual_return(benchmark_returns),
            "portfolio_vol": annual_vol(portfolio_returns), "benchmark_vol": annual_vol(benchmark_returns),
            "max_drawdown": max_drawdown(portfolio_returns),
            "interpretation": "No hay suficientes observaciones comunes para una comparación benchmark robusta."
        }

    p = portfolio_returns.loc[idx]
    b = benchmark_returns.loc[idx]
    excess_p = p - rf / 252
    excess_b = b - rf / 252
    slope, intercept, r_value, p_value, std_err = stats.linregress(excess_b, excess_p)

    diff = p - b
    tracking_error = float(diff.std() * np.sqrt(252))
    information_ratio = float((diff.mean() * 252) / tracking_error) if tracking_error > 0 else 0.0

    interpretation = "El alpha de Jensen evalúa si el portafolio agrega valor sobre lo explicado por el riesgo sistemático. Tracking Error e Information Ratio permiten juzgar la calidad del desempeño relativo frente al benchmark."
    return {
        "alpha": float(intercept * 252), "tracking_error": tracking_error, "information_ratio": information_ratio,
        "portfolio_sharpe": sharpe_ratio(p, rf), "benchmark_sharpe": sharpe_ratio(b, rf),
        "portfolio_cum_return": float((1 + p).prod() - 1), "benchmark_cum_return": float((1 + b).prod() - 1),
        "portfolio_annual_return": annual_return(p), "benchmark_annual_return": annual_return(b),
        "portfolio_vol": annual_vol(p), "benchmark_vol": annual_vol(b), "max_drawdown": max_drawdown(p),
        "interpretation": interpretation,
    }
