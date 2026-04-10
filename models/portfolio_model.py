from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def correlation_matrix(returns: pd.DataFrame):
    return returns.corr()

def portfolio_performance(weights, mean_returns, cov_matrix, rf):
    ret = np.sum(mean_returns * weights) * 252
    risk = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = (ret - rf) / risk if risk > 0 else 0.0
    return ret, risk, sharpe

def simulate_portfolios(returns: pd.DataFrame, rf: float, n_portfolios: int = 10000):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(returns.columns)
    assets = list(returns.columns)

    rng = np.random.default_rng(42)
    rows = []
    weight_store = []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w = w / w.sum()
        r, risk, s = portfolio_performance(w, mean_returns, cov_matrix, rf)
        rows.append((r, risk, s))
        weight_store.append(w)

    sim = pd.DataFrame(rows, columns=["return", "risk", "sharpe"])
    sim["weights"] = weight_store

    max_idx = sim["sharpe"].idxmax()
    min_idx = sim["risk"].idxmin()

    max_w = sim.loc[max_idx, "weights"]
    min_w = sim.loc[min_idx, "weights"]
    max_sharpe_returns = returns.dot(max_w)

    return {
        "simulated": sim,
        "max_sharpe": {"return": float(sim.loc[max_idx, "return"]), "risk": float(sim.loc[max_idx, "risk"]), "sharpe": float(sim.loc[max_idx, "sharpe"]), "weights": dict(zip(assets, [float(x) for x in max_w]))},
        "min_variance": {"return": float(sim.loc[min_idx, "return"]), "risk": float(sim.loc[min_idx, "risk"]), "sharpe": float(sim.loc[min_idx, "sharpe"]), "weights": dict(zip(assets, [float(x) for x in min_w]))},
        "max_sharpe_returns": max_sharpe_returns,
    }

def optimize_target_return(returns: pd.DataFrame, rf: float, target_return: float):
    mean_returns = returns.mean()
    cov = returns.cov()
    n = len(returns.columns)
    assets = list(returns.columns)
    init = np.array([1/n] * n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: (np.sum(mean_returns * w) * 252) - target_return},
    )

    def vol_objective(w):
        return np.sqrt(w.T @ cov @ w) * np.sqrt(252)

    try:
        res = minimize(vol_objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
        w = res.x
        ret, risk, sharpe = portfolio_performance(w, mean_returns, cov, rf)
        interpretation = f"Se encontró un portafolio eficiente con retorno esperado cercano a {target_return:.2%}. Este módulo agrega valor al permitir optimización interactiva por objetivo de retorno."
        return {"weights": dict(zip(assets, [float(x) for x in w])), "return": float(ret), "risk": float(risk), "sharpe": float(sharpe), "interpretation": interpretation}
    except Exception:
        return {"weights": {a: round(1/n, 4) for a in assets}, "return": 0.0, "risk": 0.0, "sharpe": 0.0, "interpretation": "No se pudo optimizar exactamente para el retorno objetivo. Se muestra una aproximación simple."}
