from __future__ import annotations
import pandas as pd
from scipy import stats

def capm_report(asset_returns: pd.Series, market_returns: pd.Series, rf: float, ticker: str):
    if market_returns is None or len(market_returns) == 0:
        return {
            "ticker": ticker,
            "beta": 1.0,
            "alpha_anual": 0.0,
            "r_squared": 0.0,
            "expected_return": rf,
            "classification": "Neutro",
            "interpretation": "No fue posible descargar benchmark, por lo que el CAPM quedó con parámetros por defecto.",
            "scatter_df": pd.DataFrame(columns=["market", "asset", "fit"]),
        }

    idx = asset_returns.index.intersection(market_returns.index)
    if len(idx) < 30:
        return {
            "ticker": ticker,
            "beta": 1.0,
            "alpha_anual": 0.0,
            "r_squared": 0.0,
            "expected_return": rf,
            "classification": "Neutro",
            "interpretation": "No hay suficientes observaciones comunes con el benchmark para estimar beta de forma robusta.",
            "scatter_df": pd.DataFrame(columns=["market", "asset", "fit"]),
        }

    a = asset_returns.loc[idx]
    m = market_returns.loc[idx]
    slope, intercept, r_value, p_value, std_err = stats.linregress(m, a)
    market_return = float(m.mean() * 252)
    expected = rf + slope * (market_return - rf)

    if slope > 1.2:
        classification = "Agresivo"
    elif slope > 0.8:
        classification = "Neutro"
    elif slope > 0.4:
        classification = "Defensivo"
    else:
        classification = "Muy defensivo"

    fit = intercept + slope * m
    scatter_df = pd.DataFrame({"market": m, "asset": a, "fit": fit})
    interpretation = f"{ticker} tiene beta {slope:.2f}, clasificada como {classification.lower()}. La beta resume el riesgo sistemático; el componente no sistemático puede reducirse mediante diversificación."

    return {
        "ticker": ticker,
        "beta": float(slope),
        "alpha_anual": float(intercept * 252),
        "r_squared": float(r_value ** 2),
        "expected_return": float(expected),
        "classification": classification,
        "interpretation": interpretation,
        "scatter_df": scatter_df,
    }
