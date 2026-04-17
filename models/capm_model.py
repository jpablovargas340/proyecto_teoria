from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def capm_report(asset_returns: pd.Series, market_returns: pd.Series, rf: float, ticker: str):
    if market_returns is None or len(market_returns) == 0:
        return {
            "ticker": ticker,
            "beta": np.nan,
            "alpha_anual": np.nan,
            "r_squared": np.nan,
            "expected_return": np.nan,
            "classification": "Sin benchmark",
            "interpretation": "No fue posible descargar el benchmark de mercado; por eso no se puede estimar beta ni graficar la relación CAPM.",
            "scatter_df": pd.DataFrame(columns=["market", "asset", "fit"]),
        }

    idx = asset_returns.index.intersection(market_returns.index)
    if len(idx) < 30:
        return {
            "ticker": ticker,
            "beta": np.nan,
            "alpha_anual": np.nan,
            "r_squared": np.nan,
            "expected_return": np.nan,
            "classification": "Muestra insuficiente",
            "interpretation": "No hay suficientes observaciones comunes con el benchmark para estimar beta de forma robusta.",
            "scatter_df": pd.DataFrame(columns=["market", "asset", "fit"]),
        }

    paired = pd.DataFrame({
        "asset": asset_returns.loc[idx],
        "market": market_returns.loc[idx]
    }).dropna()

    if len(paired) < 30:
        return {
            "ticker": ticker,
            "beta": np.nan,
            "alpha_anual": np.nan,
            "r_squared": np.nan,
            "expected_return": np.nan,
            "classification": "Muestra insuficiente",
            "interpretation": "No hay suficientes observaciones válidas y comunes con el benchmark para estimar beta de forma robusta.",
            "scatter_df": pd.DataFrame(columns=["market", "asset", "fit"]),
        }

    a = paired["asset"]
    m = paired["market"]

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

    interpretation = (
        f"{ticker} tiene beta {slope:.2f}, clasificada como {classification.lower()}. "
        f"Su retorno esperado por CAPM es {expected:.2%} y el R² es {r_value**2:.2%}. "
        "La beta resume el riesgo sistemático; el componente no sistemático puede reducirse mediante diversificación."
    )

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