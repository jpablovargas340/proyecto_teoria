from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def fit_garch_suite(returns: pd.Series):
    r = returns.dropna() * 100
    if len(r) < 120:
        return {
            "table": pd.DataFrame([{"Modelo": "Insuficiente", "LogLik": None, "AIC": None, "BIC": None}]),
            "forecast": None,
            "std_resid": None,
            "cond_vol": None,
            "residual_jb": None,
            "interpretation": "No hay suficientes observaciones para comparar modelos GARCH con robustez."
        }
    try:
        from arch import arch_model
    except Exception:
        return {
            "table": pd.DataFrame([{"Modelo": "arch no instalado", "LogLik": None, "AIC": None, "BIC": None}]),
            "forecast": None,
            "std_resid": None,
            "cond_vol": None,
            "residual_jb": None,
            "interpretation": "No se pudo cargar la librería arch. Instala dependencias para el módulo de volatilidad condicional."
        }

    specs = {
        "ARCH(1)": arch_model(r, vol="GARCH", p=1, q=0, dist="normal"),
        "GARCH(1,1)": arch_model(r, vol="GARCH", p=1, q=1, dist="normal"),
        "EGARCH(1,1)": arch_model(r, vol="EGARCH", p=1, q=1, dist="normal"),
    }

    rows = []
    best_fit = None
    best_name = None
    best_aic = np.inf

    for name, model in specs.items():
        try:
            fit = model.fit(disp="off", show_warning=False)
            rows.append({"Modelo": name, "LogLik": float(fit.loglikelihood), "AIC": float(fit.aic), "BIC": float(fit.bic)})
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_name = name
        except Exception:
            rows.append({"Modelo": name, "LogLik": None, "AIC": None, "BIC": None})

    residual_jb = None
    forecast = None
    std_resid = None
    cond_vol = None
    if best_fit is not None:
        try:
            std_resid = pd.Series(best_fit.std_resid).dropna()
            cond_vol = pd.Series(best_fit.conditional_volatility / 100).dropna()
            jb = stats.jarque_bera(std_resid)
            residual_jb = (float(jb[0]), float(jb[1]))
            fcast = best_fit.forecast(horizon=10)
            forecast = (np.sqrt(fcast.variance.iloc[-1].values) / 100).tolist()
        except Exception:
            pass

    interpretation = (
        f"El mejor modelo por AIC fue {best_name}. La comparación entre ARCH, GARCH y EGARCH sugiere heterocedasticidad condicional y permite pronosticar volatilidad futura."
        if best_name else
        "No fue posible seleccionar un modelo robusto."
    )

    return {
        "table": pd.DataFrame(rows).sort_values("AIC", na_position="last"),
        "forecast": forecast,
        "std_resid": std_resid,
        "cond_vol": cond_vol,
        "residual_jb": residual_jb,
        "interpretation": interpretation,
    }
