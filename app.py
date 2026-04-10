import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from models.data_api import DEFAULT_TICKERS, get_market_data, get_macro_data
from models.technical import technical_pack, signal_summary, indicator_interpretations
from models.returns_analysis import full_returns_report
from models.garch_model import fit_garch_suite
from models.capm_model import capm_report
from models.risk_model import risk_report
from models.portfolio_model import correlation_matrix, simulate_portfolios, optimize_target_return
from models.performance_model import benchmark_report
from models.ui_components import (
    apply_custom_style, hero, section_title, info_box, signal_card_html,
    plot_price_technical, plot_rsi, plot_macd, plot_stochastic,
    plot_histogram, plot_boxplot, plot_qq_proxy, plot_squared_returns,
    plot_var_distribution, plot_capm_scatter, plot_corr_heatmap,
    plot_efficient_frontier, plot_cumulative_vs_benchmark,
    plot_garch_conditional_vol, plot_garch_residuals, format_pct_df
)

st.set_page_config(
    page_title="Proyecto Integrador - Teoría del Riesgo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_style()
hero(
    "📈 Proyecto Integrador - Teoría del Riesgo",
    "Dashboard profesional con análisis técnico, rendimientos, GARCH, CAPM, VaR/CVaR, Markowitz, señales y benchmark."
)

with st.sidebar:
    st.header("Configuración")
    tickers_text = st.text_input("Tickers", value=",".join(DEFAULT_TICKERS))
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    start_date = st.date_input("Fecha inicial", value=date.today() - timedelta(days=365 * 2))
    confidence = st.selectbox("Nivel de confianza VaR", [0.90, 0.95, 0.99], index=1)
    portfolio_value = st.number_input("Valor del portafolio", min_value=1000.0, value=100000.0, step=1000.0)

    st.markdown("### Parámetros técnicos")
    sma_short = st.slider("SMA corta", 5, 50, 20)
    sma_long = st.slider("SMA larga", 20, 200, 50)
    ema_short = st.slider("EMA corta", 5, 50, 20)
    ema_long = st.slider("EMA larga", 20, 200, 50)
    bb_window = st.slider("Ventana Bollinger", 10, 40, 20)
    bb_std = st.slider("Desv. estándar Bollinger", 1, 3, 2)
    rsi_period = st.slider("Periodo RSI", 7, 30, 14)
    rsi_upper = st.slider("RSI sobrecompra", 60, 90, 70)
    rsi_lower = st.slider("RSI sobreventa", 10, 40, 30)
    stoch_k = st.slider("Periodo %K estocástico", 7, 30, 14)
    stoch_d = st.slider("Periodo %D estocástico", 2, 10, 3)

    rf_override_enabled = st.checkbox("Sobrescribir tasa libre de riesgo")
    rf_override = None
    if rf_override_enabled:
        rf_override = st.number_input("Rf anual", min_value=0.0, max_value=0.40, value=0.03, step=0.005)

with st.spinner("Descargando datos y construyendo análisis..."):
    prices, market = get_market_data(tickers, str(start_date))
    macro = get_macro_data()
    if rf_override is not None:
        macro["risk_free_rate"] = rf_override

if prices.empty:
    st.error("No fue posible descargar precios. Verifica internet, tickers o dependencias.")
    st.info("Prueba con: AAPL, MSFT, AMZN, GOOGL, SPY")
    st.stop()

returns = prices.pct_change().dropna()
log_returns = np.log(prices / prices.shift(1)).dropna()
if returns.empty or log_returns.empty:
    st.error("No fue posible calcular rendimientos con los datos descargados.")
    st.stop()

market_returns = pd.Series(dtype=float)
if isinstance(market, pd.Series) and not market.empty:
    market_returns = np.log(market / market.shift(1)).dropna()

optimizer_10k = simulate_portfolios(log_returns, macro["risk_free_rate"], n_portfolios=10000)
optimizer_5k = simulate_portfolios(log_returns, macro["risk_free_rate"], n_portfolios=5000)
benchmark = benchmark_report(optimizer_10k["max_sharpe_returns"], market_returns, macro["risk_free_rate"])

tabs = st.tabs([
    "Resumen ejecutivo",
    "Análisis técnico",
    "Rendimientos",
    "ARCH / GARCH",
    "CAPM",
    "VaR / CVaR",
    "Markowitz",
    "Señales",
    "Macro y benchmark",
])

with tabs[0]:
    section_title("Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos válidos", len(prices.columns))
    c2.metric("Observaciones", len(prices))
    c3.metric("Rf anual", f"{macro['risk_free_rate']:.2%}")
    c4.metric("Inflación aprox.", f"{macro['inflation_rate']:.2%}")

    a, b, c, d = st.columns(4)
    a.metric("Retorno máx. Sharpe", f"{optimizer_10k['max_sharpe']['return']:.2%}")
    b.metric("Riesgo máx. Sharpe", f"{optimizer_10k['max_sharpe']['risk']:.2%}")
    c.metric("Alpha de Jensen", f"{benchmark['alpha']:.2%}")
    d.metric("Information Ratio", f"{benchmark['information_ratio']:.2f}")

    info_box(
        "Lectura sugerida",
        "Este tablero integra análisis técnico, distribución de rendimientos, riesgo extremo, volatilidad condicional, CAPM y optimización. La secuencia sugerida es: contexto general → señales → riesgo y retorno → asignación óptima."
    )

    st.plotly_chart(
        plot_cumulative_vs_benchmark(optimizer_10k["max_sharpe_returns"], market_returns),
        use_container_width=True,
        key="cum_benchmark_resumen_final"
    )

with tabs[1]:
    section_title("Análisis técnico e indicadores")
    ticker = st.selectbox("Activo", list(prices.columns), key="tech_select_final")
    pack = technical_pack(
        prices[ticker].dropna(),
        sma_short=sma_short, sma_long=sma_long,
        ema_short=ema_short, ema_long=ema_long,
        rsi_period=rsi_period, bb_window=bb_window,
        bb_std=bb_std, stoch_k=stoch_k, stoch_d=stoch_d
    )

    st.plotly_chart(
        plot_price_technical(prices[ticker].dropna(), pack),
        use_container_width=True,
        key="price_technical_chart_final"
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_rsi(pack["rsi"]), use_container_width=True, key="rsi_chart_final")
    c2.plotly_chart(plot_macd(pack["macd"], pack["macd_signal"], pack["macd_hist"]), use_container_width=True, key="macd_chart_final")

    c3, c4 = st.columns(2)
    c3.plotly_chart(plot_stochastic(pack["stochastic_k"], pack["stochastic_d"]), use_container_width=True, key="stochastic_chart_final")
    with c4:
        st.markdown("### Panel interpretativo")
        interp = indicator_interpretations(pack, prices[ticker].dropna().iloc[-1], rsi_upper, rsi_lower)
        for k, v in interp.items():
            info_box(k, v)

with tabs[2]:
    section_title("Rendimientos y propiedades empíricas")
    ticker = st.selectbox("Activo ", list(log_returns.columns), key="returns_select_final")
    rep = full_returns_report(log_returns[ticker].dropna())

    a, b, c, d = st.columns(4)
    a.metric("Media simple anual", f"{rep['simple_annual_mean']:.2%}")
    b.metric("Media log anual", f"{rep['log_annual_mean']:.2%}")
    c.metric("Volatilidad anual", f"{rep['annual_vol']:.2%}")
    d.metric("Curtosis", f"{rep['kurtosis']:.2f}")

    e, f = st.columns(2)
    e.plotly_chart(plot_histogram(rep["returns"]), use_container_width=True, key="histogram_returns_chart_final")
    f.plotly_chart(plot_boxplot(rep["returns"]), use_container_width=True, key="boxplot_returns_chart_final")

    g, h = st.columns(2)
    g.plotly_chart(plot_qq_proxy(rep["returns"]), use_container_width=True, key="qqplot_returns_chart_final")
    h.plotly_chart(plot_squared_returns(rep["returns"]), use_container_width=True, key="squared_returns_chart_final")

    st.subheader("Pruebas de normalidad")
    st.dataframe(pd.DataFrame([
        {"Prueba": "Jarque-Bera", "Estadístico": rep["jb_stat"], "p-valor": rep["jb_pvalue"]},
        {"Prueba": "Shapiro-Wilk", "Estadístico": rep["sw_stat"], "p-valor": rep["sw_pvalue"]},
    ]), use_container_width=True)

    st.subheader("Hechos estilizados")
    facts_df = pd.DataFrame([
        {"Hecho estilizado": "Colas pesadas", "Resultado": rep["facts"]["heavy_tails"]},
        {"Hecho estilizado": "Clustering de volatilidad", "Resultado": rep["facts"]["volatility_clustering"]},
        {"Hecho estilizado": "Eficiencia lineal aproximada", "Resultado": rep["facts"]["market_efficiency_proxy"]},
        {"Hecho estilizado": "Efecto apalancamiento", "Resultado": rep["facts"]["leverage_effect"]},
    ])
    st.dataframe(facts_df, use_container_width=True)
    info_box("Interpretación", rep["interpretation"])

with tabs[3]:
    section_title("Modelos ARCH / GARCH")
    ticker = st.selectbox("Activo  ", list(log_returns.columns), key="garch_select_final")
    greport = fit_garch_suite(log_returns[ticker].dropna())
    st.dataframe(greport["table"], use_container_width=True)

    c1, c2 = st.columns(2)
    if greport["cond_vol"] is not None:
        c1.plotly_chart(plot_garch_conditional_vol(greport["cond_vol"], greport["forecast"]), use_container_width=True, key="garch_cond_vol_chart")
    if greport["std_resid"] is not None:
        c2.plotly_chart(plot_garch_residuals(greport["std_resid"]), use_container_width=True, key="garch_resid_chart")

    info_box("Diagnóstico", greport["interpretation"])
    if greport["residual_jb"] is not None:
        st.write(f"**Jarque-Bera sobre residuos estandarizados:** estadístico={greport['residual_jb'][0]:.4f}, p-valor={greport['residual_jb'][1]:.6f}")

with tabs[4]:
    section_title("CAPM y riesgo sistemático")
    capm_rows = [capm_report(log_returns[t].dropna(), market_returns, macro["risk_free_rate"], t) for t in log_returns.columns]
    capm_df = pd.DataFrame(capm_rows)
    st.dataframe(format_pct_df(capm_df[["ticker", "beta", "alpha_anual", "r_squared", "expected_return", "classification"]], pct_cols=["alpha_anual", "expected_return"]), use_container_width=True)

    ticker = st.selectbox("Activo   ", list(log_returns.columns), key="capm_select_final")
    selected = capm_report(log_returns[ticker].dropna(), market_returns, macro["risk_free_rate"], ticker)
    st.plotly_chart(plot_capm_scatter(selected["scatter_df"], ticker, selected["beta"]), use_container_width=True, key="capm_scatter_chart_final")
    info_box("Interpretación", selected["interpretation"])
    info_box("Marco teórico", "La beta mide riesgo sistemático, es decir, la porción del riesgo que no puede eliminarse mediante diversificación. El riesgo no sistemático se reduce al construir portafolios bien diversificados.")

with tabs[5]:
    section_title("Valor en Riesgo y CVaR")
    ticker = st.selectbox("Activo    ", list(log_returns.columns), key="risk_select_final")
    rrep = risk_report(log_returns[ticker].dropna(), portfolio_value, confidence)
    st.dataframe(format_pct_df(rrep["table"], pct_cols=["VaR %", "VaR anual %"]), use_container_width=True)
    st.plotly_chart(plot_var_distribution(log_returns[ticker].dropna(), rrep), use_container_width=True, key="var_distribution_chart_final")
    info_box("Interpretación", rrep["interpretation"])
    st.write(f"**Backtesting de Kupiec:** violaciones esperadas={rrep['kupiec']['expected_violations']:.2f}, reales={rrep['kupiec']['actual_violations']}, p-valor={rrep['kupiec']['p_value']:.6f}")

with tabs[6]:
    section_title("Optimización de portafolio — Markowitz")
    corr = correlation_matrix(log_returns)
    st.plotly_chart(plot_corr_heatmap(corr), use_container_width=True, key="corr_heatmap_chart_final")
    st.plotly_chart(plot_efficient_frontier(optimizer_10k["simulated"], optimizer_10k["max_sharpe"], optimizer_10k["min_variance"]), use_container_width=True, key="efficient_frontier_chart_final")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Portafolio de máximo Sharpe")
        st.dataframe(format_pct_df(pd.DataFrame(list(optimizer_10k["max_sharpe"]["weights"].items()), columns=["Activo", "Peso"]), pct_cols=["Peso"]), use_container_width=True)
        st.write(f"Retorno esperado: {optimizer_10k['max_sharpe']['return']:.2%} | Riesgo: {optimizer_10k['max_sharpe']['risk']:.2%} | Sharpe: {optimizer_10k['max_sharpe']['sharpe']:.2f}")
    with c2:
        st.subheader("Portafolio de mínima varianza")
        st.dataframe(format_pct_df(pd.DataFrame(list(optimizer_10k["min_variance"]["weights"].items()), columns=["Activo", "Peso"]), pct_cols=["Peso"]), use_container_width=True)
        st.write(f"Retorno esperado: {optimizer_10k['min_variance']['return']:.2%} | Riesgo: {optimizer_10k['min_variance']['risk']:.2%} | Sharpe: {optimizer_10k['min_variance']['sharpe']:.2f}")

    target = st.slider("Rendimiento objetivo anual", min_value=0.0, max_value=0.50, value=0.15, step=0.01)
    target_port = optimize_target_return(log_returns, macro["risk_free_rate"], target)
    st.subheader("Portafolio eficiente para rendimiento objetivo")
    st.write(target_port["interpretation"])
    st.dataframe(format_pct_df(pd.DataFrame(list(target_port["weights"].items()), columns=["Activo", "Peso"]), pct_cols=["Peso"]), use_container_width=True)

with tabs[7]:
    section_title("Señales y alertas")
    cols = st.columns(3)
    items = list(prices.columns)
    for i, t in enumerate(items):
        sig = signal_summary(
            prices[t].dropna(),
            rsi_upper=rsi_upper, rsi_lower=rsi_lower,
            sma_short=sma_short, sma_long=sma_long,
            bb_window=bb_window, bb_std=bb_std,
            stoch_k=stoch_k, stoch_d=stoch_d
        )
        with cols[i % 3]:
            st.markdown(signal_card_html(t, sig), unsafe_allow_html=True)

with tabs[8]:
    section_title("Contexto macro y benchmark")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rf", f"{macro['risk_free_rate']:.2%}")
    c2.metric("Inflación", f"{macro['inflation_rate']:.2%}")
    c3.metric("USD/COP", f"{macro['usd_cop']:,.0f}")
    c4.metric("Máx. Drawdown", f"{benchmark['max_drawdown']:.2%}")

    st.plotly_chart(plot_cumulative_vs_benchmark(optimizer_5k["max_sharpe_returns"], market_returns), use_container_width=True, key="cum_benchmark_macro_final")

    perf_df = pd.DataFrame([
        {"Métrica": "Alpha de Jensen", "Valor": benchmark["alpha"]},
        {"Métrica": "Tracking Error", "Valor": benchmark["tracking_error"]},
        {"Métrica": "Information Ratio", "Valor": benchmark["information_ratio"]},
        {"Métrica": "Sharpe Portafolio", "Valor": benchmark["portfolio_sharpe"]},
        {"Métrica": "Sharpe Benchmark", "Valor": benchmark["benchmark_sharpe"]},
        {"Métrica": "Retorno acumulado Portafolio", "Valor": benchmark["portfolio_cum_return"]},
        {"Métrica": "Retorno acumulado Benchmark", "Valor": benchmark["benchmark_cum_return"]},
        {"Métrica": "Retorno anualizado Portafolio", "Valor": benchmark["portfolio_annual_return"]},
        {"Métrica": "Retorno anualizado Benchmark", "Valor": benchmark["benchmark_annual_return"]},
        {"Métrica": "Volatilidad Portafolio", "Valor": benchmark["portfolio_vol"]},
        {"Métrica": "Volatilidad Benchmark", "Valor": benchmark["benchmark_vol"]},
        {"Métrica": "Máximo Drawdown", "Valor": benchmark["max_drawdown"]},
    ])
    st.dataframe(format_pct_df(perf_df, pct_cols=["Valor"]), use_container_width=True)
    info_box("Interpretación", benchmark["interpretation"])
