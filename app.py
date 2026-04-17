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
from models.portfolio_model import (
    correlation_matrix, simulate_portfolios,
    optimize_target_return, portfolio_performance
)
from models.performance_model import benchmark_report
from models.ui_components import (
    apply_custom_style, hero, section_title, info_box, signal_card_html,
    plot_price_technical, plot_rsi, plot_macd, plot_stochastic,
    plot_histogram, plot_boxplot, plot_qq_proxy, plot_squared_returns,
    plot_var_distribution, plot_capm_scatter, plot_corr_heatmap,
    plot_efficient_frontier, plot_cumulative_vs_benchmark,
    plot_garch_conditional_vol, plot_garch_residuals, format_pct_df,
    plot_portfolio_weights_donut, render_weights_table, render_portfolio_kpis,
    COLORS_ASSETS,
)

st.set_page_config(
    page_title="Teoría del Riesgo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_style()
hero(
    "Teoría del Riesgo",
    "Análisis técnico, rendimientos, volatilidad condicional, CAPM, riesgo extremo y optimización de portafolios."
)

# ────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración general")
    tickers_text = st.text_input("Tickers (separados por coma)", value=",".join(DEFAULT_TICKERS))
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    start_date = st.date_input("Fecha inicial", value=date.today() - timedelta(days=365 * 2))
    confidence = st.selectbox("Nivel de confianza VaR", [0.90, 0.95, 0.99], index=1)
    portfolio_value = st.number_input(
        "Valor del portafolio (USD)",
        min_value=1000.0,
        value=100_000.0,
        step=1000.0
    )

    st.markdown("---")
    st.markdown("### 📐 Parámetros técnicos")
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

    st.markdown("---")
    rf_override_enabled = st.checkbox("Sobrescribir tasa libre de riesgo")
    rf_override = None
    if rf_override_enabled:
        rf_override = st.number_input(
            "Rf anual",
            min_value=0.0,
            max_value=0.40,
            value=0.03,
            step=0.005
        )

# ────────────────────────────────────────────────
#  CARGA DE DATOS
# ────────────────────────────────────────────────
with st.spinner("⏳ Descargando datos y construyendo análisis…"):
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
    st.error("No fue posible calcular rendimientos.")
    st.stop()

market_returns = pd.Series(dtype=float)
if isinstance(market, pd.Series) and not market.empty:
    market_returns = np.log(market / market.shift(1)).dropna()
elif "SPY" in log_returns.columns:
    market_returns = log_returns["SPY"].dropna()

optimizer_10k = simulate_portfolios(log_returns, macro["risk_free_rate"], n_portfolios=10_000)
optimizer_5k = simulate_portfolios(log_returns, macro["risk_free_rate"], n_portfolios=5_000)
benchmark = benchmark_report(
    optimizer_10k["max_sharpe_returns"],
    market_returns,
    macro["risk_free_rate"]
)

# ────────────────────────────────────────────────
#  TABS
# ────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Resumen",
    "📊 Técnico",
    "📈 Rendimientos",
    "🌀 GARCH",
    "📉 CAPM",
    "⚠️ VaR / CVaR",
    "🎯 Markowitz",
    "⚖️ Pesos Portafolio",
    "📡 Señales",
    "🌍 Macro & Benchmark",
])

# ── TAB 0: RESUMEN ───────────────────────────────
with tabs[0]:
    section_title("Resumen ejecutivo", "🏠")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos válidos", len(prices.columns))
    c2.metric("Observaciones", len(prices))
    c3.metric("Rf anual", f"{macro['risk_free_rate']:.2%}")
    c4.metric("Inflación aprox.", f"{macro['inflation_rate']:.2%}")

    a, b, c, d = st.columns(4)
    a.metric("Retorno Máx. Sharpe", f"{optimizer_10k['max_sharpe']['return']:.2%}")
    b.metric("Riesgo Máx. Sharpe", f"{optimizer_10k['max_sharpe']['risk']:.2%}")
    c.metric("Alpha de Jensen", f"{benchmark['alpha']:.2%}")
    d.metric("Information Ratio", f"{benchmark['information_ratio']:.2f}")

    info_box(
        "Interpretación",
        "Este tablero reúne varias aproximaciones al estudio del riesgo financiero. "
        "Primero se observa el comportamiento del precio y de los rendimientos; "
        "luego se examinan la volatilidad condicional, el riesgo sistemático y el riesgo extremo; "
        "por último, se estudia la construcción de portafolios eficientes."
    )

    st.plotly_chart(
        plot_cumulative_vs_benchmark(optimizer_10k["max_sharpe_returns"], market_returns),
        use_container_width=True,
        key="cum_bench_res"
    )

    info_box(
        "Lectura",
        "La comparación acumulada permite evaluar si el portafolio de máximo Sharpe supera o no al benchmark "
        "en términos de crecimiento del capital. Si la curva del portafolio se mantiene por encima, "
        "hay evidencia visual de un mejor desempeño relativo durante el periodo analizado."
    )

# ── TAB 1: TÉCNICO ────────────────────────────────
with tabs[1]:
    section_title("Análisis técnico e indicadores", "📊")
    ticker = st.selectbox("Activo", list(prices.columns), key="tech_sel")
    pack = technical_pack(
        prices[ticker].dropna(),
        sma_short=sma_short,
        sma_long=sma_long,
        ema_short=ema_short,
        ema_long=ema_long,
        rsi_period=rsi_period,
        bb_window=bb_window,
        bb_std=bb_std,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
    )

    st.plotly_chart(
        plot_price_technical(prices[ticker].dropna(), pack),
        use_container_width=True,
        key="price_tech"
    )

    info_box(
        "Interpretación",
        f"En {ticker}, el gráfico de precios permite identificar la tendencia dominante y contrastarla con señales de sobreextensión. "
        "Las medias móviles ayudan a distinguir la dirección del movimiento, mientras que las bandas de Bollinger permiten observar "
        "si el precio se está alejando de su comportamiento reciente."
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_rsi(pack["rsi"]), use_container_width=True, key="rsi_ch")
    c2.plotly_chart(
        plot_macd(pack["macd"], pack["macd_signal"], pack["macd_hist"]),
        use_container_width=True,
        key="macd_ch"
    )

    c3, c4 = st.columns(2)
    c3.plotly_chart(
        plot_stochastic(pack["stochastic_k"], pack["stochastic_d"]),
        use_container_width=True,
        key="stoch_ch"
    )
    with c4:
        st.markdown("##### Lectura de indicadores")
        interp = indicator_interpretations(
            pack,
            prices[ticker].dropna().iloc[-1],
            rsi_upper,
            rsi_lower
        )
        for k, v in interp.items():
            info_box(k, v)

# ── TAB 2: RENDIMIENTOS ───────────────────────────
with tabs[2]:
    section_title("Rendimientos y propiedades empíricas", "📈")
    ticker = st.selectbox("Activo ", list(log_returns.columns), key="ret_sel")
    rep = full_returns_report(log_returns[ticker].dropna())

    a, b, c, d = st.columns(4)
    a.metric("Media simple anual", f"{rep['simple_annual_mean']:.2%}")
    b.metric("Media log anual", f"{rep['log_annual_mean']:.2%}")
    c.metric("Volatilidad anual", f"{rep['annual_vol']:.2%}")
    d.metric("Curtosis", f"{rep['kurtosis']:.2f}")

    e, f = st.columns(2)
    e.plotly_chart(plot_histogram(rep["returns"]), use_container_width=True, key="hist_ch")
    f.plotly_chart(plot_boxplot(rep["returns"]), use_container_width=True, key="box_ch")

    g, h = st.columns(2)
    g.plotly_chart(plot_qq_proxy(rep["returns"]), use_container_width=True, key="qq_ch")
    h.plotly_chart(plot_squared_returns(rep["returns"]), use_container_width=True, key="sq_ch")

    st.subheader("Pruebas de normalidad")
    st.dataframe(pd.DataFrame([
        {"Prueba": "Jarque-Bera", "Estadístico": rep["jb_stat"], "p-valor": rep["jb_pvalue"]},
        {"Prueba": "Shapiro-Wilk", "Estadístico": rep["sw_stat"], "p-valor": rep["sw_pvalue"]},
    ]), use_container_width=True)

    st.subheader("Hechos estilizados")
    st.dataframe(pd.DataFrame([
        {"Hecho estilizado": "Colas pesadas", "Resultado": rep["facts"]["heavy_tails"]},
        {"Hecho estilizado": "Clustering de volatilidad", "Resultado": rep["facts"]["volatility_clustering"]},
        {"Hecho estilizado": "Eficiencia lineal aprox.", "Resultado": rep["facts"]["market_efficiency_proxy"]},
        {"Hecho estilizado": "Efecto apalancamiento", "Resultado": rep["facts"]["leverage_effect"]},
    ]), use_container_width=True)

    info_box(
        "Lectura",
        "El histograma y el boxplot muestran la forma de la distribución y ayudan a detectar asimetrías o valores extremos. "
        "El gráfico tipo Q-Q permite comparar los datos con una distribución normal, "
        "mientras que los rendimientos al cuadrado sirven para visualizar episodios de agrupamiento de volatilidad."
    )
    info_box("Interpretación", rep["interpretation"])

# ── TAB 3: GARCH ─────────────────────────────────
with tabs[3]:
    section_title("Modelos ARCH / GARCH", "🌀")
    ticker = st.selectbox("Activo  ", list(log_returns.columns), key="garch_sel")
    greport = fit_garch_suite(log_returns[ticker].dropna())
    st.dataframe(greport["table"], use_container_width=True)

    c1, c2 = st.columns(2)
    if greport["cond_vol"] is not None:
        c1.plotly_chart(
            plot_garch_conditional_vol(greport["cond_vol"], greport["forecast"]),
            use_container_width=True,
            key="garch_vol"
        )
    if greport["std_resid"] is not None:
        c2.plotly_chart(
            plot_garch_residuals(greport["std_resid"]),
            use_container_width=True,
            key="garch_res"
        )

    info_box("Diagnóstico", greport["interpretation"])
    info_box(
        "Lectura",
        "Si la volatilidad condicional presenta picos persistentes, se concluye que el riesgo no permanece constante en el tiempo. "
        "Esto justifica el uso de modelos GARCH, ya que una varianza fija podría subestimar el riesgo en periodos de tensión "
        "y sobrestimarlo en periodos relativamente tranquilos."
    )

    if greport["residual_jb"] is not None:
        st.write(
            f"**Jarque-Bera residuos:** estadístico={greport['residual_jb'][0]:.4f}, "
            f"p-valor={greport['residual_jb'][1]:.6f}"
        )

# ── TAB 4: CAPM ──────────────────────────────────
with tabs[4]:
    section_title("CAPM y riesgo sistemático", "📉")
    capm_rows = [
        capm_report(log_returns[t].dropna(), market_returns, macro["risk_free_rate"], t)
        for t in log_returns.columns
    ]
    capm_df = pd.DataFrame(capm_rows)

    st.dataframe(
        format_pct_df(
            capm_df[["ticker", "beta", "alpha_anual", "r_squared", "expected_return", "classification"]],
            pct_cols=["alpha_anual", "expected_return"]
        ),
        use_container_width=True
    )

    ticker = st.selectbox("Activo   ", list(log_returns.columns), key="capm_sel")
    selected = capm_report(log_returns[ticker].dropna(), market_returns, macro["risk_free_rate"], ticker)

    if selected["scatter_df"].empty:
        st.warning("No se pudo construir el gráfico CAPM. " + selected["interpretation"])
    else:
        st.plotly_chart(
            plot_capm_scatter(selected["scatter_df"], ticker, selected["beta"]),
            use_container_width=True,
            key="capm_scat"
        )

    info_box("Interpretación", selected["interpretation"])

    if pd.notna(selected["beta"]):
        if selected["beta"] > 1.2:
            capm_reading = (
                f"{ticker} muestra una sensibilidad alta frente al mercado. "
                "En términos prácticos, cuando el benchmark se mueve, este activo tiende a reaccionar con mayor intensidad."
            )
        elif selected["beta"] >= 0.8:
            capm_reading = (
                f"{ticker} presenta una sensibilidad cercana a la del mercado. "
                "Esto sugiere un comportamiento relativamente alineado con los movimientos del benchmark."
            )
        else:
            capm_reading = (
                f"{ticker} tiene una sensibilidad menor que la del mercado. "
                "Eso indica una exposición más defensiva frente al riesgo sistemático."
            )
        info_box("Lectura", capm_reading)

    info_box(
        "Fundamento",
        "La beta representa el riesgo sistemático, es decir, la porción del riesgo que permanece incluso con diversificación. "
        "Por eso el CAPM se utiliza para estimar el retorno exigido a un activo dada su exposición al mercado."
    )

# ── TAB 5: VaR / CVaR ────────────────────────────
with tabs[5]:
    section_title("Valor en Riesgo y CVaR", "⚠️")
    ticker = st.selectbox("Activo    ", list(log_returns.columns), key="var_sel")
    rrep = risk_report(log_returns[ticker].dropna(), portfolio_value, confidence)

    st.dataframe(
        format_pct_df(rrep["table"], pct_cols=["VaR %", "VaR anual %"]),
        use_container_width=True
    )

    st.plotly_chart(
        plot_var_distribution(log_returns[ticker].dropna(), rrep),
        use_container_width=True,
        key="var_dist"
    )

    info_box(
        "Lectura",
        "La distribución permite ubicar visualmente la zona de pérdidas extremas. "
        "El VaR marca un umbral de pérdida bajo un nivel de confianza dado, mientras que el CVaR resume la pérdida promedio "
        "en los escenarios que ya excedieron ese umbral."
    )
    info_box("Interpretación", rrep["interpretation"])

    st.write(
        f"**Backtesting de Kupiec:** violaciones esperadas={rrep['kupiec']['expected_violations']:.2f}, "
        f"reales={rrep['kupiec']['actual_violations']}, "
        f"p-valor={rrep['kupiec']['p_value']:.6f}"
    )

# ── TAB 6: MARKOWITZ ─────────────────────────────
with tabs[6]:
    section_title("Optimización de portafolio — Markowitz", "🎯")
    corr = correlation_matrix(log_returns)
    st.plotly_chart(plot_corr_heatmap(corr), use_container_width=True, key="corr_heat")

    st.plotly_chart(
        plot_efficient_frontier(
            optimizer_10k["simulated"],
            optimizer_10k["max_sharpe"],
            optimizer_10k["min_variance"]
        ),
        use_container_width=True,
        key="eff_front"
    )

    info_box(
        "Interpretación",
        "La nube de puntos representa portafolios factibles generados por simulación. "
        "La frontera eficiente reúne las combinaciones que ofrecen el mayor retorno esperado para cada nivel de riesgo. "
        "Dentro de ella, el portafolio de máximo Sharpe prioriza eficiencia riesgo-retorno, mientras que el de mínima varianza prioriza estabilidad."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⭐ Portafolio de máximo Sharpe")
        render_portfolio_kpis(optimizer_10k["max_sharpe"])
        render_weights_table(optimizer_10k["max_sharpe"]["weights"])

    with c2:
        st.subheader("💎 Portafolio de mínima varianza")
        render_portfolio_kpis(optimizer_10k["min_variance"])
        render_weights_table(optimizer_10k["min_variance"]["weights"])

    st.markdown("---")
    target = st.slider(
        "Rendimiento objetivo anual",
        0.0,
        0.50,
        0.15,
        0.01,
        format="%.0f%%"
    )
    target_port = optimize_target_return(log_returns, macro["risk_free_rate"], target)

    st.subheader("🎯 Portafolio eficiente para rendimiento objetivo")
    render_portfolio_kpis(target_port)
    info_box("Interpretación", target_port["interpretation"])
    info_box(
        "Lectura",
        "El portafolio objetivo muestra el costo de exigir un retorno específico. "
        "En general, a medida que se busca una rentabilidad más alta, también aumenta la volatilidad que el inversionista debe aceptar."
    )
    render_weights_table(target_port["weights"])

# ── TAB 7: PESOS DEL PORTAFOLIO ──────────────────
with tabs[7]:
    section_title("Pesos del portafolio — Configuración manual", "⚖️")

    info_box(
        "Interpretación",
        "Esta sección permite evaluar cómo cambia el perfil de riesgo y retorno cuando la asignación de pesos es decidida manualmente. "
        "Así se puede comparar una decisión discrecional con los portafolios óptimos obtenidos bajo el enfoque de Markowitz."
    )

    assets = list(log_returns.columns)
    n_assets = len(assets)
    mean_returns_a = log_returns.mean()
    cov_matrix_a = log_returns.cov()

    mode = st.radio(
        "Modo de asignación de pesos",
        ["Manual (sliders)", "Igual ponderación", "Copiar Máx. Sharpe", "Copiar Mín. Varianza"],
        horizontal=True,
    )

    if mode == "Igual ponderación":
        default_w = {a: round(1.0 / n_assets, 4) for a in assets}
    elif mode == "Copiar Máx. Sharpe":
        default_w = optimizer_10k["max_sharpe"]["weights"]
    elif mode == "Copiar Mín. Varianza":
        default_w = optimizer_10k["min_variance"]["weights"]
    else:
        default_w = {a: round(1.0 / n_assets, 4) for a in assets}

    # Detectar cambio de modo y actualizar sliders
    if "prev_weight_mode" not in st.session_state:
        st.session_state.prev_weight_mode = mode

    if mode != st.session_state.prev_weight_mode:
        for asset in assets:
            st.session_state[f"w_{asset}"] = float(default_w.get(asset, 1.0 / n_assets))
        st.session_state.prev_weight_mode = mode

    st.markdown("#### Asignación de pesos")
    st.caption("ℹ️ Los pesos se normalizan automáticamente para sumar 100%.")

    raw_weights = {}
    cols_sliders = st.columns(min(n_assets, 4))

    for i, asset in enumerate(assets):
        col = cols_sliders[i % len(cols_sliders)]
        with col:
            # Inicializar si aún no existe
            if f"w_{asset}" not in st.session_state:
                st.session_state[f"w_{asset}"] = float(default_w.get(asset, 1.0 / n_assets))

            raw_weights[asset] = st.slider(
                asset,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=f"w_{asset}",
                format="%.2f",
            )

    total_raw = sum(raw_weights.values())
    if total_raw == 0:
        manual_weights = {a: 1.0 / n_assets for a in assets}
    else:
        manual_weights = {a: v / total_raw for a, v in raw_weights.items()}

    w_arr = np.array([manual_weights[a] for a in assets])
    man_ret, man_risk, man_sharpe = portfolio_performance(
        w_arr,
        mean_returns_a,
        cov_matrix_a,
        macro["risk_free_rate"]
    )
    manual_port = {
        "return": man_ret,
        "risk": man_risk,
        "sharpe": man_sharpe,
        "weights": manual_weights
    }

    st.markdown("---")
    st.markdown("#### Resultados del portafolio")
    render_portfolio_kpis(manual_port)

    col_donut, col_table = st.columns([1, 1])
    with col_donut:
        st.plotly_chart(
            plot_portfolio_weights_donut(manual_weights, "Distribución de pesos"),
            use_container_width=True,
            key="donut_manual"
        )
    with col_table:
        st.markdown("<br>", unsafe_allow_html=True)
        render_weights_table(manual_weights)

    st.markdown("---")
    st.markdown("#### Comparación con los portafolios eficientes")

    comp_data = {
        "Portafolio": ["📐 Manual", "⭐ Máx. Sharpe", "💎 Mín. Varianza"],
        "Retorno esperado": [
            man_ret,
            optimizer_10k["max_sharpe"]["return"],
            optimizer_10k["min_variance"]["return"]
        ],
        "Riesgo (σ)": [
            man_risk,
            optimizer_10k["max_sharpe"]["risk"],
            optimizer_10k["min_variance"]["risk"]
        ],
        "Sharpe": [
            man_sharpe,
            optimizer_10k["max_sharpe"]["sharpe"],
            optimizer_10k["min_variance"]["sharpe"]
        ],
    }

    comp_df = pd.DataFrame(comp_data)
    comp_display = comp_df.copy()
    comp_display["Retorno esperado"] = comp_df["Retorno esperado"].apply(lambda x: f"{x:.2%}")
    comp_display["Riesgo (σ)"] = comp_df["Riesgo (σ)"].apply(lambda x: f"{x:.2%}")
    comp_display["Sharpe"] = comp_df["Sharpe"].apply(lambda x: f"{x:.2f}")
    st.dataframe(comp_display, use_container_width=True, hide_index=True)

    fig_comp = plot_efficient_frontier(
        optimizer_10k["simulated"],
        optimizer_10k["max_sharpe"],
        optimizer_10k["min_variance"],
    )

    fig_comp.add_trace(__import__("plotly.graph_objects", fromlist=["Scatter"]).Scatter(
        x=[man_risk],
        y=[man_ret],
        mode="markers+text",
        marker=dict(
            size=18,
            symbol="circle",
            color="#F85149",
            line=dict(color="#0D1117", width=2)
        ),
        name="Tu portafolio",
        text=["Tu portafolio"],
        textposition="top right",
        textfont=dict(color="#F85149", size=11),
    ))

    fig_comp.update_layout(title="Ubicación del portafolio en la frontera eficiente")
    st.plotly_chart(fig_comp, use_container_width=True, key="comp_front")

    info_box(
        "Lectura",
        "La posición del portafolio manual frente a la frontera eficiente muestra si la combinación elegida aprovecha bien el riesgo asumido. "
        "Si queda por debajo de la frontera, existe otra combinación que ofrecería un mejor retorno con un nivel similar de riesgo."
    )

    st.markdown("#### Comparación visual de distribuciones")
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.plotly_chart(
            plot_portfolio_weights_donut(manual_weights, "Portafolio manual"),
            use_container_width=True,
            key="donut_man2"
        )
    with cd2:
        st.plotly_chart(
            plot_portfolio_weights_donut(optimizer_10k["max_sharpe"]["weights"], "Máx. Sharpe"),
            use_container_width=True,
            key="donut_sharpe"
        )
    with cd3:
        st.plotly_chart(
            plot_portfolio_weights_donut(optimizer_10k["min_variance"]["weights"], "Mín. Varianza"),
            use_container_width=True,
            key="donut_minvar"
        )

# ── TAB 8: SEÑALES ────────────────────────────────
with tabs[8]:
    section_title("Señales y alertas de mercado", "📡")
    cols = st.columns(3)
    for i, t in enumerate(prices.columns):
        sig = signal_summary(
            prices[t].dropna(),
            rsi_upper=rsi_upper,
            rsi_lower=rsi_lower,
            sma_short=sma_short,
            sma_long=sma_long,
            bb_window=bb_window,
            bb_std=bb_std,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
        )
        with cols[i % 3]:
            st.markdown(signal_card_html(t, sig), unsafe_allow_html=True)

# ── TAB 9: MACRO & BENCHMARK ─────────────────────
with tabs[9]:
    section_title("Contexto macro y benchmark", "🌍")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rf anual", f"{macro['risk_free_rate']:.2%}")
    c2.metric("Inflación", f"{macro['inflation_rate']:.2%}")
    c3.metric("USD/COP", f"{macro['usd_cop']:,.0f}")
    c4.metric("Máx. Drawdown", f"{benchmark['max_drawdown']:.2%}")

    st.plotly_chart(
        plot_cumulative_vs_benchmark(optimizer_5k["max_sharpe_returns"], market_returns),
        use_container_width=True,
        key="cum_macro"
    )

    info_box(
        "Interpretación",
        "Esta comparación resume el desempeño relativo del portafolio frente al mercado. "
        "Si el portafolio obtiene mayor retorno con una volatilidad comparable, la estrategia muestra una relación riesgo-retorno más favorable. "
        "Si además el alpha es positivo, hay evidencia de generación de valor frente al benchmark."
    )

    perf_df = pd.DataFrame([
        {"Métrica": "Alpha de Jensen", "Valor": benchmark["alpha"]},
        {"Métrica": "Tracking Error", "Valor": benchmark["tracking_error"]},
        {"Métrica": "Information Ratio", "Valor": benchmark["information_ratio"]},
        {"Métrica": "Sharpe Portafolio", "Valor": benchmark["portfolio_sharpe"]},
        {"Métrica": "Sharpe Benchmark", "Valor": benchmark["benchmark_sharpe"]},
        {"Métrica": "Retorno acum. Portafolio", "Valor": benchmark["portfolio_cum_return"]},
        {"Métrica": "Retorno acum. Benchmark", "Valor": benchmark["benchmark_cum_return"]},
        {"Métrica": "Retorno anualizado Portafolio", "Valor": benchmark["portfolio_annual_return"]},
        {"Métrica": "Retorno anualizado Benchmark", "Valor": benchmark["benchmark_annual_return"]},
        {"Métrica": "Volatilidad Portafolio", "Valor": benchmark["portfolio_vol"]},
        {"Métrica": "Volatilidad Benchmark", "Valor": benchmark["benchmark_vol"]},
        {"Métrica": "Máximo Drawdown", "Valor": benchmark["max_drawdown"]},
    ])

    st.dataframe(format_pct_df(perf_df, pct_cols=["Valor"]), use_container_width=True)
    info_box("Interpretación", benchmark["interpretation"])