from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

def apply_custom_style():
    st.markdown("""
    <style>
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #06b6d4 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(2,6,23,.18);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: .4rem;
        margin-bottom: .8rem;
        color: #f8fafc;
    }
    .info-box {
        background: #e2e8f0;
        color: #0f172a;
        border-left: 6px solid #2563eb;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15,23,42,.08);
        line-height: 1.55;
    }
    .signal-card {
        padding: 1rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,.10);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f"""<div class="hero"><h1 style="margin:0;">{title}</h1><p style="margin:.35rem 0 0 0;font-size:1.02rem;">{subtitle}</p></div>""", unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def info_box(title, text):
    st.markdown(
        f"""
        <div style="
            background: #e2e8f0;
            color: #0f172a;
            border-left: 6px solid #2563eb;
            padding: 1rem 1.1rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            font-size: 0.98rem;
            line-height: 1.55;
        ">
            <strong style="color:#1e3a8a;">{title}:</strong>
            <span style="color:#0f172a;"> {text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def signal_card_html(ticker, sig):
    return f"""
    <div class="signal-card" style="background:{sig['color']};">
      <h3 style="margin-top:0;">{ticker}</h3>
      <p><strong>Señal:</strong> {sig['signal']}</p>
      <p><strong>Confianza:</strong> {sig['confidence']}%</p>
      <p><strong>Precio actual:</strong> {sig['current_price']:.2f}</p>
      <p style="font-size:.95rem;">{sig['interpretation']}</p>
      <ul>{''.join(f'<li>{r}</li>' for r in sig['reasons'][:4])}</ul>
    </div>
    """

def plot_price_technical(prices, pack):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Precio", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=pack["sma_short"].index, y=pack["sma_short"].values, name="SMA corta"))
    fig.add_trace(go.Scatter(x=pack["sma_long"].index, y=pack["sma_long"].values, name="SMA larga"))
    fig.add_trace(go.Scatter(x=pack["ema_short"].index, y=pack["ema_short"].values, name="EMA corta", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=pack["ema_long"].index, y=pack["ema_long"].values, name="EMA larga", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=pack["bb_upper"].index, y=pack["bb_upper"].values, name="Bollinger Sup.", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=pack["bb_lower"].index, y=pack["bb_lower"].values, name="Bollinger Inf.", line=dict(dash="dash")))
    fig.update_layout(template="plotly_white", height=520, title="Precio con medias y Bandas de Bollinger")
    return fig

def plot_rsi(rsi_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, name="RSI"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(template="plotly_white", height=300, title="RSI")
    return fig

def plot_macd(macd, signal, hist):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd.index, y=macd.values, name="MACD"))
    fig.add_trace(go.Scatter(x=signal.index, y=signal.values, name="Señal"))
    fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Histograma"))
    fig.update_layout(template="plotly_white", height=300, title="MACD")
    return fig

def plot_stochastic(k, d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k.index, y=k.values, name="%K"))
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="%D"))
    fig.add_hline(y=80, line_dash="dash", line_color="red")
    fig.add_hline(y=20, line_dash="dash", line_color="green")
    fig.update_layout(template="plotly_white", height=300, title="Oscilador Estocástico")
    return fig

def plot_histogram(returns):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=40, name="Rendimientos"))
    fig.update_layout(template="plotly_white", height=300, title="Histograma")
    return fig

def plot_boxplot(returns):
    fig = go.Figure()
    fig.add_trace(go.Box(y=returns.values, name="Rendimientos", boxmean=True))
    fig.update_layout(template="plotly_white", height=300, title="Boxplot")
    return fig

def plot_qq_proxy(returns):
    qq = stats.probplot(returns.dropna(), dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Datos"))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] + qq[1][1] * qq[0][0], mode="lines", name="Línea teórica"))
    fig.update_layout(template="plotly_white", height=300, title="Q-Q Plot")
    return fig

def plot_squared_returns(returns):
    sq = returns ** 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sq.index, y=sq.values, name="Rendimientos²"))
    fig.update_layout(template="plotly_white", height=300, title="Rendimientos al cuadrado")
    return fig

def plot_var_distribution(returns, rep):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=50, name="Rendimientos"))
    fig.add_vline(x=rep["var_hist"], line_dash="dash", line_color="red", annotation_text="VaR Histórico")
    fig.add_vline(x=rep["var_param"], line_dash="dash", line_color="orange", annotation_text="VaR Param.")
    fig.add_vline(x=rep["var_mc"], line_dash="dash", line_color="purple", annotation_text="VaR MC")
    fig.add_vline(x=rep["cvar"], line_dash="solid", line_color="black", annotation_text="CVaR")
    fig.update_layout(template="plotly_white", height=380, title="Distribución con VaR y CVaR")
    return fig

def plot_capm_scatter(df, ticker, beta):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["market"], y=df["asset"], mode="markers", name="Observaciones"))
    fig.add_trace(go.Scatter(x=df["market"], y=df["fit"], mode="lines", name=f"Regresión (β={beta:.2f})"))
    fig.update_layout(template="plotly_white", height=420, title=f"CAPM - {ticker}", xaxis_title="Rendimiento mercado", yaxis_title="Rendimiento activo")
    return fig

def plot_corr_heatmap(corr):
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1, text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(template="plotly_white", height=450, title="Matriz de correlación")
    return fig

def plot_efficient_frontier(simulated, max_sharpe, min_variance):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=simulated["risk"], y=simulated["return"], mode="markers", marker=dict(size=4, color=simulated["sharpe"], colorscale="Viridis", showscale=True), name="Portafolios simulados"))
    fig.add_trace(go.Scatter(x=[max_sharpe["risk"]], y=[max_sharpe["return"]], mode="markers", marker=dict(size=16, symbol="star"), name="Máx. Sharpe"))
    fig.add_trace(go.Scatter(x=[min_variance["risk"]], y=[min_variance["return"]], mode="markers", marker=dict(size=16, symbol="diamond"), name="Mín. Varianza"))
    fig.update_layout(template="plotly_white", height=500, title="Frontera eficiente", xaxis_title="Riesgo", yaxis_title="Retorno")
    return fig

def plot_cumulative_vs_benchmark(portfolio_returns, benchmark_returns):
    fig = go.Figure()
    if len(portfolio_returns):
        p = (1 + portfolio_returns).cumprod() * 100
        fig.add_trace(go.Scatter(x=p.index, y=p.values, name="Portafolio"))
    if benchmark_returns is not None and len(benchmark_returns):
        common = benchmark_returns.loc[benchmark_returns.index.intersection(portfolio_returns.index)]
        if len(common):
            b = (1 + common).cumprod() * 100
            fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Benchmark"))
    fig.update_layout(template="plotly_white", height=420, title="Rendimiento acumulado vs benchmark", yaxis_title="Base 100")
    return fig

def plot_garch_conditional_vol(cond_vol, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol.values, name="Volatilidad condicional"))
    if forecast:
        future_x = list(range(len(cond_vol), len(cond_vol) + len(forecast)))
        fig.add_trace(go.Scatter(x=future_x, y=forecast, name="Pronóstico", mode="lines+markers"))
    fig.update_layout(template="plotly_white", height=350, title="Volatilidad condicional y pronóstico")
    return fig

def plot_garch_residuals(std_resid):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=std_resid.values, nbinsx=40, name="Residuos estandarizados"))
    fig.update_layout(template="plotly_white", height=350, title="Distribución de residuos estandarizados")
    return fig

def format_pct_df(df, pct_cols=None):
    pct_cols = pct_cols or []
    out = df.copy()
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, (int, float, np.floating)) else x)
    return out
