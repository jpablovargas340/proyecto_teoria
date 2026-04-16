from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ─────────────────────────────────────────────────────────────
#  PALETA GLOBAL  (dark-finance premium)
# ─────────────────────────────────────────────────────────────
C_BG         = "#0D1117"
C_SURFACE    = "#161B22"
C_SURFACE2   = "#1C2430"
C_BORDER     = "#30363D"
C_ACCENT     = "#58A6FF"
C_ACCENT2    = "#3FB950"
C_WARN       = "#D29922"
C_DANGER     = "#F85149"
C_TEXT       = "#E6EDF3"
C_TEXT_MUTED = "#8B949E"

COLORS_ASSETS = [
    "#58A6FF","#3FB950","#F78166","#D2A8FF",
    "#FFA657","#79C0FF","#56D364","#FF7B72",
    "#BC8CFF","#FFB86C","#8BE9FD","#50FA7B",
]

CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor=C_SURFACE,
    plot_bgcolor=C_SURFACE,
    font=dict(family="Inter, Segoe UI, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=40, r=20, t=45, b=40),
)

# ─────────────────────────────────────────────────────────────
#  ESTILOS GLOBALES
# ─────────────────────────────────────────────────────────────
def apply_custom_style():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif !important; }}
    .stApp {{ background: {C_BG}; color: {C_TEXT}; }}
    [data-testid="stSidebar"] {{ background: {C_SURFACE} !important; border-right: 1px solid {C_BORDER}; }}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {{ color: {C_TEXT} !important; }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {C_SURFACE}; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid {C_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {C_TEXT_MUTED}; border-radius: 8px;
        font-weight: 500; font-size: 0.84rem; padding: 8px 14px;
    }}
    .stTabs [aria-selected="true"] {{ background: {C_ACCENT} !important; color: #0D1117 !important; font-weight: 700; }}
    [data-testid="stMetric"] {{
        background: {C_SURFACE2}; border: 1px solid {C_BORDER}; border-radius: 12px; padding: 18px 20px;
    }}
    [data-testid="stMetric"] label {{
        color: {C_TEXT_MUTED} !important; font-size: 0.74rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.07em;
    }}
    [data-testid="stMetricValue"] {{ color: {C_TEXT} !important; font-size: 1.6rem !important; font-weight: 700 !important; }}
    [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {C_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 4px; }}
    .hero-banner {{
        background: linear-gradient(135deg, #0D1117 0%, #1B2A4A 50%, #0E3460 100%);
        border: 1px solid {C_BORDER}; border-radius: 20px;
        padding: 2rem 2.2rem; margin-bottom: 1.4rem;
        box-shadow: 0 20px 60px rgba(0,0,0,.5);
    }}
    .hero-badge {{
        display: inline-block; background: rgba(88,166,255,0.15); color: {C_ACCENT};
        border: 1px solid rgba(88,166,255,0.3); padding: 4px 12px; border-radius: 20px;
        font-size: 0.74rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .1em; margin-bottom: .8rem;
    }}
    .section-title-v2 {{
        display: flex; align-items: center; gap: 10px;
        font-size: 1.25rem; font-weight: 700; color: {C_TEXT};
        border-bottom: 2px solid {C_BORDER}; padding-bottom: 10px; margin: 1rem 0 1.2rem 0;
    }}
    .dot {{ width: 10px; height: 10px; background: {C_ACCENT}; border-radius: 50%; box-shadow: 0 0 8px {C_ACCENT}; }}
    .info-card {{
        background: {C_SURFACE2}; border: 1px solid {C_BORDER}; border-left: 4px solid {C_ACCENT};
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
        font-size: 0.9rem; line-height: 1.6; color: {C_TEXT};
    }}
    .info-card .info-title {{
        font-weight: 700; color: {C_ACCENT}; margin-bottom: 4px;
        font-size: 0.76rem; text-transform: uppercase; letter-spacing: .05em;
    }}
    .signal-card {{
        background: {C_SURFACE2}; border: 1px solid {C_BORDER}; border-radius: 16px;
        padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,.2);
    }}
    .kpi-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
    .kpi-badge {{
        display: inline-flex; align-items: center; gap: 4px;
        padding: 5px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    }}
    .weight-bar-outer {{
        background: {C_BORDER}; border-radius: 4px; height: 7px; width: 100%; overflow: hidden; margin: 4px 0;
    }}
    .weight-bar-inner {{ height: 7px; border-radius: 4px; }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────
def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge">📊 Proyecto Integrador · Teoría del Riesgo</div>
      <h1 style="margin:0;font-size:1.85rem;font-weight:800;color:{C_TEXT};letter-spacing:-.5px;">{title}</h1>
      <p style="margin:.5rem 0 0 0;font-size:.95rem;color:{C_TEXT_MUTED};line-height:1.5;">{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def section_title(text, icon=""):
    st.markdown(f"""
    <div class="section-title-v2">
      <div class="dot"></div>
      <span>{icon} {text}</span>
    </div>""", unsafe_allow_html=True)


def info_box(title, text):
    st.markdown(f"""
    <div class="info-card">
      <div class="info-title">{title}</div>
      <div>{text}</div>
    </div>""", unsafe_allow_html=True)


def signal_card_html(ticker, sig):
    sig_lower = sig["signal"].lower()
    if "compra" in sig_lower:
        accent, bg, label = C_ACCENT2, "rgba(63,185,80,.12)", "↑ COMPRA"
    elif "venta" in sig_lower:
        accent, bg, label = C_DANGER, "rgba(248,81,73,.12)", "↓ VENTA"
    else:
        accent, bg, label = C_WARN, "rgba(210,153,34,.12)", "→ NEUTRAL"
    reasons_html = "".join(
        f'<li style="margin-bottom:3px;font-size:.82rem;color:{C_TEXT_MUTED};">{r}</li>'
        for r in sig["reasons"][:4]
    )
    return f"""
    <div class="signal-card" style="border-left:4px solid {accent};">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:1.1rem;font-weight:700;color:{C_TEXT};">{ticker}</span>
        <span style="background:{bg};color:{accent};border:1px solid {accent}44;
                     padding:3px 10px;border-radius:20px;font-size:.77rem;font-weight:700;">{label}</span>
      </div>
      <div style="display:flex;gap:16px;margin-bottom:8px;">
        <div>
          <div style="font-size:.68rem;color:{C_TEXT_MUTED};text-transform:uppercase;">Confianza</div>
          <div style="font-weight:700;color:{accent};">{sig['confidence']}%</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:{C_TEXT_MUTED};text-transform:uppercase;">Precio</div>
          <div style="font-weight:700;color:{C_TEXT};">${sig['current_price']:.2f}</div>
        </div>
      </div>
      <p style="font-size:.84rem;color:{C_TEXT_MUTED};margin:6px 0;">{sig['interpretation']}</p>
      <ul style="padding-left:1.2rem;margin:0;">{reasons_html}</ul>
    </div>"""


# ─────────────────────────────────────────────────────────────
#  VISUALIZACIÓN DE PESOS
# ─────────────────────────────────────────────────────────────
def plot_portfolio_weights_donut(weights: dict, title: str = "Distribución de pesos") -> go.Figure:
    labels = list(weights.keys())
    values = [v * 100 for v in weights.values()]
    colors = COLORS_ASSETS[:len(labels)]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color=C_BG, width=3)),
        textinfo="label+percent",
        textfont=dict(size=12, color=C_TEXT),
        hovertemplate="<b>%{label}</b><br>Peso: %{value:.1f}%<extra></extra>",
        sort=False,
    )])
    fig.add_annotation(
        text=f"<b>{sum(weights.values())*100:.0f}%</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=C_TEXT), align="center",
    )
    fig.update_layout(
        **CHART_THEME, height=370,
        title=dict(text=title, font=dict(size=13, weight="bold")),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)),
    )
    return fig


def render_weights_table(weights: dict, colors=None):
    colors = colors or COLORS_ASSETS
    rows = []
    for i, (asset, w) in enumerate(weights.items()):
        color = colors[i % len(colors)]
        pct = w * 100
        rows.append(f"""
        <tr style="border-bottom:1px solid {C_BORDER};">
          <td style="padding:10px 14px;font-weight:600;color:{C_TEXT};">
            <span style="display:inline-block;width:9px;height:9px;background:{color};
                         border-radius:50%;margin-right:8px;"></span>{asset}
          </td>
          <td style="padding:10px 14px;width:52%;">
            <div class="weight-bar-outer">
              <div class="weight-bar-inner" style="width:{max(pct,1)}%;background:{color};"></div>
            </div>
          </td>
          <td style="padding:10px 14px;text-align:right;font-weight:700;color:{color};">{pct:.1f}%</td>
        </tr>""")
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:{C_SURFACE2};
                  border-radius:12px;overflow:hidden;border:1px solid {C_BORDER};">
      <thead>
        <tr style="background:{C_BG};">
          <th style="padding:10px 14px;text-align:left;font-size:.72rem;color:{C_TEXT_MUTED};
                     text-transform:uppercase;letter-spacing:.06em;">Activo</th>
          <th style="padding:10px 14px;font-size:.72rem;color:{C_TEXT_MUTED};
                     text-transform:uppercase;letter-spacing:.06em;">Distribución</th>
          <th style="padding:10px 14px;text-align:right;font-size:.72rem;color:{C_TEXT_MUTED};
                     text-transform:uppercase;letter-spacing:.06em;">Peso</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>""", unsafe_allow_html=True)


def render_portfolio_kpis(port: dict):
    ret_color   = C_ACCENT2 if port["return"] >= 0 else C_DANGER
    sharpe_color = C_ACCENT2 if port["sharpe"] >= 1 else C_WARN
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-badge" style="background:rgba(63,185,80,.12);color:{ret_color};border:1px solid {ret_color}44;">
        📈 Retorno: {port['return']:.2%}
      </div>
      <div class="kpi-badge" style="background:rgba(248,81,73,.12);color:{C_WARN};border:1px solid {C_WARN}44;">
        ⚡ Riesgo: {port['risk']:.2%}
      </div>
      <div class="kpi-badge" style="background:rgba(88,166,255,.12);color:{sharpe_color};border:1px solid {sharpe_color}44;">
        🎯 Sharpe: {port['sharpe']:.2f}
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────────────────────
def plot_price_technical(prices, pack):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Precio",
                             line=dict(width=2.5, color=C_ACCENT),
                             fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"))
    fig.add_trace(go.Scatter(x=pack["sma_short"].index, y=pack["sma_short"].values,
                             name="SMA corta", line=dict(width=1.5, color=C_ACCENT2, dash="dot")))
    fig.add_trace(go.Scatter(x=pack["sma_long"].index, y=pack["sma_long"].values,
                             name="SMA larga", line=dict(width=1.5, color=C_WARN)))
    fig.add_trace(go.Scatter(x=pack["ema_short"].index, y=pack["ema_short"].values,
                             name="EMA corta", line=dict(width=1.2, color="#D2A8FF", dash="dash")))
    fig.add_trace(go.Scatter(x=pack["ema_long"].index, y=pack["ema_long"].values,
                             name="EMA larga", line=dict(width=1.2, color="#FFA657", dash="dash")))
    fig.add_trace(go.Scatter(x=pack["bb_upper"].index, y=pack["bb_upper"].values,
                             name="BB Sup.", line=dict(dash="dot", width=1, color=C_DANGER)))
    fig.add_trace(go.Scatter(x=pack["bb_lower"].index, y=pack["bb_lower"].values,
                             name="BB Inf.", line=dict(dash="dot", width=1, color=C_ACCENT2),
                             fill="tonexty", fillcolor="rgba(88,166,255,0.04)"))
    fig.update_layout(**CHART_THEME, height=520,
                      title=dict(text="Precio · Medias Móviles · Bandas de Bollinger", font=dict(size=13)))
    return fig


def plot_rsi(rsi_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, name="RSI",
                             line=dict(color=C_ACCENT, width=2),
                             fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"))
    fig.add_hline(y=70, line_dash="dash", line_color=C_DANGER,
                  annotation_text="Sobrecompra", annotation_font_color=C_DANGER)
    fig.add_hline(y=30, line_dash="dash", line_color=C_ACCENT2,
                  annotation_text="Sobreventa", annotation_font_color=C_ACCENT2)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(63,185,80,0.05)", line_width=0)
    fig.update_layout(**CHART_THEME, height=280, title="RSI — Índice de Fuerza Relativa")
    return fig


def plot_macd(macd, signal, hist):
    colors = [C_ACCENT2 if v >= 0 else C_DANGER for v in hist.values]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd.index, y=macd.values, name="MACD",
                             line=dict(color=C_ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=signal.index, y=signal.values, name="Señal",
                             line=dict(color=C_WARN, width=1.5)))
    fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Histograma",
                         marker_color=colors, opacity=0.7))
    fig.update_layout(**CHART_THEME, height=280, title="MACD · Convergencia / Divergencia")
    return fig


def plot_stochastic(k, d):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k.index, y=k.values, name="%K",
                             line=dict(color=C_ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="%D",
                             line=dict(color=C_WARN, width=1.5, dash="dot")))
    fig.add_hline(y=80, line_dash="dash", line_color=C_DANGER)
    fig.add_hline(y=20, line_dash="dash", line_color=C_ACCENT2)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(63,185,80,0.05)", line_width=0)
    fig.update_layout(**CHART_THEME, height=280, title="Oscilador Estocástico %K / %D")
    return fig


def plot_histogram(returns):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=45, name="Rendimientos",
                               marker_color=C_ACCENT, opacity=0.75))
    fig.update_layout(**CHART_THEME, height=300, title="Histograma de rendimientos")
    return fig


def plot_boxplot(returns):
    fig = go.Figure()
    fig.add_trace(go.Box(y=returns.values, name="Rendimientos", boxmean=True,
                         marker_color=C_ACCENT, line_color=C_ACCENT,
                         fillcolor="rgba(88,166,255,0.15)"))
    fig.update_layout(**CHART_THEME, height=300, title="Distribución — Boxplot")
    return fig


def plot_qq_proxy(returns):
    qq = stats.probplot(returns.dropna(), dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Datos",
                             marker=dict(color=C_ACCENT, size=4, opacity=0.7)))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] + qq[1][1] * qq[0][0],
                             mode="lines", name="Normal teórica",
                             line=dict(color=C_DANGER, width=1.8)))
    fig.update_layout(**CHART_THEME, height=300, title="Q-Q Plot vs Normal",
                      xaxis_title="Cuantiles teóricos", yaxis_title="Cuantiles muestra")
    return fig


def plot_squared_returns(returns):
    sq = returns ** 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sq.index, y=sq.values, name="r²",
                             line=dict(color=C_WARN, width=1.5),
                             fill="tozeroy", fillcolor="rgba(210,153,34,0.06)"))
    fig.update_layout(**CHART_THEME, height=300, title="Rendimientos² — Clustering de volatilidad")
    return fig


def plot_var_distribution(returns, rep):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=55, name="Rendimientos",
                               marker_color=C_ACCENT, opacity=0.6))
    for val, color, label in [
        (rep["var_hist"],  C_DANGER,   "VaR Histórico"),
        (rep["var_param"], C_WARN,     "VaR Paramétrico"),
        (rep["var_mc"],    "#D2A8FF",  "VaR MC"),
        (rep["cvar"],      C_DANGER,   "CVaR"),
    ]:
        fig.add_vline(x=val,
                      line_dash="dash" if label != "CVaR" else "solid",
                      line_color=color, line_width=2,
                      annotation_text=label,
                      annotation_font_color=color, annotation_font_size=11)
    fig.update_layout(**CHART_THEME, height=400,
                      title="Distribución de rendimientos · VaR y CVaR")
    return fig


def plot_capm_scatter(df, ticker, beta):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["market"], y=df["asset"], mode="markers",
                             name="Observaciones",
                             marker=dict(color=C_ACCENT, size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=df["market"], y=df["fit"], mode="lines",
                             name=f"Regresión (β={beta:.2f})",
                             line=dict(color=C_DANGER, width=2.5)))
    fig.update_layout(**CHART_THEME, height=420,
                      title=f"CAPM — {ticker}",
                      xaxis_title="Rendimiento mercado (SPY)",
                      yaxis_title=f"Rendimiento {ticker}")
    return fig


def plot_corr_heatmap(corr):
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, C_DANGER], [0.5, C_SURFACE2], [1, C_ACCENT]],
        zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(size=11), hoverongaps=False,
    ))
    fig.update_layout(**CHART_THEME, height=460, title="Matriz de correlación — Log rendimientos")
    return fig


def plot_efficient_frontier(simulated, max_sharpe, min_variance):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=simulated["risk"], y=simulated["return"], mode="markers",
        marker=dict(size=3, color=simulated["sharpe"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe", tickfont=dict(color=C_TEXT)),
                    opacity=0.55),
        name="Portafolios simulados",
        hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[max_sharpe["risk"]], y=[max_sharpe["return"]], mode="markers+text",
        marker=dict(size=18, symbol="star", color=C_WARN, line=dict(color=C_BG, width=1.5)),
        name="Máx. Sharpe", text=["Máx. Sharpe"],
        textposition="top right", textfont=dict(color=C_WARN, size=11),
    ))
    fig.add_trace(go.Scatter(
        x=[min_variance["risk"]], y=[min_variance["return"]], mode="markers+text",
        marker=dict(size=18, symbol="diamond", color=C_ACCENT2, line=dict(color=C_BG, width=1.5)),
        name="Mín. Varianza", text=["Mín. Varianza"],
        textposition="top right", textfont=dict(color=C_ACCENT2, size=11),
    ))
    fig.update_layout(
        **CHART_THEME,
        height=520,
        title="Frontera eficiente de Markowitz",
        xaxis=dict(
            title="Riesgo (σ anualizado)",
            tickformat=".1%",
            gridcolor=C_BORDER
        ),
        yaxis=dict(
            title="Retorno esperado anualizado",
            tickformat=".1%",
            gridcolor=C_BORDER
        )
    )
    return fig


def plot_cumulative_vs_benchmark(portfolio_returns, benchmark_returns):
    fig = go.Figure()
    if len(portfolio_returns):
        p = (1 + portfolio_returns).cumprod() * 100
        fig.add_trace(go.Scatter(x=p.index, y=p.values, name="Portafolio óptimo",
                                 line=dict(color=C_ACCENT, width=2.5),
                                 fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"))
    if benchmark_returns is not None and len(benchmark_returns):
        common = benchmark_returns.loc[benchmark_returns.index.intersection(portfolio_returns.index)]
        if len(common):
            b = (1 + common).cumprod() * 100
            fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Benchmark (SPY)",
                                     line=dict(color=C_WARN, width=2, dash="dot")))
    fig.update_layout(**CHART_THEME, height=430,
                      title="Rendimiento acumulado — Portafolio vs Benchmark",
                      yaxis_title="Base 100")
    return fig


def plot_garch_conditional_vol(cond_vol, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol.values,
                             name="Volatilidad condicional",
                             line=dict(color=C_ACCENT, width=1.8),
                             fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"))
    if forecast:
        future_x = list(range(len(cond_vol), len(cond_vol) + len(forecast)))
        fig.add_trace(go.Scatter(x=future_x, y=forecast, name="Pronóstico",
                                 mode="lines+markers",
                                 line=dict(color=C_DANGER, width=2, dash="dash"),
                                 marker=dict(color=C_DANGER, size=7)))
    fig.update_layout(**CHART_THEME, height=360,
                      title="Volatilidad condicional GARCH y pronóstico")
    return fig


def plot_garch_residuals(std_resid):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=std_resid.values, nbinsx=45,
                               name="Residuos estand.",
                               marker_color=C_ACCENT2, opacity=0.75))
    x_range = np.linspace(float(std_resid.min()), float(std_resid.max()), 200)
    normal_y = (stats.norm.pdf(x_range)
                * len(std_resid)
                * (float(std_resid.max()) - float(std_resid.min())) / 45)
    fig.add_trace(go.Scatter(x=x_range, y=normal_y, name="Normal teórica",
                             line=dict(color=C_DANGER, width=2)))
    fig.update_layout(**CHART_THEME, height=360,
                      title="Distribución de residuos estandarizados GARCH")
    return fig


# ─────────────────────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────────────────────
def format_pct_df(df, pct_cols=None):
    pct_cols = pct_cols or []
    out = df.copy()
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: f"{x:.2%}"
                if pd.notna(x) and isinstance(x, (int, float, np.floating))
                else x
            )
    return out
