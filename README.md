# 📈 Proyecto Integrador - Teoría del Riesgo

**Autores:** Juan Pablo Vargas y Vanessa Acosta  
**Curso:** Teoría del Riesgo  
**Docente:** Javier Mauricio Sierra  

---
Link Streamlit: (https://proyectoteoria-jljfbufsmpbpryrcsbxega.streamlit.app/)

## 1. Descripción general del proyecto

Este proyecto consiste en el desarrollo de un **tablero interactivo en Streamlit** para el análisis integral de riesgo financiero sobre un portafolio de activos, utilizando datos dinámicos obtenidos desde APIs externas.

El objetivo principal es **apoyar la toma de decisiones de inversión** mediante la integración de herramientas cuantitativas de análisis financiero, incluyendo:

- análisis técnico,
- caracterización estadística de rendimientos,
- modelos de volatilidad condicional,
- CAPM y riesgo sistemático,
- Valor en Riesgo (VaR) y CVaR,
- optimización de portafolios bajo Markowitz,
- señales automáticas de trading,
- benchmark y contexto macroeconómico.

El sistema fue diseñado con una estructura simple y clara:

- `app.py`: aplicación principal en Streamlit
- `models/`: módulos de lógica financiera, consumo de APIs y visualización

---

## 2. Objetivo del proyecto

Diseñar e implementar un tablero profesional que permita:

- consultar activos dinámicamente desde APIs,
- calcular indicadores técnicos y métricas de riesgo,
- modelar volatilidad y pérdidas extremas,
- estimar riesgo sistemático mediante CAPM,
- optimizar portafolios,
- generar señales interpretables de compra/venta,
- comparar el desempeño del portafolio frente a un benchmark de mercado,
- incorporar variables macroeconómicas actualizadas.

---

## 3. Estructura del proyecto

```bash
proyecto_teoria/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
└── models/
    ├── __init__.py
    ├── data_api.py
    ├── technical.py
    ├── returns_analysis.py
    ├── garch_model.py
    ├── capm_model.py
    ├── risk_model.py
    ├── portfolio_model.py
    ├── performance_model.py
    └── ui_components.py
