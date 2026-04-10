from __future__ import annotations
import numpy as np
import pandas as pd

def sma(series: pd.Series, n: int):
    return series.rolling(n).mean()

def ema(series: pd.Series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window=20, num_std=2):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def stochastic(series: pd.Series, k_period=14, d_period=3):
    low = series.rolling(k_period).min()
    high = series.rolling(k_period).max()
    k = 100 * (series - low) / (high - low)
    d = k.rolling(d_period).mean()
    return k, d

def technical_pack(
    prices: pd.Series,
    sma_short=20, sma_long=50,
    ema_short=20, ema_long=50,
    rsi_period=14, bb_window=20,
    bb_std=2, stoch_k=14, stoch_d=3
):
    m, s, h = macd(prices)
    bu, bm, bl = bollinger(prices, bb_window, bb_std)
    k, d = stochastic(prices, stoch_k, stoch_d)
    rs = rsi(prices, rsi_period)
    return {
        "sma_short": sma(prices, sma_short),
        "sma_long": sma(prices, sma_long),
        "ema_short": ema(prices, ema_short),
        "ema_long": ema(prices, ema_long),
        "rsi": rs,
        "macd": m,
        "macd_signal": s,
        "macd_hist": h,
        "bb_upper": bu,
        "bb_mid": bm,
        "bb_lower": bl,
        "stochastic_k": k,
        "stochastic_d": d,
    }

def indicator_interpretations(pack, last_price, rsi_upper=70, rsi_lower=30):
    out = {}
    last_rsi = float(pack["rsi"].dropna().iloc[-1]) if not pack["rsi"].dropna().empty else 50.0
    if last_rsi > rsi_upper:
        out["RSI"] = f"El RSI está en {last_rsi:.1f}, lo que sugiere sobrecompra y potencial agotamiento alcista."
    elif last_rsi < rsi_lower:
        out["RSI"] = f"El RSI está en {last_rsi:.1f}, lo que sugiere sobreventa y posible rebote técnico."
    else:
        out["RSI"] = f"El RSI está en zona neutral ({last_rsi:.1f})."

    macd_last = float(pack["macd"].dropna().iloc[-1]) if not pack["macd"].dropna().empty else 0.0
    sig_last = float(pack["macd_signal"].dropna().iloc[-1]) if not pack["macd_signal"].dropna().empty else 0.0
    out["MACD"] = "El MACD está por encima de la señal, indicando momentum positivo." if macd_last > sig_last else "El MACD está por debajo de la señal, indicando pérdida de momentum."

    bb_u = float(pack["bb_upper"].dropna().iloc[-1]) if not pack["bb_upper"].dropna().empty else last_price
    bb_l = float(pack["bb_lower"].dropna().iloc[-1]) if not pack["bb_lower"].dropna().empty else last_price
    if last_price >= bb_u:
        out["Bollinger"] = "El precio está en la banda superior, lo que sugiere tensión alcista o sobreextensión."
    elif last_price <= bb_l:
        out["Bollinger"] = "El precio está en la banda inferior, lo que sugiere presión bajista o posible reversión."
    else:
        out["Bollinger"] = "El precio se mantiene dentro de las bandas en zona intermedia."

    k_last = float(pack["stochastic_k"].dropna().iloc[-1]) if not pack["stochastic_k"].dropna().empty else 50.0
    d_last = float(pack["stochastic_d"].dropna().iloc[-1]) if not pack["stochastic_d"].dropna().empty else 50.0
    if k_last > 80:
        out["Estocástico"] = "El estocástico está en zona de sobrecompra."
    elif k_last < 20:
        out["Estocástico"] = "El estocástico está en zona de sobreventa."
    else:
        out["Estocástico"] = "El estocástico está en zona neutral."

    sma_s = float(pack["sma_short"].dropna().iloc[-1]) if not pack["sma_short"].dropna().empty else last_price
    sma_l = float(pack["sma_long"].dropna().iloc[-1]) if not pack["sma_long"].dropna().empty else last_price
    out["Cruce de medias"] = "La media corta está por encima de la larga, compatible con sesgo alcista." if sma_s > sma_l else "La media corta está por debajo de la larga, compatible con sesgo bajista."
    return out

def signal_summary(prices: pd.Series, rsi_upper=70, rsi_lower=30, sma_short=20, sma_long=50, bb_window=20, bb_std=2, stoch_k=14, stoch_d=3):
    pack = technical_pack(prices, sma_short, sma_long, sma_short, sma_long, 14, bb_window, bb_std, stoch_k, stoch_d)
    reasons = []
    buy = 0
    sell = 0

    last_price = float(prices.dropna().iloc[-1])
    last_rsi = float(pack["rsi"].dropna().iloc[-1]) if not pack["rsi"].dropna().empty else 50.0
    if last_rsi < rsi_lower:
        buy += 1; reasons.append(f"RSI en sobreventa ({last_rsi:.1f})")
    elif last_rsi > rsi_upper:
        sell += 1; reasons.append(f"RSI en sobrecompra ({last_rsi:.1f})")

    macd_last = float(pack["macd"].dropna().iloc[-1]) if not pack["macd"].dropna().empty else 0.0
    sig_last = float(pack["macd_signal"].dropna().iloc[-1]) if not pack["macd_signal"].dropna().empty else 0.0
    hist_last = float(pack["macd_hist"].dropna().iloc[-1]) if not pack["macd_hist"].dropna().empty else 0.0
    if macd_last > sig_last and hist_last > 0:
        buy += 1; reasons.append("Momentum alcista por MACD")
    elif macd_last < sig_last and hist_last < 0:
        sell += 1; reasons.append("Momentum bajista por MACD")

    bb_u = float(pack["bb_upper"].dropna().iloc[-1]) if not pack["bb_upper"].dropna().empty else last_price
    bb_l = float(pack["bb_lower"].dropna().iloc[-1]) if not pack["bb_lower"].dropna().empty else last_price
    if last_price <= bb_l:
        buy += 1; reasons.append("Precio tocando banda inferior de Bollinger")
    elif last_price >= bb_u:
        sell += 1; reasons.append("Precio tocando banda superior de Bollinger")

    k_last = float(pack["stochastic_k"].dropna().iloc[-1]) if not pack["stochastic_k"].dropna().empty else 50.0
    d_last = float(pack["stochastic_d"].dropna().iloc[-1]) if not pack["stochastic_d"].dropna().empty else 50.0
    if k_last < 20 and k_last > d_last:
        buy += 1; reasons.append("Estocástico con reversión alcista")
    elif k_last > 80 and k_last < d_last:
        sell += 1; reasons.append("Estocástico con reversión bajista")

    sma_s = float(pack["sma_short"].dropna().iloc[-1]) if not pack["sma_short"].dropna().empty else last_price
    sma_l = float(pack["sma_long"].dropna().iloc[-1]) if not pack["sma_long"].dropna().empty else last_price
    if sma_s > sma_l:
        buy += 1; reasons.append("Media corta por encima de media larga")
    elif sma_s < sma_l:
        sell += 1; reasons.append("Media corta por debajo de media larga")

    if buy > sell:
        signal = "COMPRAR"; color = "#16a34a"
    elif sell > buy:
        signal = "VENDER"; color = "#dc2626"
    else:
        signal = "NEUTRAL"; color = "#ca8a04"

    confidence = round(max(buy, sell) / 5 * 100)
    interpretation = f"La señal final es {signal.lower()} con una confianza aproximada de {confidence}%. Se basa en RSI, MACD, Bollinger, Estocástico y cruces de medias."
    return {
        "signal": signal,
        "confidence": confidence,
        "reasons": reasons,
        "color": color,
        "interpretation": interpretation,
        "current_price": last_price,
    }
