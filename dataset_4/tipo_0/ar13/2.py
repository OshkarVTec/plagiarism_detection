#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estrategia de trading simple basada en cruces de Promedio Móvil.
Descarga datos históricos de una acción con yfinance y genera señales
de compra/venta cuando el PM corto cruza al PM largo.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def download_data(ticker, period="1y", interval="1d"):
    """
    Descarga datos históricos de la acción especificada usando yfinance.
    Retorna un DataFrame con columnas: 'Date','Open','High','Low','Close','Adj Close','Volume'.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError("No se encontraron datos para el ticker.")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"[ERROR] Falló la descarga de datos: {e}")
        sys.exit(1)

def compute_moving_averages(data, short_window=20, long_window=50):
    """
    Calcula las columnas de Promedio Móvil a corto (short_window) y largo (long_window).
    Modifica el DataFrame en sitio.
    """
    data[f"MA{short_window}"] = data["Close"].rolling(window=short_window).mean()
    data[f"MA{long_window}"] = data["Close"].rolling(window=long_window).mean()

def generate_signals(data, short_window=20, long_window=50):
    """
    Genera señales de compra (+1) y venta (-1) cuando el MA corto cruza al MA largo.
    Retorna un DataFrame con la columna 'Signal'.
    """
    data["Signal"] = 0
    data["Signal"] = np.where(data[f"MA{short_window}"] > data[f"MA{long_window}"], 1, 0)
    data["Signal"] = data["Signal"].diff()  # +1 = cruce al alza, -1 = cruce a la baja

def backtest_strategy(data, initial_capital=10000.0):
    """
    Backtest de la estrategia:
    - Compra al cierre cuando Signal==1.
    - Vende al cierre cuando Signal==-1.
    - Mantiene posición en efectivo entre transacciones.
    Retorna DataFrame con columnas adicionales: 'Positions','Holdings','Cash','Total'.
    """
    data["Positions"] = 0
    data["Cash"] = initial_capital
    data["Holdings"] = 0.0
    data["Total"] = initial_capital

    position = 0  # 0 = sin posición, 1 = posición abierta
    for i in range(len(data)):
        price = data.loc[i, "Close"]
        signal = data.loc[i, "Signal"]

        if signal == 1 and position == 0:
            # Comprar una acción entera
            qty = initial_capital // price
            cost = qty * price
            data.at[i, "Positions"] = qty
            position = qty
            data.at[i, "Cash"] = initial_capital - cost
            data.at[i, "Holdings"] = qty * price
            data.at[i, "Total"] = data.at[i, "Cash"] + data.at[i, "Holdings"]
        elif signal == -1 and position > 0:
            # Vender todas las acciones
            revenue = position * price
            data.at[i, "Positions"] = -position
            data.at[i, "Cash"] = data.at[i-1, "Cash"] + revenue
            data.at[i, "Holdings"] = 0.0
            data.at[i, "Total"] = data.at[i, "Cash"]
            position = 0
        else:
            # Mantener estado
            data.at[i, "Positions"] = 0
            if position > 0:
                data.at[i, "Holdings"] = position * price
                data.at[i, "Cash"] = data.at[i-1, "Cash"]
                data.at[i, "Total"] = data.at[i, "Cash"] + data.at[i, "Holdings"]
            else:
                data.at[i, "Cash"] = data.at[i-1, "Cash"]
                data.at[i, "Holdings"] = 0.0
                data.at[i, "Total"] = data.at[i, "Cash"]

def plot_results(data, ticker, short_window, long_window):
    """
    Grafica el precio de cierre, promedios móviles y equity curve de la estrategia.
    Guarda la figura en 'trading_result.png'.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Precio y Promedios Móviles
    ax1.plot(data["Date"], data["Close"], label=f"{ticker} Close", alpha=0.5)
    ax1.plot(data["Date"], data[f"MA{short_window}"], label=f"MA {short_window}")
    ax1.plot(data["Date"], data[f"MA{long_window}"], label=f"MA {long_window}")
    ax1.set_title(f"{ticker} Precio y Promedios Móviles")
    ax1.set_ylabel("Precio")
    ax1.legend()

    # Equity Curve
    ax2.plot(data["Date"], data["Total"], label="Equity Curve", color="purple")
    ax2.set_title("Equity Curve de la Estrategia")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Valor de la cartera")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("trading_result.png")
    print("[INFO] Gráfica de resultados guardada en 'trading_result.png'")
