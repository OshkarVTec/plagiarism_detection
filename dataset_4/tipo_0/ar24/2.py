#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard en tiempo real usando Dash (por Plotly) para visualizar datos de un CSV:
  - Lee un archivo CSV con datos de ventas (fecha, producto, cantidad, precio).
  - Muestra:
      * Gráfica de barras de ventas por producto.
      * Gráfica de línea de ingresos acumulados por fecha.
      * Tabla interactiva de los datos.
  - Permite filtrar por rango de fechas usando componentes Dash.
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import datetime
import sys
import os

# Nombre del archivo CSV de ejemplo
DATA_FILE = "ventas.csv"

def load_data(file_path):
    """
    Lee el CSV en un DataFrame de pandas y parsea columna 'fecha' como datetime.
    Columnas esperadas: fecha (YYYY-MM-DD), producto, cantidad, precio.
    Retorna DataFrame con columna adicional 'ingreso' = cantidad * precio.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] No se encontró el archivo: {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path, parse_dates=['fecha'])
    df['ingreso'] = df['cantidad'] * df['precio']
    return df

# Cargar datos inicialmente
df_orig = load_data(DATA_FILE)

# Crear la aplicación Dash
app = dash.Dash(__name__)
app.title = "Dashboard de Ventas"

# Layout de la aplicación
app.layout = html.Div(children=[
    html.H1(children="Dashboard de Ventas", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Selecciona rango de fechas:"),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df_orig['fecha'].min(),
            end_date=df_orig['fecha'].max(),
            display_format='YYYY-MM-DD'
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    html.Div([
        dcc.Graph(id='bar-chart-producto'),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='line-chart-ingresos'),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.H2("Datos de Ventas"),
    dash_table.DataTable(
        id='table-ventas',
        columns=[{"name": i, "id": i} for i in df_orig.columns],
        data=df_orig.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    )
])

@app.callback(
    [Output('bar-chart-producto', 'figure'),
     Output('line-chart-ingresos', 'figure'),
     Output('table-ventas', 'data')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_charts(start_date, end_date):
    """
    Callback que se activa al cambiar el rango de fechas.
    Filtra los datos y actualiza:
      - Gráfica de barras de ventas por producto (suma de ingreso).
      - Gráfica de línea de ingresos acumulados por fecha.
      - Datos de la tabla.
    """
    if start_date is None or end_date is None:
        dff = df_orig.copy()
    else:
        mask = (df_orig['fecha'] >= start_date) & (df_orig['fecha'] <= end_date)
        dff = df_orig.loc[mask]

    # Agregación por producto
    ventas_prod = dff.groupby('producto')['ingreso'].sum().reset_index()

    # Gráfica de barras
    bar_fig = {
        'data': [{
            'x': ventas_prod['producto'],
            'y': ventas_prod['ingreso'],
            'type': 'bar',
            'marker': {'color': 'green'}
        }],
        'layout': {
            'title': 'Ingresos por Producto',
            'xaxis': {'title': 'Producto'},
            'yaxis': {'title': 'Ingresos ($)'}
        }
    }

    # Ingresos acumulados por fecha
    ingresos_fecha = dff.groupby('fecha')['ingreso'].sum().cumsum().reset_index()

    # Gráfica de línea
    line_fig = {
        'data': [{
            'x': ingresos_fecha['fecha'],
            'y': ingresos_fecha['ingreso'],
            'type': 'line',
            'marker': {'color': 'blue'}
        }],
        'layout': {
            'title': 'Ingresos Acumulados por Fecha',
            'xaxis': {'title': 'Fecha'},
            'yaxis': {'title': 'Ingresos Acumulados ($)'}
        }
    }

    # Datos de tabla como lista de diccionarios
    table_data = dff.to_dict('records')

    return bar_fig, line_fig, table_data
