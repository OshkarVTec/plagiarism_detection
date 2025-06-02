#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para leer un archivo CSV (con encabezados) y calcular estadísticas básicas
para cada columna numérica.  
Entrada desde stdin: nombre del archivo CSV.  
Salida a stdout:
  - Para cada columna numérica: nombre, count, mean, std, min, 25%, 50%, 75%, max
  - Valores con 4 decimales.
Se asume:
  - Separador: coma “,”
  - Comillas dobles para campos (estándar)
  - Columnas no numéricas se ignoran (pero se listan).
"""

import sys
import csv
import math

def leer_csv(ruta):
    """
    Abre archivo CSV y retorna lista de diccionarios (cada fila).
    Lee encabezados de la primera línea.
    """
    try:
        with open(ruta, newline='', encoding='utf-8') as f:
            lector = csv.DictReader(f)
            filas = [row for row in lector]
            if not filas:
                raise ValueError("CSV vacío.")
            return lector.fieldnames, filas
    except Exception as e:
        print(f"[ERROR] Al leer CSV '{ruta}': {e}")
        sys.exit(1)

def es_numerico(valor):
    """
    Retorna True si valor (string) es convertible a float.
    """
    try:
        float(valor)
        return True
    except:
        return False

def calcular_estadisticas(fieldnames, filas):
    """
    Para cada columna:
      - Si todos los valores (no vacíos) son numéricos: calcular count, mean, std, min, pctiles.
      - Si hay valores no numéricos: marcar como no numérica.
    Retorna dict col_stats donde
      col_stats[col] = {
        'numeric': True/False,
        'values': [floats si numeric],
      }
    """
    col_stats = {}
    for col in fieldnames:
        valores = []
        es_num = True
        for row in filas:
            val = row[col].strip()
            if val == "":
                continue
            if es_numerico(val):
                valores.append(float(val))
            else:
                es_num = False
                break
        if es_num and valores:
            col_stats[col] = {'numeric': True, 'values': valores}
        else:
            col_stats[col] = {'numeric': False, 'values': []}
    return col_stats

def estadisticas_descriptivas(valores):
    """
    Dada lista de floats, retorna dict con:
      count, mean, std, min, 25%, 50%, 75%, max.
    std = desviación estándar muestral (sqrt(Σ(xi - mean)^2 / (n-1)))
    Percentiles: método simple de ordenación e indexamiento.
    """
    n = len(valores)
    if n == 0:
        return None
    ordenado = sorted(valores)
    count = n
    mean = sum(ordenado) / n
    var = sum((x - mean) ** 2 for x in ordenado) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    mn = ordenado[0]
    mx = ordenado[-1]
    def pct(p):
        idx = int(math.ceil(p / 100 * n)) - 1
        idx = max(0, min(idx, n-1))
        return ordenado[idx]
    p25 = pct(25)
    p50 = pct(50)
    p75 = pct(75)
    return {
        'count': count,
        'mean': mean,
        'std': std,
        'min': mn,
        '25%': p25,
        '50%': p50,
        '75%': p75,
        'max': mx
    }

def print_stats(col_stats):
    """
    Imprime estadísticas para cada columna:
      - Si es numérica: “col\tcount\tmean\tstd\tmin\t25%\t50%\t75%\tmax”
      - Si no: “col\tNON_NUMERIC”
    """
    for col, info in col_stats.items():
        if not info['numeric']:
            print(f"{col}\tNON_NUMERIC")
        else:
            stats = estadisticas_descriptivas(info['values'])
            print(f"{col}\t{stats['count']}\t{stats['mean']:.4f}\t{stats['std']:.4f}\t"
                  f"{stats['min']:.4f}\t{stats['25%']:.4f}\t{stats['50%']:.4f}\t"
                  f"{stats['75%']:.4f}\t{stats['max']:.4f}")

def main():
    """
    1. Leer ruta de CSV desde stdin.
    2. Leer CSV con leer_csv.
    3. Calcular col_stats con calcular_estadisticas.
    4. Imprimir resultados con print_stats.
    """
    ruta = sys.stdin.readline().strip()
    headers, filas = leer_csv(ruta)
    col_stats = calcular_estadisticas(headers, filas)
    print_stats(col_stats)

if __name__ == "__main__":
    main()
