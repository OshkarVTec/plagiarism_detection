#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de procesamiento de CSV para estadísticas:
  - Lee ruta del CSV desde stdin.
  - Carga CSV (csv.DictReader).
  - Para cada columna:
      * Si todos los valores no vacíos son numéricos (float), calcula:
          count, media, desviación estándar, mínimo, percentiles 25, 50, 75, máximo.
      * Si hay valores no numéricos: marca como NON_NUMERIC.
  - Escribe a stdout:
      COL\tcount\tmean\tstd\tmin\t25%\t50%\t75%\tmax
    o COL\tNON_NUMERIC
"""

import sys
import csv
import math

def abrir_csv(path):
    """
    Abre archivo CSV, retorna (encabezados, lista de filas como dicts).
    """
    try:
        with open(path, newline='', encoding='utf-8') as f:
            lector = csv.DictReader(f)
            filas = [fila for fila in lector]
            if not filas:
                raise ValueError("CSV sin datos.")
            return lector.fieldnames, filas
    except Exception as e:
        print(f"[ERROR] No se pudo abrir CSV '{path}': {e}")
        sys.exit(1)

def es_num_str(s):
    """
    Retorna True si s es convertible a float.
    """
    try:
        float(s)
        return True
    except:
        return False

def analizar_columnas(headers, filas):
    """
    Para cada columna en headers:
      - Recopila valores no vacíos.
      - Si todos pueden convertirse a float → numeric
      - Sino → NON_NUMERIC
    Retorna dict stats donde:
      stats[col] = { 'numeric': True/False, 'vals': [floats] }
    """
    stats = {}
    for col in headers:
        vals = []
        todo_num = True
        for fila in filas:
            val = fila[col].strip()
            if val == '':
                continue
            if es_num_str(val):
                vals.append(float(val))
            else:
                todo_num = False
                break
        if todo_num and vals:
            stats[col] = {'numeric': True, 'vals': vals}
        else:
            stats[col] = {'numeric': False, 'vals': []}
    return stats

def calc_desc(vals):
    """
    Dada lista de floats vals:
      - count, media, std (muestral), min, 25%, 50%, 75%, max.
    """
    n = len(vals)
    if n == 0:
        return None
    ordv = sorted(vals)
    count = n
    mean = sum(ordv) / n
    var = sum((x - mean) ** 2 for x in ordv) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    mn = ordv[0]
    mx = ordv[-1]
    def pct(p):
        idx = int(math.ceil(p/100 * n)) - 1
        idx = max(0, min(idx, n-1))
        return ordv[idx]
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

def imprimir(stats):
    """
    Para cada columna:
      - Si no numérica: “col\tNON_NUMERIC”
      - Si numérica: “col\tcount\tmean\tstd\tmin\t25%\t50%\t75%\tmax”
    Formatos con 4 decimales.
    """
    for col, info in stats.items():
        if not info['numeric']:
            print(f"{col}\tNON_NUMERIC")
        else:
            s = calc_desc(info['vals'])
            print(f"{col}\t{s['count']}\t{s['mean']:.4f}\t{s['std']:.4f}\t"
                  f"{s['min']:.4f}\t{s['25%']:.4f}\t{s['50%']:.4f}\t"
                  f"{s['75%']:.4f}\t{s['max']:.4f}")

def main():
    """
    1. Leer ruta CSV.
    2. Llamar a abrir_csv().
    3. Llamar a analizar_columnas().
    4. Imprimir resultados.
    """
    path = sys.stdin.readline().strip()
    encabezados, filas = abrir_csv(path)
    stats = analizar_columnas(encabezados, filas)
    imprimir(stats)

if __name__ == "__main__":
    main()
