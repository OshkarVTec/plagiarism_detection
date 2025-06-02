#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de Matrix Chain Multiplication (MCM) usando DP:
Misma lógica para hallar el costo mínimo de multiplicar n matrices con dimensiones p[].
Lee de stdin:
  - Línea 1: N (número de matrices)
  - Línea 2: d0 d1 ... dN (N+1 valores de dimensión)
Imprime en stdout:
  - Un solo entero: el costo mínimo de las multiplicaciones.
"""

import sys

def cargar_dimensiones():
    """
    Lee N y luego N+1 dimensiones d[0..N]. Valida que sean enteros positivos.
    Retorna N y la lista d de longitud N+1.
    """
    try:
        linea = sys.stdin.readline().strip()
        if not linea:
            raise ValueError("Falta el número de matrices.")
        N = int(linea)
        if N <= 0:
            raise ValueError("N debe ser mayor que cero.")
    except Exception as e:
        print(f"[ERROR] Entrada inválida para N: {e}")
        sys.exit(1)

    tokens = sys.stdin.readline().strip().split()
    if len(tokens) != N + 1:
        print(f"[ERROR] Se esperaban {N+1} dimensiones, recibidas {len(tokens)}.")
        sys.exit(1)
    d = []
    for idx, tok in enumerate(tokens):
        try:
            val = int(tok)
            if val <= 0:
                raise ValueError("Cada dimensión debe ser un entero positivo.")
        except Exception as ex:
            print(f"[ERROR] Dimensión inválida en índice {idx}: {ex}")
            sys.exit(1)
        d.append(val)
    return N, d

def costo_mcm(N, d):
    """
    Calcula costo mínimo de MCM con DP:
      - Crea mcm_tab de tamaño (N+1) x (N+1), donde mcm_tab[i][j] es el costo mínimo
        de multiplicar matrices Ai..Aj.
      - Inicializa mcm_tab[i][i] = 0.
      - Para tamaño secuencia L en [2..N]:
          * Para i en [1..N-L+1]:
              j = i + L - 1
              mcm_tab[i][j] = infinito
              Para k en [i..j-1]:
                  q = mcm_tab[i][k] + mcm_tab[k+1][j] + d[i-1]*d[k]*d[j]
                  si q < mcm_tab[i][j]:
                      mcm_tab[i][j] = q
    Retorna mcm_tab[1][N].
    """
    # Inicializar matriz de costos
    mcm_tab = [[0] * (N + 1) for _ in range(N + 1)]
    INF = float('inf')

    for L in range(2, N + 1):
        for i in range(1, N - L + 2):
            j = i + L - 1
            mcm_tab[i][j] = INF
            for k in range(i, j):
                q = mcm_tab[i][k] + mcm_tab[k+1][j] + d[i-1] * d[k] * d[j]
                if q < mcm_tab[i][j]:
                    mcm_tab[i][j] = q
    return mcm_tab[1][N]

def main():
    """
    Flujo de ejecución:
      1. Leer N y lista d[] de tamaños con cargar_dimensiones().
      2. Calcular costo mínimo con costo_mcm().
      3. Imprimir el resultado en stdout.
    """
    N, d = cargar_dimensiones()
    resultado = costo_mcm(N, d)
    print(resultado)

if __name__ == "__main__":
    main()
