#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del problema de Matrix Chain Multiplication (MCM) usando programación dinámica.
Dado un arreglo p[] de dimensiones de matrices tal que la matriz Ai tiene dimensiones
p[i-1] x p[i], se calcula el número mínimo de multiplicaciones escalares necesarias
para multiplicar la secuencia de matrices A1 x A2 x ... x An.
Lee desde stdin:
  - Primera línea: n (número de matrices, por lo que p tendrá longitud n+1).
  - Segunda línea: p0 p1 p2 ... pn (n+1 enteros, dimensiones).
Imprime en stdout:
  - Un entero: el costo mínimo de multiplicación.
"""

import sys

def leer_dimensiones():
    """
    Lee n y luego n+1 dimensiones p[0..n].
    Retorna n y la lista p de longitud n+1.
    """
    try:
        linea_n = sys.stdin.readline().strip()
        if not linea_n:
            raise ValueError("No se recibió la cantidad de matrices.")
        n = int(linea_n)
        if n <= 0:
            raise ValueError("n debe ser mayor que cero.")
    except Exception as e:
        print(f"[ERROR] Entrada inválida para n: {e}")
        sys.exit(1)

    partes = sys.stdin.readline().strip().split()
    if len(partes) != n + 1:
        print(f"[ERROR] Se esperaban {n+1} dimensiones, pero se recibieron {len(partes)}.")
        sys.exit(1)
    p = []
    for idx, tok in enumerate(partes):
        try:
            val = int(tok)
            if val <= 0:
                raise ValueError("Dimensiones deben ser positivas.")
        except Exception as e:
            print(f"[ERROR] Dimensión no válida en posición {idx}: {e}")
            sys.exit(1)
        p.append(val)
    return n, p

def matrix_chain_order(n, p):
    """
    Calcula el costo mínimo de multiplicación utilizando DP:
      - Crea una matriz m de tamaño (n+1) x (n+1), donde m[i][j] es el costo mínimo
        de multiplicar matrices Ai x ... x Aj.
      - Inicializa m[i][i] = 0 para todo i.
      - Para l en 2..n (longitud de subsecuencia):
          * Para i en 1..n-l+1:
              j = i + l - 1
              m[i][j] = infinito
              Para k en i..j-1:
                  q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]
                  si q < m[i][j]:
                      m[i][j] = q
    Retorna m[1][n], el costo mínimo total.
    """
    # Inicializar tabla m con ceros
    m = [[0] * (n + 1) for _ in range(n + 1)]
    INF = float('inf')

    # L representa la longitud de la subsecuencia de matrices
    for L in range(2, n + 1):
        for i in range(1, n - L + 2):
            j = i + L - 1
            m[i][j] = INF
            for k in range(i, j):
                costo = m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j]
                if costo < m[i][j]:
                    m[i][j] = costo
    return m[1][n]

def main():
    """
    Flujo principal:
      1. Leer n y arreglo p[] de dimensiones.
      2. Llamar a matrix_chain_order para obtener costo mínimo.
      3. Imprimir el resultado.
    """
    n, p = leer_dimensiones()
    costo_min = matrix_chain_order(n, p)
    print(costo_min)

if __name__ == "__main__":
    main()
