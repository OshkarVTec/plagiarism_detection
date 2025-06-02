#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del problema de la mochila 0/1 usando programación dinámica.
Lee desde stdin:
  - Primera línea: dos enteros n (número de elementos) y W (capacidad máxima).
  - Siguientes n líneas: cada una contiene dos enteros w y v (peso y valor del elemento).
Calcula el valor máximo que se puede obtener sin exceder la capacidad W,
y también determina cuáles elementos forman parte de la solución óptima.
Imprime en stdout:
  - Primera línea: valor máximo obtenido.
  - Segunda línea: lista de índices (0 a n-1) de los elementos seleccionados, separados por espacios.
"""

import sys

def leer_entrada():
    """
    Lee los datos de entrada desde stdin y retorna:
      - n: número de elementos
      - W: capacidad máxima de la mochila
      - pesos: lista de pesos de longitud n
      - valores: lista de valores de longitud n
    """
    try:
        primera_linea = sys.stdin.readline().strip().split()
        if len(primera_linea) != 2:
            raise ValueError("Se esperaban dos enteros en la primera línea: n y W.")
        n = int(primera_linea[0])
        W = int(primera_linea[1])
    except Exception as e:
        print(f"[ERROR] Entrada inválida (n y W): {e}")
        sys.exit(1)

    pesos = []
    valores = []
    for i in range(n):
        linea = sys.stdin.readline().strip().split()
        if len(linea) != 2:
            print(f"[ERROR] Línea {i+2} inválida: se esperaban dos enteros (peso y valor).")
            sys.exit(1)
        try:
            w = int(linea[0])
            v = int(linea[1])
            if w < 0 or v < 0:
                raise ValueError("Peso y valor deben ser no negativos.")
        except Exception as e:
            print(f"[ERROR] Entrada inválida en línea {i+2}: {e}")
            sys.exit(1)
        pesos.append(w)
        valores.append(v)

    return n, W, pesos, valores

def knapsack_01(n, W, pesos, valores):
    """
    Resuelve el problema de la mochila 0/1 con n elementos y capacidad W.
    Usa una tabla dp de tamaño (n+1) x (W+1), donde:
      dp[i][c] = valor máximo usando los primeros i elementos y capacidad c.
    También reconstruye los elementos elegidos mediante backtracking.
    Retorna tupla (valor_máximo, lista_elementos_seleccionados).
    """
    # Inicializar tabla dp con ceros
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # Llenar la tabla de programación dinámica
    for i in range(1, n + 1):
        wi = pesos[i - 1]
        vi = valores[i - 1]
        for c in range(W + 1):
            # Caso 1: no tomar el elemento i-1
            dp[i][c] = dp[i - 1][c]
            # Caso 2: tomar el elemento i-1 (si cabe)
            if wi <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - wi] + vi)

    # Valor máximo en dp[n][W]
    valor_maximo = dp[n][W]

    # Reconstrucción de la solución
    seleccionados = []
    c = W
    # Empezamos desde i = n y c = W
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            # El elemento i-1 fue elegido
            seleccionados.append(i - 1)
            c -= pesos[i - 1]
        # Si c llega a 0, ya no quedan elementos posibles
        if c == 0:
            break

    # Dado que hicimos backtracking de n a 1, invertimos la lista
    seleccionados.reverse()
    return valor_maximo, seleccionados

def imprimir_resultado(valor, items):
    """
    Imprime en stdout:
      - Primera línea: valor máximo obtenido.
      - Segunda línea: índices de elementos seleccionados separados por espacios.
    Si no hay elementos seleccionados, imprime una línea en blanco tras el valor.
    """
    print(valor)
    if items:
        print(" ".join(str(i) for i in items))
    else:
        print()

def main():
    """
    Flujo principal:
      1. Leer entrada de stdin.
      2. Resolver knapsack_01 mediante DP.
      3. Imprimir el resultado.
    """
    n, W, pesos, valores = leer_entrada()
    valor_maximo, seleccionados = knapsack_01(n, W, pesos, valores)
    imprimir_resultado(valor_maximo, seleccionados)

if __name__ == "__main__":
    main()
