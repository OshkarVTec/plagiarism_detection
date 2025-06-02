#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del algoritmo de Floyd-Warshall para hallar las distancias mínimas
entre todos los pares de vértices en un grafo dirigido con pesos (pueden ser positivos).
Lee desde stdin:
  - Primera línea: n (número de vértices)
  - Siguientes n líneas: cada una con n enteros, representando la matriz de adyacencia
    donde un valor >= 0 es el peso de la arista, y -1 indica que no hay arista directa.
Calcula la matriz de distancias mínimas y la imprime en stdout con el mismo formato:
  - n líneas, cada una con n valores separados por espacios.
  - Si no hay camino entre i y j, imprimir -1.
"""

import sys

def leer_matriz():
    """
    Lee el número de vértices n y luego n líneas con n enteros cada una.
    Retorna n y la matriz de adyacencia en forma de lista de listas.
    """
    try:
        primera = sys.stdin.readline().strip()
        if not primera:
            raise ValueError("No se recibió número de vértices.")
        n = int(primera)
        if n <= 0:
            raise ValueError("n debe ser positivo.")
    except Exception as e:
        print(f"[ERROR] Entrada inválida para número de vértices: {e}")
        sys.exit(1)

    matriz = []
    for i in range(n):
        linea = sys.stdin.readline().strip().split()
        if len(linea) != n:
            print(f"[ERROR] Se esperaban {n} valores en la fila {i}, pero se recibieron {len(linea)}.")
            sys.exit(1)
        fila = []
        for j, tok in enumerate(linea):
            try:
                peso = int(tok)
            except ValueError:
                print(f"[ERROR] Valor no entero en posición ({i},{j}): '{tok}'.")
                sys.exit(1)
            fila.append(peso)
        matriz.append(fila)

    return n, matriz

def inicializar_distancias(n, ady):
    """
    Crea la matriz de distancias dist[][] a partir de la matriz de adyacencia ady:
      - Si ady[i][j] >= 0, entonces dist[i][j] = ady[i][j].
      - Si i == j, dist[i][j] = 0.
      - Si ady[i][j] == -1, dist[i][j] = infinito (representado con None temporalmente).
    Retorna la matriz dist inicializada.
    """
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif ady[i][j] >= 0:
                dist[i][j] = ady[i][j]
            # si ady[i][j] == -1, queda en INF
    return dist

def floyd_warshall(n, dist):
    """
    Ejecuta el algoritmo de Floyd-Warshall sobre la matriz dist de tamaño n x n:
    Para cada k, i, j:
      if dist[i][k] + dist[k][j] < dist[i][j]:
          dist[i][j] = dist[i][k] + dist[k][j]
    Al final, dist[i][j] será la distancia mínima entre i y j, o INF si no hay camino.
    """
    for k in range(n):
        for i in range(n):
            # Si dist[i][k] es infinito, no tiene sentido intentar mejorar desde ese camino
            if dist[i][k] == float('inf'):
                continue
            for j in range(n):
                if dist[k][j] == float('inf'):
                    continue
                nueva = dist[i][k] + dist[k][j]
                if nueva < dist[i][j]:
                    dist[i][j] = nueva

def imprimir_distancias(n, dist):
    """
    Imprime en stdout la matriz de distancias de tamaño n x n:
      - Si dist[i][j] es infinito, imprime -1.
      - Si dist[i][j] es un número, lo imprime tal cual.
    Cada fila en una línea, valores separados por espacios.
    """
    for i in range(n):
        fila_salida = []
        for j in range(n):
            if dist[i][j] == float('inf'):
                fila_salida.append(str(-1))
            else:
                fila_salida.append(str(dist[i][j]))
        print(" ".join(fila_salida))

def main():
    """
    Flujo principal:
      1. Leer n y matriz de adyacencia.
      2. Inicializar matriz de distancias.
      3. Ejecutar Floyd-Warshall.
      4. Imprimir matriz resultante.
    """
    n, ady_mat = leer_matriz()
    dist = inicializar_distancias(n, ady_mat)
    floyd_warshall(n, dist)
    imprimir_distancias(n, dist)

if __name__ == "__main__":
    main()
