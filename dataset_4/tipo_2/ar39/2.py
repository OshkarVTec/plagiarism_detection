#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión Tipo 2 de Floyd-Warshall:
Misma funcionalidad para hallar distancias mínimas entre todos los pares, 
con renombrado de funciones, variables y reescritura de comentarios.
Lee de stdin:
  - Línea 1: V (cantidad de vértices)
  - V líneas con V enteros cada una: matriz de adyacencia, donde peso >= 0 indica arista,
    y -1 significa ausencia de arista.
Calcula la matriz de distancias mínimas y escribe en stdout:
  - V filas, cada una con V valores separados por espacios,
    usando -1 para indicar que no hay camino.
"""

import sys

def obtener_matriz():
    """
    Lee la cantidad de vértices y luego la matriz de adyacencia:
      - Primera línea: V
      - Siguientes V líneas: cada una con V valores enteros.
    Retorna V y la lista de listas con los pesos.
    """
    try:
        linea = sys.stdin.readline().strip()
        if not linea:
            raise ValueError("No proporcionó la cantidad de vértices.")
        V = int(linea)
        if V <= 0:
            raise ValueError("La cantidad de vértices debe ser mayor que cero.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida de V: {e}")
        sys.exit(1)

    matriz = []
    for fila_idx in range(V):
        partes = sys.stdin.readline().strip().split()
        if len(partes) != V:
            print(f"[ERROR] Se esperaban {V} enteros en la fila {fila_idx}, recibidos {len(partes)}.")
            sys.exit(1)
        fila = []
        for col_idx, tok in enumerate(partes):
            try:
                val = int(tok)
            except ValueError:
                print(f"[ERROR] Valor no válido en posición ({fila_idx},{col_idx}): '{tok}'.")
                sys.exit(1)
            fila.append(val)
        matriz.append(fila)

    return V, matriz

def iniciar_dist(V, adj):
    """
    A partir de la matriz de adyacencia adj (de V x V):
      - Si adj[i][j] >= 0, distancias[i][j] = adj[i][j].
      - Si i == j, distancias[i][i] = 0.
      - Si adj[i][j] == -1, distancias[i][j] = infinito.
    Retorna la matriz dist inicial.
    """
    INF = float('inf')
    dist = [[INF] * V for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j:
                dist[i][j] = 0
            elif adj[i][j] >= 0:
                dist[i][j] = adj[i][j]
            # Si adj[i][j] == -1, dejamos INF
    return dist

def calcular_todas_distancias(V, distancias):
    """
    Ejecución del algoritmo de Floyd-Warshall:
      Para cada vértice k, i, j:
        si distancias[i][k] + distancias[k][j] < distancias[i][j]:
            actualizar distancias[i][j].
    Al final, distancias[i][j] contendrá el valor mínimo o infinito.
    """
    for k in range(V):
        for i in range(V):
            if distancias[i][k] == float('inf'):
                continue
            for j in range(V):
                if distancias[k][j] == float('inf'):
                    continue
                alt = distancias[i][k] + distancias[k][j]
                if alt < distancias[i][j]:
                    distancias[i][j] = alt

def mostrar_matriz(V, distancias):
    """
    Imprime la matriz de distancias resultante:
      - Si distancias[i][j] es infinito, imprime -1.
      - En otro caso, imprime el valor numérico.
    Cada fila en una sola línea, valores separados por espacios.
    """
    for i in range(V):
        salida_fila = []
        for j in range(V):
            if distancias[i][j] == float('inf'):
                salida_fila.append(str(-1))
            else:
                salida_fila.append(str(distancias[i][j]))
        print(" ".join(salida_fila))

def main():
    """
    Lógica principal:
      1. Leer V y matriz de adyacencia.
      2. Inicializar matriz de distancias.
      3. Aplicar Floyd-Warshall.
      4. Mostrar la matriz final.
    """
    V, matriz_ady = obtener_matriz()
    dist_init = iniciar_dist(V, matriz_ady)
    calcular_todas_distancias(V, dist_init)
    mostrar_matriz(V, dist_init)

if __name__ == "__main__":
    main()
