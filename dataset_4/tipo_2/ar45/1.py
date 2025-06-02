#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del algoritmo A* para encontrar el camino más corto en una grilla.
La grilla se lee desde stdin:
  - Primera línea: filas (R) y columnas (C)
  - Siguientes R líneas: cada línea con C caracteres:
      '.' = celda libre, '#' = pared
  - Última línea: coordenadas del punto de inicio y fin:
      r_start c_start r_goal c_goal
Se asume 0 ≤ r < R, 0 ≤ c < C. Se puede mover en las 4 direcciones (arriba, abajo, izquierda, derecha).
Distancia de cada movimiento es 1. Heurística: distancia de Manhattan.
Al final, imprime en stdout:
  - Si existe camino: primero la longitud (número de pasos),
    luego cada coordenada del camino en orden, una por línea: “r c”.
  - Si no existe, imprime “NO_PATH”.
"""

import sys
import heapq

def leer_entrada():
    """
    Lee de stdin:
      - R, C (número de filas, columnas)
      - R líneas de mapa (“.” o “#”)
      - r_start c_start r_goal c_goal
    Retorna: R, C, mapa (lista de listas de chars), inicio (tupla), meta (tupla).
    """
    try:
        linea = sys.stdin.readline().strip().split()
        if len(linea) != 2:
            raise ValueError("Se esperaban 2 enteros en primera línea.")
        R, C = map(int, linea)
        if R <= 0 or C <= 0:
            raise ValueError("Rows y cols deben ser > 0.")
    except Exception as e:
        print(f"[ERROR] Lectura de dimensiones: {e}")
        sys.exit(1)

    mapa = []
    for i in range(R):
        fila = sys.stdin.readline().rstrip('\n')
        if len(fila) != C:
            print(f"[ERROR] Fila {i} debe tener {C} caracteres.")
            sys.exit(1)
        for ch in fila:
            if ch not in ('.', '#'):
                print(f"[ERROR] Carácter inválido en mapa: '{ch}'.")
                sys.exit(1)
        mapa.append(list(fila))

    try:
        coords = sys.stdin.readline().strip().split()
        if len(coords) != 4:
            raise ValueError("Se esperaban 4 enteros para inicio y meta.")
        r_start, c_start, r_goal, c_goal = map(int, coords)
        for val, limite in [(r_start, R), (r_goal, R), (c_start, C), (c_goal, C)]:
            if val < 0 or val >= limite:
                raise ValueError("Coordenada fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Lectura de inicio/meta: {e}")
        sys.exit(1)

    return R, C, mapa, (r_start, c_start), (r_goal, c_goal)

def heuristica(a, b):
    """
    Heurística: distancia de Manhattan entre a=(r1,c1) y b=(r2,c2).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def obtener_vecinos(pos, R, C, mapa):
    """
    Retorna lista de celdas vecinas libres (cuatro direcciones) desde pos=(r,c).
    """
    r, c = pos
    direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vecs = []
    for dr, dc in direcciones:
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and mapa[nr][nc] == '.':
            vecs.append((nr, nc))
    return vecs

def reconstruir_camino(prev, start, goal):
    """
    Reconstruye y retorna la lista de nodos desde start hasta goal usando el diccionario prev.
    Cada entrada prev[nodo] = nodo_previo.
    """
    camino = []
    actual = goal
    while actual != start:
        camino.append(actual)
        actual = prev.get(actual)
        if actual is None:
            # No hay camino
            return []
    camino.append(start)
    camino.reverse()
    return camino

def astar(R, C, mapa, start, goal):
    """
    Implementación de A*:
      - g_score[nodo]: costo desde start a nodo.
      - f_score[nodo]: g_score[nodo] + heurística(nodo, goal).
      - Usa un montón (heap) de tuplas (f_score, contador, nodo) para priorizar.
      - prev[nodo]: predecesor en mejor camino.
    Retorna lista de coordenadas del camino, o lista vacía si no existe.
    """
    INF = float('inf')
    g_score = {start: 0}
    f_score = {start: heuristica(start, goal)}
    prev = {}
    abierto = []
    contador = 0  # Para romper empates en heapq
    heapq.heappush(abierto, (f_score[start], contador, start))

    cerrados = set()

    while abierto:
        _, _, current = heapq.heappop(abierto)
        if current == goal:
            return reconstruir_camino(prev, start, goal)

        if current in cerrados:
            continue
        cerrados.add(current)

        for vecino in obtener_vecinos(current, R, C, mapa):
            if vecino in cerrados:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(vecino, INF):
                prev[vecino] = current
                g_score[vecino] = tentative_g
                f_score[vecino] = tentative_g + heuristica(vecino, goal)
                contador += 1
                heapq.heappush(abierto, (f_score[vecino], contador, vecino))

    # Si agotamos abierto sin llegar a goal
    return []

def main():
    """
    Flujo principal:
      1. Leer datos de entrada.
      2. Ejecutar A*.
      3. Si existe camino, imprimir longitud y coordenadas.
      4. Sino, imprimir “NO_PATH”.
    """
    R, C, mapa, start, goal = leer_entrada()
    path = astar(R, C, mapa, start, goal)
    if not path:
        print("NO_PATH")
    else:
        print(len(path) - 1)  # número de pasos
        for r, c in path:
            print(f"{r} {c}")

if __name__ == "__main__":
    main()
