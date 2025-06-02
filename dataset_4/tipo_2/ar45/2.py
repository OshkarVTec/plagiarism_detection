    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión Tipo 2 del algoritmo A* en una grilla:
Lee de stdin:
  - Fila 1: filas (F) y columnas (C)
  - Siguientes F líneas: cada una de longitud C, con “.” o “#”
  - Última línea: r_inicio c_inicio r_meta c_meta
Movimientos posibles en las 4 direcciones, costo = 1 por paso. Heurística: Manhattan.
Salida:
  - Si hay ruta: imprimir primero el número de movimientos,
    luego cada coordenada “r c” en orden.
  - Si no, imprimir “NO_PATH”.
"""

import sys
import heapq

def leer_entrada_2():
    """
    Lee:
      - F y C
      - F líneas del mapa (.” libres, # paredes)
      - Coordenadas de inicio y meta
    Retorna F, C, grid, inicio, meta.
    """
    try:
        datos = sys.stdin.readline().strip().split()
        if len(datos) != 2:
            raise ValueError("Necesito 2 números en la primera línea.")
        F, C = map(int, datos)
        if F <= 0 or C <= 0:
            raise ValueError("Filas y columnas deben ser mayores que cero.")
    except Exception as e:
        print(f"[ERROR] Leer dimensiones: {e}")
        sys.exit(1)

    grid = []
    for i in range(F):
        linea = sys.stdin.readline().rstrip('\n')
        if len(linea) != C:
            print(f"[ERROR] La fila {i} debe tener {C} caracteres.")
            sys.exit(1)
        for caract in linea:
            if caract not in ('.', '#'):
                print(f"[ERROR] Carácter inválido en mapa: {caract}")
                sys.exit(1)
        grid.append(list(linea))

    try:
        coords = sys.stdin.readline().strip().split()
        if len(coords) != 4:
            raise ValueError("Se requieren 4 enteros para inicio y meta.")
        r0, c0, r1, c1 = map(int, coords)
        for val, lim in [(r0, F), (r1, F), (c0, C), (c1, C)]:
            if val < 0 or val >= lim:
                raise ValueError("Coordenadas fuera de límites.")
    except Exception as e:
        print(f"[ERROR] Leer inicio/meta: {e}")
        sys.exit(1)

    return F, C, grid, (r0, c0), (r1, c1)

def manhattan(u, v):
    """
    Calcula la distancia Manhattan entre u=(r2,c2) y v=(r1,c1).
    """
    return abs(u[0] - v[0]) + abs(u[1] - v[1])

def vecinos_libres(pos, F, C, grid):
    """
    A partir de pos=(r,c), devuelve lista de vecinos válidos (.”).
    Movimientos: arriba, abajo, izquierda, derecha.
    """
    r, c = pos
    movs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    resultado = []
    for dr, dc in movs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < F and 0 <= nc < C and grid[nr][nc] == '.':
            resultado.append((nr, nc))
    return resultado

def reconstruir(prev_map, inicio, meta):
    """
    Usa prev_map[nodo] = predecesor para reconstruir ruta desde inicio a meta.
    Devuelve lista de nodos en orden.
    """
    ruta = []
    actual = meta
    while actual != inicio:
        ruta.append(actual)
        actual = prev_map.get(actual)
        if actual is None:
            return []
    ruta.append(inicio)
    ruta.reverse()
    return ruta

def busqueda_astar(F, C, grid, inicio, meta):
    """
    Implementación principal de A*:
      - g_cost[nodo]: costo real desde inicio hasta nodo.
      - f_cost[nodo] = g_cost[nodo] + heurística(nodo, meta).
      - Se usa un heap con tuplas (f_cost, contador, nodo) para seleccionar.
      - prev_map almacena el padre de cada nodo para reconstruir.
    Retorna la lista de nodos del camino o lista vacía si no hay ruta.
    """
    INFINITO = float('inf')
    g_cost = {inicio: 0}
    f_cost = {inicio: manhattan(inicio, meta)}
    prev_map = {}
    abierto = []
    etiqueta = 0
    heapq.heappush(abierto, (f_cost[inicio], etiqueta, inicio))
    cerrados = set()

    while abierto:
        _, _, actual = heapq.heappop(abierto)
        if actual == meta:
            return reconstruir(prev_map, inicio, meta)
        if actual in cerrados:
            continue
        cerrados.add(actual)
        for nb in vecinos_libres(actual, F, C, grid):
            if nb in cerrados:
                continue
            nuevo_g = g_cost[actual] + 1
            if nuevo_g < g_cost.get(nb, INFINITO):
                prev_map[nb] = actual
                g_cost[nb] = nuevo_g
                f_cost[nb] = nuevo_g + manhattan(nb, meta)
                etiqueta += 1
                heapq.heappush(abierto, (f_cost[nb], etiqueta, nb))
    return []

def main():
    """
    1. Leer datos con leer_entrada_2()
    2. Llamar a busqueda_astar()
    3. Si no hay camino, imprimir “NO_PATH”. Si lo hay:
       - Imprimir número de pasos (len(ruta)-1)
       - Luego cada “r c” en ruta.
    """
    F, C, grid, inicio, meta = leer_entrada_2()
    ruta = busqueda_astar(F, C, grid, inicio, meta)
    if not ruta:
        print("NO_PATH")
    else:
        print(len(ruta) - 1)
        for r, c in ruta:
            print(f"{r} {c}")

if __name__ == "__main__":
    main()
