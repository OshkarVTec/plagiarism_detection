#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de BFS para calcular la distancia mínima (en número de aristas)
desde un vértice fuente a todos los demás en un grafo no dirigido no ponderado.
Lee desde stdin:
  - Primera línea: n m (n = número de vértices, m = número de aristas)
  - Siguientes m líneas: u v (arista no dirigida entre u y v)
  - Última línea: s (vértice fuente)
Imprime en stdout n líneas con formato:
  i d
donde d es la distancia mínima de s a i, o -1 si i no es alcanzable.
Se asume que los vértices están numerados de 0 a n-1.
"""

import sys
from collections import deque

def leer_grafo():
    """
    Lee n, m y la lista de aristas. Valida rangos y retorna n, lista de pares (u, v).
    """
    try:
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            raise ValueError("Se esperaban dos enteros en la primera línea (n, m).")
        n = int(parts[0])
        m = int(parts[1])
        if n <= 0 or m < 0:
            raise ValueError("n debe ser > 0, m >= 0.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida (n, m): {e}")
        sys.exit(1)

    aristas = []
    for i in range(m):
        linea = sys.stdin.readline().strip().split()
        if len(linea) != 2:
            print(f"[ERROR] Línea {i+2} inválida: se esperaban dos enteros.")
            sys.exit(1)
        try:
            u = int(linea[0])
            v = int(linea[1])
            if u < 0 or u >= n or v < 0 or v >= n:
                raise ValueError("Valores de vértices fuera de rango.")
        except Exception as e:
            print(f"[ERROR] Al leer arista en línea {i+2}: {e}")
            sys.exit(1)
        aristas.append((u, v))
    return n, aristas

def construir_ady(n, aristas):
    """
    Construye lista de adyacencia para grafo no dirigido:
      ady[u] contiene a v, ady[v] contiene a u.
    """
    ady = [[] for _ in range(n)]
    for (u, v) in aristas:
        ady[u].append(v)
        ady[v].append(u)
    return ady

def bfs_distancias(n, ady, fuente):
    """
    Realiza BFS desde 'fuente' para calcular distancias mínimas:
      - dist[i] = distancia en número de aristas de fuente a i, o -1 si no es alcanzable.
    Retorna lista dist[] de longitud n.
    """
    dist = [-1] * n
    dist[fuente] = 0
    queue = deque([fuente])

    while queue:
        u = queue.popleft()
        for v in ady[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def imprimir_distancias(dist):
    """
    Imprime cada vértice i y su distancia dist[i] en líneas separadas.
    """
    for i, d in enumerate(dist):
        print(f"{i} {d}")

def main():
    """
    Flujo principal:
      1. Leer grafo y lista de aristas.
      2. Leer vértice fuente.
      3. Construir adyacencia.
      4. Ejecutar BFS para distancias.
      5. Imprimir resultados.
    """
    n, aristas = leer_grafo()
    linea_fuente = sys.stdin.readline().strip()
    try:
        fuente = int(linea_fuente)
        if fuente < 0 or fuente >= n:
            raise ValueError("Fuente fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida de fuente: {e}")
        sys.exit(1)

    ady = construir_ady(n, aristas)
    dist = bfs_distancias(n, ady, fuente)
    imprimir_distancias(dist)

if __name__ == "__main__":
    main()
