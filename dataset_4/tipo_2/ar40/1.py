#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del algoritmo de Prim para encontrar el Árbol de Expansión Mínima (MST)
en un grafo no dirigido conectado con pesos positivos.  
Lee desde stdin:
  - Primera línea: n m (n = número de vértices, m = número de aristas)
  - Siguientes m líneas: u v w (arista no dirigida entre u y v con peso w)
Imprime en stdout:
  - Peso total del MST
  - Lista de aristas en el MST en el formato "u v w", una por línea.
Se asume que los vértices están numerados de 0 a n-1.
"""

import sys
import heapq

def leer_entrada():
    """
    Lee n y m, luego m aristas. Valida rangos y retorna n y lista de tuplas (u, v, w).
    """
    try:
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            raise ValueError("Se esperaban dos enteros en la primera línea.")
        n = int(parts[0])
        m = int(parts[1])
        if n <= 0 or m < 0:
            raise ValueError("n debe ser > 0 y m >= 0.")
    except Exception as e:
        print(f"[ERROR] Entrada inválida (n, m): {e}")
        sys.exit(1)

    aristas = []
    for i in range(m):
        linea = sys.stdin.readline().strip().split()
        if len(linea) != 3:
            print(f"[ERROR] Línea {i+2} inválida: se esperaban 3 enteros.")
            sys.exit(1)
        try:
            u = int(linea[0])
            v = int(linea[1])
            w = int(linea[2])
            if u < 0 or u >= n or v < 0 or v >= n or w < 0:
                raise ValueError("Valores fuera de rango o peso negativo.")
        except Exception as e:
            print(f"[ERROR] Al leer arista en línea {i+2}: {e}")
            sys.exit(1)
        aristas.append((u, v, w))
    return n, aristas

def construir_ady(n, aristas):
    """
    Construye lista de adyacencia para grafo no dirigido:
      ady[u] es lista de (peso, v), y ady[v] lista de (peso, u).
    """
    ady = [[] for _ in range(n)]
    for (u, v, w) in aristas:
        ady[u].append((w, v))
        ady[v].append((w, u))
    return ady

def prim(n, ady):
    """
    Ejecuta Prim a partir del vértice 0. Retorna (peso_total, lista_aristas_MST):
      - Se usa un min-heap (peso, u, v) para seleccionar la arista de menor peso
        que conecte al conjunto ya construido con el resto de vértices.
    """
    visitado = [False] * n
    peso_mst = 0
    mst_aristas = []
    heap = []

    # Iniciar desde el vértice 0
    visitado[0] = True
    for (w, v) in ady[0]:
        heapq.heappush(heap, (w, 0, v))

    while heap and len(mst_aristas) < n - 1:
        w, u, v = heapq.heappop(heap)
        if visitado[v]:
            continue
        # Incluir arista u-v en el MST
        visitado[v] = True
        peso_mst += w
        mst_aristas.append((u, v, w))
        # Añadir todas las aristas salientes de v
        for (w2, vec) in ady[v]:
            if not visitado[vec]:
                heapq.heappush(heap, (w2, v, vec))

    # Comprobar que se incluyeron n-1 aristas
    if len(mst_aristas) != n - 1:
        print("[ERROR] El grafo no está completamente conectado.")
        sys.exit(1)

    return peso_mst, mst_aristas

def imprimir_mst(peso, aristas_mst):
    """
    Imprime el peso total y luego cada arista del MST en formato "u v w".
    """
    print(peso)
    for (u, v, w) in aristas_mst:
        print(f"{u} {v} {w}")

def main():
    """
    Flujo principal:
      1. Leer n, aristas.
      2. Construir lista de adyacencia.
      3. Ejecutar Prim.
      4. Imprimir resultado.
    """
    n, aristas = leer_entrada()
    ady = construir_ady(n, aristas)
    peso_total, mst = prim(n, ady)
    imprimir_mst(peso_total, mst)

if __name__ == "__main__":
    main()
