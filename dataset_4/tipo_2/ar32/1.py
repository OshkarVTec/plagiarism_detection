#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de Dijkstra para grafos dirigidos con pesos no negativos.
Lee desde stdin:
  - Primera línea: n m (número de nodos, número de aristas)
  - Siguientes m líneas: u v w (arista de u a v con peso w)
  - Última línea: s (nodo fuente para calcular distancias)
Imprime en stdout la distancia mínima desde s a cada nodo (o 'INF' si inalcanzable).
"""

import sys
import heapq


class Grafo:
    """
    Grafo dirigido representado con lista de adyacencia.
    Cada entrada ady[u] es lista de tuplas (v, peso).
    """

    def __init__(self, num_nodos):
        if num_nodos <= 0:
            raise ValueError("El número de nodos debe ser positivo.")
        self.n = num_nodos
        self.ady = [[] for _ in range(self.n)]

    def agregar_arista(self, u, v, peso):
        """
        Agrega arista dirigida u -> v con peso dado.
        Verifica que los índices estén en el rango [0, n-1].
        """
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise IndexError("Nodo fuera de rango.")
        if peso < 0:
            raise ValueError("Peso debe ser no negativo.")
        self.ady[u].append((v, peso))


def dijkstra(grafo, fuente):
    """
    Ejecuta el algoritmo de Dijkstra desde el nodo 'fuente'.
    Retorna lista de distancias mínimas dist[], donde dist[i] es la distancia
    desde 'fuente' hasta i (o float('inf') si inalcanzable).
    """
    n = grafo.n
    # Inicializar distancias con infinito y fuente en 0
    dist = [float('inf')] * n
    dist[fuente] = 0

    # Cola de prioridad con pares (distancia_actual, nodo)
    # Se usa heapq para mantener el vértice con distancia mínima al frente
    pq = []
    heapq.heappush(pq, (0, fuente))

    # Mientras queden nodos por procesar en la cola
    while pq:
        dist_u, u = heapq.heappop(pq)
        # Si la distancia extraída es mayor que la almacenada, continuar
        if dist_u > dist[u]:
            continue

        # Recorrer todos los vecinos de u
        for v, peso_uv in grafo.ady[u]:
            # Si encontramos un camino más corto a v pasando por u
            if dist[u] + peso_uv < dist[v]:
                dist[v] = dist[u] + peso_uv
                heapq.heappush(pq, (dist[v], v))

    return dist


def leer_entrada():
    """
    Lee desde stdin:
      - Una línea con n m
      - m líneas con u v w
      - Una línea con s (fuente)
    Retorna (grafo, fuente).
    """
    try:
        primera = sys.stdin.readline().strip().split()
        if len(primera) != 2:
            raise ValueError("Se esperaban dos enteros en la primera línea.")
        n = int(primera[0])
        m = int(primera[1])
    except Exception as e:
        print(f"[ERROR] Lectura inválida de n y m: {e}")
        sys.exit(1)

    grafo = Grafo(n)

    # Leer cada arista
    for i in range(m):
        linea = sys.stdin.readline().strip().split()
        if len(linea) != 3:
            print("[ERROR] Cada línea de arista debe tener tres enteros: u v w.")
            sys.exit(1)
        try:
            u = int(linea[0])
            v = int(linea[1])
            w = int(linea[2])
            grafo.agregar_arista(u, v, w)
        except Exception as e:
            print(f"[ERROR] Al agregar arista: {e}")
            sys.exit(1)

    # Leer nodo fuente
    try:
        s_linea = sys.stdin.readline().strip()
        fuente = int(s_linea)
        if fuente < 0 or fuente >= n:
            raise ValueError("Fuente fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida de nodo fuente: {e}")
        sys.exit(1)

    return grafo, fuente


def main():
    """
    Punto de entrada del programa:
    1. Leer grafo y fuente de la entrada estándar.
    2. Ejecutar Dijkstra.
    3. Imprimir distancias.
    """
    grafo, fuente = leer_entrada()
    distancias = dijkstra(grafo, fuente)

    # Imprimir resultados: si dist[i] es infinito, mostrar 'INF'
    for i, d in enumerate(distancias):
        if d == float('inf'):
            print(f"{i}: INF")
        else:
            print(f"{i}: {d}")


if __name__ == "__main__":
    main()
