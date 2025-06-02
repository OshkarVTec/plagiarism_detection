#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión modificada (Tipo 2) de Dijkstra:
Misma lógica básica, con variables renombradas, comentarios reescritos y
orden de bloques levemente distinto. Lee de stdin:
  - Número de vértices y aristas
  - Luego, cada arista con (origen destino peso)
  - Finalmente, nodo de inicio para la ej. de Dijkstra
Imprime distancia mínima desde el origen a cada vértice (o 'NO_INF' si no alcanzable).
"""

import sys
import heapq


class DirigidoPonderado:
    """
    Grafo dirigido con pesos no negativos. Estructura: adyacencia[v] = lista de (vecino, costo).
    """
    def __init__(self, total_vertices):
        if total_vertices <= 0:
            raise ValueError("Total de vértices debe ser mayor que cero.")
        self.total_vertices = total_vertices
        # Inicializar lista de adyacencia vacía para cada vértice
        self.ady = [[] for _ in range(self.total_vertices)]

    def insertar_arista(self, origen, destino, costo):
        """
        Inserta arco desde -> destino con peso 'costo'.
        Valida rangos de vértices y no admite pesos negativos.
        """
        if origen < 0 or origen >= self.total_vertices or destino < 0 or destino >= self.total_vertices:
            raise IndexError("Vértice fuera de rango.")
        if costo < 0:
            raise ValueError("Costo de la arista debe ser no negativo.")
        self.ady[origen].append((destino, costo))


def dijkstra_heap(grafo, inicio):
    """
    Algoritmo de Dijkstra usando heapq:
      - dist_vec[v] = distancia mínima desde 'inicio' a v (inf si no visitado)
      - pq mantiene pares (distancia_actual, vértice) y extrae el menor
      - Relaja aristas hasta procesar todos los alcanzables
    Retorna la lista de distancias dist_vec.
    """
    n = grafo.total_vertices
    # Inicializar distancias a infinito
    dist_vec = [float('inf')] * n
    dist_vec[inicio] = 0

    # Cola de prioridad (min-heap) con tuplas (distancia_acumulada, vértice)
    pq = [(0, inicio)]

    while pq:
        dist_u, u = heapq.heappop(pq)
        # Si la tupla extraída tiene distancia mayor que la actual, ignorar
        if dist_u > dist_vec[u]:
            continue

        # Explorar todas las aristas que salen de u
        for (v, peso_uv) in grafo.ady[u]:
            nueva_dist = dist_vec[u] + peso_uv
            # Si encontramos un camino más corto a v
            if nueva_dist < dist_vec[v]:
                dist_vec[v] = nueva_dist
                heapq.heappush(pq, (nueva_dist, v))

    return dist_vec


def leer_grafo_stdin():
    """
    Lee la estructura de grafo desde stdin:
      - Línea inicial: V E (vértices y aristas)
      - E líneas siguientes: cada 'a b c' denota arista a->b con peso c
      - Última línea: nodo inicio para Dijkstra
    Retorna (grafo, nodo_inicio).
    """
    # Primer línea: total de vértices y cantidad de arcos
    try:
        primeras = sys.stdin.readline().strip().split()
        if len(primeras) != 2:
            raise ValueError("Formato inválido para vértices/aristas.")
        V = int(primeras[0])
        E = int(primeras[1])
    except Exception as e:
        print(f"[ERROR] Error al leer V y E: {e}")
        sys.exit(1)

    g = DirigidoPonderado(V)

    # Leer cada arco
    for i in range(E):
        datos = sys.stdin.readline().strip().split()
        if len(datos) != 3:
            print("[ERROR] Cada arco debe tener tres valores: origen destino peso.")
            sys.exit(1)
        try:
            a = int(datos[0])
            b = int(datos[1])
            c = int(datos[2])
            g.insertar_arista(a, b, c)
        except Exception as ex:
            print(f"[ERROR] Al insertar arista: {ex}")
            sys.exit(1)

    # Leer nodo de inicio
    try:
        linea_inicio = sys.stdin.readline().strip()
        inicio = int(linea_inicio)
        if inicio < 0 or inicio >= V:
            raise ValueError("Nodo de inicio fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Error al leer nodo inicio: {e}")
        sys.exit(1)

    return g, inicio


def main():
    """
    1. Leer grafo y nodo inicial con leer_grafo_stdin()
    2. Calcular distancias con dijkstra_heap()
    3. Mostrar resultados: 'VERTEX: dist' o 'VERTEX: NO_INF' si inalcanzable
    """
    grafo, start_vertex = leer_grafo_stdin()
    distancias = dijkstra_heap(grafo, start_vertex)

    # Formato de salida
    for idx, d in enumerate(distancias):
        if d == float('inf'):
            print(f"{idx}: NO_INF")
        else:
            print(f"{idx}: {d}")


if __name__ == '__main__':
    main()
