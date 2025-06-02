#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del algoritmo de Bellman-Ford para grafos dirigidos con peso (pueden ser negativos).
Lee desde stdin:
  - Primera línea: n m (n = número de vértices, m = número de aristas)
  - Siguientes m líneas: u v w (arista dirigida de u a v con peso w)
  - Última línea: s (nodo fuente)
Calcula la distancia mínima desde s a todos los vértices.
Si existe un ciclo de peso negativo alcanzable desde s, imprime "CICLO_NEGATIVO" y termina.
Sino, imprime en stdout n líneas:  
  i d  
donde i es el índice del vértice y d es la distancia (o "INF" si no es alcanzable).
"""

import sys

class Graph:
    """
    Grafo dirigido ponderado:
    - n: número de vértices.
    - edges: lista de tuplas (u, v, w) representando aristas u->v con peso w.
    """
    def __init__(self, n):
        if n <= 0:
            raise ValueError("Número de vértices debe ser positivo.")
        self.n = n
        self.edges = []

    def add_edge(self, u, v, w):
        """
        Agrega una arista a la lista de aristas.
        Verifica índices y peso.
        """
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise IndexError("Vértice fuera de rango.")
        # Peso puede ser negativo
        self.edges.append((u, v, w))


def bellman_ford(graph, source):
    """
    Ejecuta Bellman-Ford desde el nodo 'source'.
    Retorna lista dist[] con distancias mínimas o None si hay ciclo negativo.
    """
    n = graph.n
    # Inicializar distancias con infinito
    INF = float('inf')
    dist = [INF] * n
    dist[source] = 0

    # Relajación de aristas: repetir n-1 veces
    for i in range(n - 1):
        updated = False
        for (u, v, w) in graph.edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        # Si no hubo actualización en esta iteración, podemos detener el bucle
        if not updated:
            break

    # Verificar si existe ciclo de peso negativo
    for (u, v, w) in graph.edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            return None  # Indica ciclo negativo

    return dist


def read_input():
    """
    Lee la entrada desde stdin:
      - n m
      - m líneas con u v w
      - s (fuente)
    Retorna (graph, source).
    """
    try:
        first_line = sys.stdin.readline().strip().split()
        if len(first_line) != 2:
            raise ValueError("Se esperaban dos enteros en la primera línea (n m).")
        n = int(first_line[0])
        m = int(first_line[1])
    except Exception as e:
        print(f"[ERROR] Entrada inválida (n y m): {e}")
        sys.exit(1)

    graph = Graph(n)
    for i in range(m):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 3:
            print(f"[ERROR] Línea {i+2} inválida: se esperaban tres enteros (u v w).")
            sys.exit(1)
        try:
            u = int(parts[0])
            v = int(parts[1])
            w = int(parts[2])
            graph.add_edge(u, v, w)
        except Exception as e:
            print(f"[ERROR] Al procesar arista en línea {i+2}: {e}")
            sys.exit(1)

    try:
        source_line = sys.stdin.readline().strip()
        source = int(source_line)
        if source < 0 or source >= n:
            raise ValueError("Fuente fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida de fuente: {e}")
        sys.exit(1)

    return graph, source


def print_result(distances):
    """
    Imprime distancias en el formato:
      i d
    donde i es índice de vértice y d es la distancia o 'INF'.
    """
    for i, d in enumerate(distances):
        if d == float('inf'):
            print(f"{i} INF")
        else:
            print(f"{i} {d}")


def main():
    """
    Flujo principal:
      1. Leer grafo y fuente.
      2. Ejecutar bellman_ford.
      3. Si hay ciclo negativo, imprimir 'CICLO_NEGATIVO'.
      4. Sino, imprimir distancias.
    """
    graph, source = read_input()
    distances = bellman_ford(graph, source)
    if distances is None:
        print("CICLO_NEGATIVO")
    else:
        print_result(distances)


if __name__ == "__main__":
    main()
