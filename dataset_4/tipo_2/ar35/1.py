#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de Kruskal para encontrar el Árbol de Expansión Mínima (MST)
en un grafo no dirigido con pesos no negativos.
Lee desde stdin:
  - Primera línea: n m (n = número de vértices, m = número de aristas)
  - Siguientes m líneas: u v w (arista entre u y v con peso w)
Imprime en stdout:
  - Peso total del MST
  - Lista de aristas en el MST en formato "u v w" una por línea
"""

import sys

class DisjointSet:
    """
    Estructura Union-Find (Disjoint Set Union - DSU) con path compression
    y union by rank.
    """
    def __init__(self, n):
        # Cada vértice es su propio padre inicialmente
        self.parent = [i for i in range(n)]
        # Rango aproximado para cada conjunto
        self.rank = [0] * n

    def find(self, x):
        """
        Encuentra el representante (root) del conjunto al que pertenece x.
        Aplica path compression.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        Une los conjuntos que contienen x e y.
        Usa union by rank para minimizar la altura del árbol.
        """
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        # Adjuntar árbol de menor rango bajo la raíz de mayor rango
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
        return True

class Edge:
    """
    Representa una arista de un grafo: origen, destino y peso.
    """
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

def kruskal(n, edges):
    """
    Aplica el algoritmo de Kruskal para encontrar el MST en un grafo con n vértices
    y lista de aristas 'edges'.
    Retorna (peso_total, lista_aristas_MST).
    """
    # Ordenar aristas por peso ascendente
    sorted_edges = sorted(edges, key=lambda e: e.w)
    dsu = DisjointSet(n)
    mst_weight = 0
    mst_edges = []

    for edge in sorted_edges:
        # Si unir las componentes no forma ciclo
        if dsu.union(edge.u, edge.v):
            mst_weight += edge.w
            mst_edges.append(edge)
        # Si ya tenemos n-1 aristas, podemos detenernos
        if len(mst_edges) == n - 1:
            break

    return mst_weight, mst_edges

def read_input():
    """
    Lee la entrada estándar:
      - Una línea con n m
      - m líneas con u v w
    Retorna (n, lista_de_aristas).
    """
    try:
        line = sys.stdin.readline().strip().split()
        if len(line) != 2:
            raise ValueError("Formato inválido: se esperaban dos enteros en la primera línea.")
        n = int(line[0])
        m = int(line[1])
    except Exception as e:
        print(f"[ERROR] Entrada inválida: {e}")
        sys.exit(1)

    edges = []
    for i in range(m):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 3:
            print("[ERROR] Cada línea de arista debe tener tres enteros.")
            sys.exit(1)
        u = int(parts[0])
        v = int(parts[1])
        w = int(parts[2])
        # Verificar rangos
        if u < 0 or u >= n or v < 0 or v >= n or w < 0:
            print("[ERROR] Valores fuera de rango o peso negativo.")
            sys.exit(1)
        edges.append(Edge(u, v, w))

    return n, edges

def main():
    """
    1. Leer número de vértices 'n' y lista de aristas.
    2. Ejecutar Kruskal para obtener MST.
    3. Imprimir peso total y aristas del MST.
    """
    n, edges = read_input()
    total_weight, mst = kruskal(n, edges)
    # Si no se encontró MST completo (grafo no conectado)
    if len(mst) != n - 1:
        print("[ERROR] El grafo no es conexo; no existe MST completo.")
        return
    print(total_weight)
    for e in mst:
        print(f"{e.u} {e.v} {e.w}")

if __name__ == "__main__":
    main()
