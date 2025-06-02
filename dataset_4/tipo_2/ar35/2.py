#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de Kruskal para MST:
Misma funcionalidad pero con variables y comentarios modificados.
Lee de stdin: cantidad de vértices y aristas, luego aristas con peso.
Imprime peso total y lista de arcos del MST.
"""

import sys

class UF:
    """
    Estructura de conjuntos disjuntos (Union-Find) con path compression y unión por rango.
    """
    def __init__(self, size):
        # Inicializar cada elemento como su propio padre
        self.parent = [i for i in range(size)]
        # Inicializar rangos en cero
        self.rank = [0] * size

    def find_set(self, x):
        """
        Encuentra la raíz representativa de x y aplica path compression.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]

    def union_set(self, a, b):
        """
        Une los conjuntos que contienen a y b. Devuelve True si la unión fue exitosa,
        False si ya estaban en el mismo conjunto (para evitar ciclos).
        """
        root_a = self.find_set(a)
        root_b = self.find_set(b)
        if root_a == root_b:
            return False
        # Unir por rango para mantener el árbol equilibrado
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True

class Arc:
    """
    Representa un arco (arista) en el grafo: nodo_origen, nodo_destino y peso.
    """
    def __init__(self, src, dst, cost):
        self.src = src
        self.dst = dst
        self.cost = cost

def mst_kruskal(vertices, arcos):
    """
    Aplica el algoritmo de Kruskal:
      - Ordena la lista de arcos por peso ascendente.
      - Recorre arcos y usa UF para unir componentes si no forman ciclo.
      - Acumula arcos seleccionados en lista 'solucion'.
    Retorna (peso_mst, lista_mst_en_arcos).
    """
    sorted_arcs = sorted(arcos, key=lambda arc: arc.cost)
    uf = UF(vertices)
    total_mst_cost = 0
    solution = []

    for arc in sorted_arcs:
        # Si unir no formaría ciclo
        if uf.union_set(arc.src, arc.dst):
            total_mst_cost += arc.cost
            solution.append(arc)
        # Si ya tenemos suficientes arcos, podemos dejar de procesar
        if len(solution) == vertices - 1:
            break

    return total_mst_cost, solution

def cargar_datos():
    """
    Lee datos desde stdin:
      - Primera línea: V E (vértices, aristas).
      - E líneas siguientes: cada línea 'u v w' para un arco entre u y v con peso w.
    Valida rangos y pesos no negativos.
    Retorna (V, lista_de_arcos).
    """
    try:
        headers = sys.stdin.readline().strip().split()
        if len(headers) != 2:
            raise ValueError("Se esperaban dos enteros al inicio.")
        V = int(headers[0])
        E = int(headers[1])
    except Exception as ex:
        print(f"[ERROR] Lectura de cabecera inválida: {ex}")
        sys.exit(1)

    # Lista para almacenar arcos
    arcos = []
    for i in range(E):
        datos = sys.stdin.readline().strip().split()
        if len(datos) != 3:
            print("[ERROR] Cada línea de arco debe tener tres enteros.")
            sys.exit(1)
        u = int(datos[0])
        v = int(datos[1])
        w = int(datos[2])
        if u < 0 or u >= V or v < 0 or v >= V or w < 0:
            print("[ERROR] Valores inválidos en arista o peso negativo.")
            sys.exit(1)
        arcos.append(Arc(u, v, w))
    return V, arcos

def main():
    """
    1. Llamar a cargar_datos() para obtener V y lista de arcos.
    2. Ejecutar mst_kruskal() y obtener costo total y arcos seleccionados.
    3. Verificar conectividad y mostrar resultados.
    """
    V, arcos = cargar_datos()
    cost, mst_arcs = mst_kruskal(V, arcos)
    # Si no formamos MST completo (grafo desconectado)
    if len(mst_arcs) != V - 1:
        print("[ERROR] Grafo desconectado; MST no cubre todos los vértices.")
        return
    print(cost)
    for arc in mst_arcs:
        print(f"{arc.src} {arc.dst} {arc.cost}")

if __name__ == "__main__":
    main()
