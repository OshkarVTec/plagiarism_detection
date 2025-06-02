#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de ordenamiento topológico (Topological Sort) en un grafo dirigido.
Se usa DFS recursivo para detectar ciclos y construir el orden.
"""

import sys
import threading

sys.setrecursionlimit(10**7)


class Grafo:
    """
    Representa un grafo dirigido usando lista de adyacencia.
    """
    def __init__(self, numero_nodos):
        # Inicializamos una lista de listas vacías para cada nodo (0..numero_nodos-1)
        self.n = numero_nodos
        self.ady = [[] for _ in range(self.n)]

    def agregar_arista(self, u, v):
        """
        Agrega una arista dirigida desde u hasta v.
        """
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise IndexError("Nodo fuera de rango")
        self.ady[u].append(v)


def dfs_visit(grafo, nodo, estado, stack):
    """
    Visita recursiva de DFS para ordenamiento topológico.
    - estado[i] = 0 (no visitado), 1 (visitando), 2 (visitado completamente)
    - Si se detecta un vértice en estado=1, hay un ciclo.
    - Al terminar de explorar un nodo, se pone en stack.
    """
    estado[nodo] = 1  # Marcamos como en proceso
    for vecino in grafo.ady[nodo]:
        if estado[vecino] == 0:
            # Si aún no visitado, lo exploramos
            ciclo = dfs_visit(grafo, vecino, estado, stack)
            if ciclo:
                return True
        elif estado[vecino] == 1:
            # Encontramos un ciclo de regreso
            return True

    estado[nodo] = 2  # Marcamos como completamente visitado
    stack.append(nodo)
    return False


def ordenamiento_topologico(grafo):
    """
    Realiza ordenamiento topológico en el grafo.
    Retorna una lista con nodos en orden topológico si no hay ciclos,
    de lo contrario levanta una excepción.
    """
    estado = [0] * grafo.n
    pila = []
    for u in range(grafo.n):
        if estado[u] == 0:
            tiene_ciclo = dfs_visit(grafo, u, estado, pila)
            if tiene_ciclo:
                raise ValueError("El grafo contiene un ciclo; no existe orden topológico.")

    # La pila tiene nodos en orden inverso al topológico
    return pila[::-1]


def leer_grafo_entrada():
    """
    Lee de stdin:
      - Primera línea: n (número de nodos) y m (número de aristas).
      - Siguientes m líneas: cada par u v, arista u->v.
    """
    try:
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            raise ValueError("Se esperaban dos enteros en la primera línea.")
        n = int(parts[0])
        m = int(parts[1])
    except Exception as e:
        print(f"[ERROR] Entrada inválida: {e}")
        sys.exit(1)

    grafo = Grafo(n)
    for _ in range(m):
        line = sys.stdin.readline().strip()
        if not line:
            continue
        uv = line.split()
        if len(uv) != 2:
            print("[ERROR] Cada línea debe contener dos enteros.")
            sys.exit(1)
        u = int(uv[0])
        v = int(uv[1])
        try:
            grafo.agregar_arista(u, v)
        except IndexError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    return grafo


def main():
    """
    Punto de entrada:
    1. Lee el grafo de stdin.
    2. Ejecuta ordenamiento topológico.
    3. Imprime la secuencia resultante o mensaje de error.
    """
    grafo = leer_grafo_entrada()
    try:
        resultado = ordenamiento_topologico(grafo)
        print("Orden Topológico:", " ".join(map(str, resultado)))
    except ValueError as ve:
        print(f"[ERROR] {ve}")


if __name__ == "__main__":
    # Para evitar límites de recursión en DFS
    threading.Thread(target=main).start()
