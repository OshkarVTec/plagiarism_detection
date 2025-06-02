#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de DFS para contar el número de componentes conexas
en un grafo no dirigido.  
Lee desde stdin:
  - Primera línea: n m (n = número de vértices, m = número de aristas)
  - Siguientes m líneas: u v (arista no dirigida entre u y v)
Imprime en stdout:
  - Un entero: el número de componentes conexas.
Se asume que los vértices están numerados de 0 a n-1.
"""

import sys
sys.setrecursionlimit(10**7)

def leer_grafo():
    """
    Lee n y m, luego m aristas. Valida que los valores estén en rango.
    Retorna n y lista de pares (u, v).
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
                raise ValueError("Vértices fuera de rango.")
        except Exception as e:
            print(f"[ERROR] Al leer arista en línea {i+2}: {e}")
            sys.exit(1)
        aristas.append((u, v))
    return n, aristas

def construir_ady(n, aristas):
    """
    Construye lista de adyacencia para grafo no dirigido:
      ady[u] incluye a v, y ady[v] incluye a u.
    """
    ady = [[] for _ in range(n)]
    for (u, v) in aristas:
        ady[u].append(v)
        ady[v].append(u)
    return ady

def dfs(u, ady, visitado):
    """
    DFS recursivo desde el vértice u, marcando como visitado a todos
    los vértices alcanzables en su componente.
    """
    visitado[u] = True
    for v in ady[u]:
        if not visitado[v]:
            dfs(v, ady, visitado)

def contar_componentes(n, ady):
    """
    Recorre todos los vértices y, si uno no está visitado,
    inicia una DFS para marcar toda su componente. Cuenta las componentes.
    """
    visitado = [False] * n
    contador = 0
    for i in range(n):
        if not visitado[i]:
            dfs(i, ady, visitado)
            contador += 1
    return contador

def main():
    """
    Flujo principal:
      1. Leer grafo y lista de aristas.
      2. Construir adyacencia.
      3. Contar componentes conexas con contar_componentes().
      4. Imprimir el resultado.
    """
    n, aristas = leer_grafo()
    ady = construir_ady(n, aristas)
    num_cc = contar_componentes(n, ady)
    print(num_cc)

if __name__ == "__main__":
    main()
