#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de DFS para hallar el número de componentes conexas en
un grafo no dirigido.  
Lee de stdin:
  - Línea 1: N M (N vértices, M aristas)
  - M líneas siguientes: a b (arista no dirigida entre a y b)
Imprime en stdout un entero: el número de componentes conexas.
"""

import sys
sys.setrecursionlimit(10**7)

def obtener_grafo():
    """
    Lee N y M, valida y luego lee M pares (a, b).
    Retorna N y lista_arcos de pares de vértices.
    """
    try:
        cab = sys.stdin.readline().strip().split()
        if len(cab) != 2:
            raise ValueError("Primera línea debe tener dos enteros.")
        N = int(cab[0])
        M = int(cab[1])
        if N <= 0 or M < 0:
            raise ValueError("N debe ser > 0 y M >= 0.")
    except Exception as e:
        print(f"[ERROR] Entrada inválida (N, M): {e}")
        sys.exit(1)

    lista_arcos = []
    for idx in range(M):
        partes = sys.stdin.readline().strip().split()
        if len(partes) != 2:
            print(f"[ERROR] Línea {idx+2} inválida: se requieren dos enteros.")
            sys.exit(1)
        try:
            a = int(partes[0])
            b = int(partes[1])
            if a < 0 or a >= N or b < 0 or b >= N:
                raise ValueError("Vértices fuera de rango.")
        except Exception as ex:
            print(f"[ERROR] Al procesar arista en línea {idx+2}: {ex}")
            sys.exit(1)
        lista_arcos.append((a, b))
    return N, lista_arcos

def armar_ady(N, lista_arcos):
    """
    Crea la lista de adyacencia para grafo no dirigido:
      para cada arco (a, b), adj[a].append(b) y adj[b].append(a).
    """
    adj = [[] for _ in range(N)]
    for (a, b) in lista_arcos:
        adj[a].append(b)
        adj[b].append(a)
    return adj

def dfs_explorar(u, adj, seen):
    """
    Marca todos los vértices alcanzables desde u mediante DFS recursivo.
    """
    seen[u] = True
    for w in adj[u]:
        if not seen[w]:
            dfs_explorar(w, adj, seen)

def calcular_componentes(N, adj):
    """
    Recorre cada vértice y, si no está marcado en seen, inicia DFS para
    marcar su componente. Devuelve la cantidad de componentes.
    """
    seen = [False] * N
    cont = 0
    for i in range(N):
        if not seen[i]:
            dfs_explorar(i, adj, seen)
            cont += 1
    return cont

def main():
    """
    Flujo principal:
      1. Leer N y lista de arcos con obtener_grafo().
      2. Construir lista de adyacencia con armar_ady().
      3. Calcular componentes con calcular_componentes().
      4. Imprimir resultado.
    """
    N, lista_arcos = obtener_grafo()
    adj = armar_ady(N, lista_arcos)
    num_componentes = calcular_componentes(N, adj)
    print(num_componentes)

if __name__ == "__main__":
    main()
