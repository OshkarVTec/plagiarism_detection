#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de Prim:
Misma funcionalidad para hallar MST en grafo no dirigido, con renombrado de funciones,
variables y comentarios reescritos.  
Lee de stdin:
  - Línea 1: N M (N vértices, M aristas)
  - M líneas: a b c (arista no dirigida entre a y b con peso c)
Imprime en stdout:
  - Costo total del MST
  - Luego M' líneas (M' = N-1) con "a b c".
"""

import sys
import heapq

def leer_grafo():
    """
    Lee N y M de stdin y luego M arcos (u v w).
    Valida rangos y retorna N y lista de (origen, destino, costo).
    """
    try:
        cabecera = sys.stdin.readline().strip().split()
        if len(cabecera) != 2:
            raise ValueError("Formato inválido: se esperaban dos enteros.")
        N = int(cabecera[0])
        M = int(cabecera[1])
        if N <= 0 or M < 0:
            raise ValueError("N debe ser > 0 y M >= 0.")
    except Exception as e:
        print(f"[ERROR] Entrada inválida (N, M): {e}")
        sys.exit(1)

    arcos = []
    for idx in range(M):
        partes = sys.stdin.readline().strip().split()
        if len(partes) != 3:
            print(f"[ERROR] Línea {idx+2} inválida: se requieren tres enteros.")
            sys.exit(1)
        try:
            a = int(partes[0])
            b = int(partes[1])
            c = int(partes[2])
            if a < 0 or a >= N or b < 0 or b >= N or c < 0:
                raise ValueError("Valores fuera de rango o costo negativo.")
        except Exception as ex:
            print(f"[ERROR] Al leer arco en línea {idx+2}: {ex}")
            sys.exit(1)
        arcos.append((a, b, c))
    return N, arcos

def crear_adyacencia(N, arcos):
    """
    Construye lista de adyacencia para grafo no dirigido:
      - Para cada arco (a, b, c), se añade (c, b) a ady[a] y (c, a) a ady[b].
    """
    ady = [[] for _ in range(N)]
    for (a, b, c) in arcos:
        ady[a].append((c, b))
        ady[b].append((c, a))
    return ady

def prim_mst(N, ady):
    """
    Ejecuta Prim desde el nodo 0:
      - usa un min-heap de tuplas (costo, u, v) que representan arcos candidatas.
      - inicia marcando 0 como visitado y añadiendo sus arcos.
      - mientras el heap no esté vacío y falten arcos para completar MST:
          * extrae arista de menor costo que conecte a un vértice no visitado.
          * marca vértice como visitado, acumula costo, agrega arista al resultado,
            y empuja al heap los arcos salientes de ese vértice.
      - si no se obtienen N-1 arcos, el grafo no es conexo.
    Retorna (costo_total, lista_arcos_MST).
    """
    visit = [False] * N
    visit[0] = True
    mst_costo = 0
    mst_lista = []
    heap = []

    # Insertar arcos iniciales desde 0
    for (peso, v) in ady[0]:
        heapq.heappush(heap, (peso, 0, v))

    while heap and len(mst_lista) < N - 1:
        costo, u, v = heapq.heappop(heap)
        if visit[v]:
            continue
        visit[v] = True
        mst_costo += costo
        mst_lista.append((u, v, costo))
        for (peso2, w) in ady[v]:
            if not visit[w]:
                heapq.heappush(heap, (peso2, v, w))

    if len(mst_lista) != N - 1:
        print("[ERROR] Grafo desconectado; no se puede formar MST completo.")
        sys.exit(1)

    return mst_costo, mst_lista

def imprimir_resultados(costo, lista_mst):
    """
    Muestra en stdout:
      - Primera línea: costo total del MST.
      - Cada línea siguiente: "u v w" para cada arista en MST.
    """
    print(costo)
    for (u, v, w) in lista_mst:
        print(f"{u} {v} {w}")

def main():
    """
    Flujo principal:
      1. Leer N, arcos con leer_grafo().
      2. Construir adyacencia con crear_adyacencia().
      3. Ejecutar prim_mst().
      4. Imprimir con imprimir_resultados().
    """
    N, arcos = leer_grafo()
    ady = crear_adyacencia(N, arcos)
    costo_tot, mst_list = prim_mst(N, ady)
    imprimir_resultados(costo_tot, mst_list)

if __name__ == "__main__":
    main()
