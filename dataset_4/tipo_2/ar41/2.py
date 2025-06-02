#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 del algoritmo BFS para distancias mínimas en grafo no dirigido.
Lee de stdin:
  - Línea 1: V E (V vértices, E aristas)
  - E líneas: a b (arista no dirigida entre a y b)
  - Última línea: origen (vértice de inicio)
Imprime en stdout V líneas con:
  i d
donde d es la distancia mínima desde origen a i, o -1 si no se alcanza.
"""

import sys
from collections import deque

def leer_estructura():
    """
    Lee V y E de stdin, luego E pares de vértices.
    Valida y retorna V y la lista de pares (a, b).
    """
    try:
        cab = sys.stdin.readline().strip().split()
        if len(cab) != 2:
            raise ValueError("Primera línea debe tener dos enteros.")
        V = int(cab[0])
        E = int(cab[1])
        if V <= 0 or E < 0:
            raise ValueError("V > 0 y E >= 0.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida (V, E): {e}")
        sys.exit(1)

    lista_arcos = []
    for idx in range(E):
        partes = sys.stdin.readline().strip().split()
        if len(partes) != 2:
            print(f"[ERROR] Línea {idx+2} inválida: se requieren dos enteros.")
            sys.exit(1)
        try:
            a = int(partes[0])
            b = int(partes[1])
            if a < 0 or a >= V or b < 0 or b >= V:
                raise ValueError("Vértices fuera de rango.")
        except Exception as ex:
            print(f"[ERROR] Al leer arista en línea {idx+2}: {ex}")
            sys.exit(1)
        lista_arcos.append((a, b))
    return V, lista_arcos

def generar_ady(V, arcos):
    """
    Construye la lista de adyacencia para grafo no dirigido:
      - Para cada arista (a, b), se añade b a ady[a] y a a ady[b].
    """
    adyacencia = [[] for _ in range(V)]
    for (a, b) in arcos:
        adyacencia[a].append(b)
        adyacencia[b].append(a)
    return adyacencia

def calcular_bfs(V, ady, inicio):
    """
    Ejecuta BFS desde 'inicio' para obtener distancias mínimas:
      - distancias[i] guarda el número de pasos de inicio a i, o -1 si no se llega.
    Retorna lista distancias.
    """
    distancias = [-1] * V
    distancias[inicio] = 0
    cola = deque([inicio])

    while cola:
        nodo = cola.popleft()
        for vecino in ady[nodo]:
            if distancias[vecino] == -1:
                distancias[vecino] = distancias[nodo] + 1
                cola.append(vecino)
    return distancias

def mostrar_distancias(dist):
    """
    Imprime cada vértice i y su distancia dist[i] en líneas separadas.
    """
    for idx, d in enumerate(dist):
        print(f"{idx} {d}")

def main():
    """
    Flujo principal:
      1. Leer V, arcos con leer_estructura().
      2. Leer vértice origen.
      3. Construir adyacencia con generar_ady().
      4. Ejecutar BFS con calcular_bfs().
      5. Imprimir resultados con mostrar_distancias().
    """
    V, lista_arcos = leer_estructura()
    linea_origen = sys.stdin.readline().strip()
    try:
        origen = int(linea_origen)
        if origen < 0 or origen >= V:
            raise ValueError("Origen fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida de vértice origen: {e}")
        sys.exit(1)

    ady = generar_ady(V, lista_arcos)
    distancias = calcular_bfs(V, ady, origen)
    mostrar_distancias(distancias)

if __name__ == "__main__":
    main()
