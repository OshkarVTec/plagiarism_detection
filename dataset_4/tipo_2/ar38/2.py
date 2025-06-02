#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión Tipo 2 del algoritmo de Bellman-Ford:
Misma funcionalidad para encontrar distancias mínimas en grafo dirigido con posibles pesos negativos.
Lee desde stdin:
  - Primera línea: V E (V = cantidad de vértices, E = cantidad de arcos)
  - Siguientes E líneas: a b c (arco de a a b con costo c)
  - Última línea: origen (nodo fuente)
Si existe ciclo negativo alcanzable, imprime "NEGATIVO", sino imprime V líneas con:
  i dist_i (o "INFINITO" si no es alcanzable)
"""

import sys

class DirigidoConPeso:
    """
    Grafo dirigido con pesos negativos permitidos.
    Atributos:
      - V: número de vértices.
      - lista_arcos: lista de tuplas (u, v, peso).
    """
    def __init__(self, V):
        if V <= 0:
            raise ValueError("Debe haber al menos un vértice.")
        self.V = V
        self.lista_arcos = []

    def agrega_arco(self, u, v, peso):
        """
        Agrega arco dirigido u->v con peso 'peso'.
        Valida índices y peso.
        """
        if u < 0 or u >= self.V or v < 0 or v >= self.V:
            raise IndexError("Vértice inválido.")
        # Peso puede ser negativo
        self.lista_arcos.append((u, v, peso))


def bellman_ford_algo(grafo, inicio):
    """
    Ejecuta Bellman-Ford desde 'inicio'.
    Retorna arreglo dist donde dist[i] es distancia mínima o None si hay ciclo negativo.
    """
    n = grafo.V
    INF = float('inf')
    dist = [INF] * n
    dist[inicio] = 0

    # Repetir V-1 iteraciones de relajación
    for _ in range(n - 1):
        cambio = False
        for (u, v, costo) in grafo.lista_arcos:
            if dist[u] != INF and dist[u] + costo < dist[v]:
                dist[v] = dist[u] + costo
                cambio = True
        if not cambio:
            break

    # Verificar ciclo negativo
    for (u, v, costo) in grafo.lista_arcos:
        if dist[u] != INF and dist[u] + costo < dist[v]:
            return None

    return dist


def leer_grafo_stdin():
    """
    Lee del stdin:
      - Línea 1: V E
      - E líneas: u v w
      - Última línea: nodo_origen
    Retorna (grafo, nodo_origen).
    """
    try:
        data = sys.stdin.readline().strip().split()
        if len(data) != 2:
            raise ValueError("Se requieren dos enteros en la primera línea (V E).")
        V = int(data[0])
        E = int(data[1])
    except Exception as e:
        print(f"[ERROR] Formato inválido de V y E: {e}")
        sys.exit(1)

    grafo = DirigidoConPeso(V)
    for i in range(E):
        partes = sys.stdin.readline().strip().split()
        if len(partes) != 3:
            print(f"[ERROR] Línea {i+2} inválida: se necesitan tres enteros.")
            sys.exit(1)
        try:
            u = int(partes[0])
            v = int(partes[1])
            w = int(partes[2])
            grafo.agrega_arco(u, v, w)
        except Exception as ex:
            print(f"[ERROR] Al procesar arco en línea {i+2}: {ex}")
            sys.exit(1)

    try:
        origen_line = sys.stdin.readline().strip()
        origen = int(origen_line)
        if origen < 0 or origen >= V:
            raise ValueError("Nodo origen fuera de rango.")
    except Exception as e:
        print(f"[ERROR] Lectura inválida de nodo origen: {e}")
        sys.exit(1)

    return grafo, origen


def imprimir_distancias(distancias):
    """
    Imprime las distancias en formato:
      i valor_i
    donde valor_i es el valor o "INFINITO".
    """
    for idx, d in enumerate(distancias):
        if d == float('inf'):
            print(f"{idx} INFINITO")
        else:
            print(f"{idx} {d}")


def main():
    """
    1. Leer grafo y nodo origen con leer_grafo_stdin().
    2. Ejecutar bellman_ford_algo().
    3. Si devuelve None, imprimir "NEGATIVO".
    4. Sino, llamar a imprimir_distancias().
    """
    grafo, origen = leer_grafo_stdin()
    distancias = bellman_ford_algo(grafo, origen)
    if distancias is None:
        print("NEGATIVO")
    else:
        imprimir_distancias(distancias)


if __name__ == "__main__":
    main()
