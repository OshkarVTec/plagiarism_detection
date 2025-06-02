#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión modificada (Tipo 2) del ordenamiento topológico.
Mismos pasos lógicos que el original, pero renombrado de variables, reescritura de comentarios
y reordenamientos leves en la estructura.
"""

import sys
import threading

sys.setrecursionlimit(10**7)


class Dirigido:
    """
    Grafo dirigido: cada nodo mantiene lista de sucesores.
    """
    def __init__(self, total_nodos):
        self.total = total_nodos
        # Creamos lista de adyacencia vacía para cada índice
        self.vecinos = [[] for _ in range(self.total)]

    def anyadir_arco(self, desde, hacia):
        """
        Inserta arco dirigido desde -> hacia.
        """
        if desde < 0 or desde >= self.total or hacia < 0 or hacia >= self.total:
            raise IndexError("Nodo inválido")
        self.vecinos[desde].append(hacia)


def explorar(u, grafo, marca, resultado):
    """
    Exploración DFS:
    - marca[i] = 0 (sin visitar), 1 (procesando), 2 (terminado)
    - Si se halla vuelta a un nodo con marca=1, hay ciclo.
    - Una vez lista la expansión de u, se añade a resultado.
    """
    marca[u] = 1  # Marcamos como en proceso
    for v in grafo.vecinos[u]:
        if marca[v] == 0:
            if explorar(v, grafo, marca, resultado):
                return True
        elif marca[v] == 1:
            # Ciclo detectado
            return True

    marca[u] = 2  # Marcado como completamente procesado
    resultado.append(u)
    return False


def toposort(grafo):
    """
    Orden topológico:
    - inicializa vector de marcas
    - recorre todos los nodos que siguen sin procesar
    - invoca DFS y comprueba ciclos
    - devuelve la lista inversa para el orden correcto
    """
    marcas = [0] * grafo.total
    pila = []
    for nodo in range(grafo.total):
        if marcas[nodo] == 0:
            hay_ciclo = explorar(nodo, grafo, marcas, pila)
            if hay_ciclo:
                raise ValueError("El grafo tiene ciclos, no es posible ordenar topológicamente.")
    # Invertimos la lista acumulada
    orden = pila[::-1]
    return orden


def cargar_grafo():
    """
    Lee de la entrada estándar:
    primera línea: N M, número de nodos y arcos
    luego M líneas con pares (u v) que representan arcos u -> v
    """
    try:
        line = sys.stdin.readline().strip()
        partes = line.split()
        if len(partes) != 2:
            raise ValueError("Formato inválido en primera línea.")
        N = int(partes[0])
        M = int(partes[1])
    except Exception as e:
        print(f"[ERROR] Entrada incorrecta: {e}")
        sys.exit(1)

    G = Dirigido(N)
    for _ in range(M):
        linea = sys.stdin.readline().strip()
        if not linea:
            continue
        uv = linea.split()
        if len(uv) != 2:
            print("[ERROR] Cada arco debe definirse con dos enteros.")
            sys.exit(1)
        u = int(uv[0])
        v = int(uv[1])
        try:
            G.anyadir_arco(u, v)
        except IndexError as ex:
            print(f"[ERROR] {ex}")
            sys.exit(1)
    return G


def main():
    """
    Flujo principal:
    1. Carga el grafo de stdin con cargar_grafo().
    2. Llama a toposort() para obtener lista ordenada.
    3. Imprime el resultado o el error si existe ciclo.
    """
    grafo_entero = cargar_grafo()
    try:
        orden_final = toposort(grafo_entero)
        print("Secuencia Topológica:", " ".join(str(x) for x in orden_final))
    except ValueError as err:
        print(f"[ERROR] {err}")


if __name__ == '__main__':
    # Ejecutar en hilo para permitir recursión profunda
    threading.Thread(target=main).start()
