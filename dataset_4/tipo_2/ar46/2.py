#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de Algoritmo Genético para TSP:
Misma funcionalidad: aproximar ruta óptima para n ciudades.  
Lee desde stdin:
  - Línea 1: n (número de ciudades)
  - Siguientes n líneas: “x y” coordenadas de cada ciudad
Parámetros internos:
  - TAMA_POB = 100
  - GEN_MAX = 200
  - PROB_MUT = 0.02
  - TAM_TOUR = 5
Salida:
  - Distancia total de la mejor ruta (6 decimales)
  - n+1 índices de ciudades (ciclo completo)
"""

import sys
import random
import math

TAMA_POB = 100
GEN_MAX = 200
PROB_MUT = 0.02
TAM_TOUR = 5

def cargar_ciudades():
    """
    Lee n y luego n coordenadas. Retorna lista de (x, y).
    """
    try:
        linea = sys.stdin.readline().strip()
        if not linea:
            raise ValueError("No se especificó n.")
        n = int(linea)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 ciudades.")
    except Exception as e:
        print(f"[ERROR] Lectura de n: {e}")
        sys.exit(1)

    lista = []
    for i in range(n):
        partes = sys.stdin.readline().strip().split()
        if len(partes) != 2:
            print(f"[ERROR] Línea {i+1} inválida: se requieren 2 valores.")
            sys.exit(1)
        try:
            x, y = float(partes[0]), float(partes[1])
        except:
            print(f"[ERROR] Coordenadas no numéricas en línea {i+1}.")
            sys.exit(1)
        lista.append((x, y))
    return lista

def euclidea(p1, p2):
    """
    Calcula la distancia euclidiana entre p1 y p2.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def costo_ruta(ruta, ciudades):
    """
    Dada ruta (lista de índices), calcula costo total (incluye salto final al origen).
    """
    total = 0.0
    for i in range(len(ruta)-1):
        total += euclidea(ciudades[ruta[i]], ciudades[ruta[i+1]])
    total += euclidea(ciudades[ruta[-1]], ciudades[ruta[0]])
    return total

def crear_poblacion(ciudades):
    """
    Genera población aleatoria de permutaciones de 0..n-1.
    """
    n = len(ciudades)
    base = list(range(n))
    pop = []
    for _ in range(TAMA_POB):
        indiv = base[:]
        random.shuffle(indiv)
        pop.append(indiv)
    return pop

def torneo(pop, costos):
    """
    Selección por torneo: elige TAM_TOUR individuos al azar, devuelve copia del mejor (menor costo).
    """
    comp = random.sample(range(len(pop)), TAM_TOUR)
    mejor = comp[0]
    for idx in comp[1:]:
        if costos[idx] < costos[mejor]:
            mejor = idx
    return pop[mejor][:]

def cruza_orden(m1, m2):
    """
    Order Crossover simplificado:
    - Elige dos puntos aleatorios c_i y c_j, copia segmento de m1.
    - Luego rellena con ciudades de m2 en orden que no estén ya copiadas.
    """
    n = len(m1)
    c_i, c_j = sorted(random.sample(range(n), 2))
    hijo = [-1] * n
    for i in range(c_i, c_j+1):
        hijo[i] = m1[i]
    pos = (c_j + 1) % n
    for i in range(n):
        idx = (c_j + 1 + i) % n
        if m2[idx] not in hijo:
            hijo[pos] = m2[idx]
            pos = (pos + 1) % n
    return hijo

def mutar_swap(indiv):
    """
    Para cada posición, con probabilidad PROB_MUT, intercambia con otra posición aleatoria.
    """
    for i in range(len(indiv)):
        if random.random() < PROB_MUT:
            j = random.randrange(len(indiv))
            indiv[i], indiv[j] = indiv[j], indiv[i]

def ga_tsp_solver(ciudades):
    """
    Algoritmo Genético principal:
    1. Crear población.
    2. Por cada generación:
       a) Calcular costos de cada indiv.
       b) Mantener mejor global.
       c) Generar nueva pop con torneo, cruza_orden y mutación.
    3. Devolver mejor ruta y su costo.
    """
    pobl = crear_poblacion(ciudades)
    mejor_global = None
    mejor_costo = float('inf')

    for gen in range(GEN_MAX):
        costos = [costo_ruta(indiv, ciudades) for indiv in pobl]
        for i, cost in enumerate(costos):
            if cost < mejor_costo:
                mejor_costo = cost
                mejor_global = pobl[i][:]

        nueva = []
        while len(nueva) < TAMA_POB:
            m1 = torneo(pobl, costos)
            m2 = torneo(pobl, costos)
            hijo = cruza_orden(m1, m2)
            mutar_swap(hijo)
            nueva.append(hijo)
        pobl = nueva

    return mejor_global, mejor_costo

def main():
    """
    1. Leer ciudades.
    2. Ejecutar ga_tsp_solver.
    3. Imprimir costo (6 decimales) y secuencia de índices de ciudades + índice 0 final.
    """
    ciudades = cargar_ciudades()
    ruta, costo = ga_tsp_solver(ciudades)
    print(f"{costo:.6f}")
    for idx in ruta:
        print(idx)
    print(ruta[0])

if __name__ == "__main__":
    main()
