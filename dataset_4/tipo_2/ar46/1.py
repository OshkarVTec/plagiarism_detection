#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de un Algoritmo Genético (GA) básico para aproximar la solución
al problema del Agente Viajero (TSP).  
Lee desde stdin:
  - Primera línea: n (número de ciudades)
  - Siguientes n líneas: cada línea “x y” con coordenadas (enteros o reales)
Parámetros internos (se pueden ajustar en el código):
  - Tamaño de población (POP_SIZE)
  - Número de generaciones (NUM_GEN)
  - Probabilidad de mutación (MUT_RATE)
  - Número de individuos para selección por torneo (TOUR_SIZE)
Al final, imprime en stdout:
  - Primera línea: distancia total de la mejor ruta encontrada (con 6 decimales).
  - Siguientes n+1 líneas: índice de ciudad en el orden del ciclo (empezando y terminando en 0).
    Se utilizan índices 0..n-1 según el orden de entrada.
"""

import sys
import random
import math
import copy

# Parámetros GA (ajustables)
POP_SIZE = 100
NUM_GEN = 200
MUT_RATE = 0.02
TOUR_SIZE = 5

def leer_ciudades():
    """
    Lee n y luego n coordenadas (x y). Retorna lista de tuplas (x, y).
    """
    try:
        linea = sys.stdin.readline().strip()
        if not linea:
            raise ValueError("No se recibió n.")
        n = int(linea)
        if n < 2:
            raise ValueError("Debe haber al menos 2 ciudades.")
    except Exception as e:
        print(f"[ERROR] Leer número de ciudades: {e}")
        sys.exit(1)

    ciudades = []
    for i in range(n):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            print(f"[ERROR] Fila {i+1}: se requieren 2 coordenadas.")
            sys.exit(1)
        try:
            x, y = float(parts[0]), float(parts[1])
        except:
            print(f"[ERROR] Coordenadas no válidas en línea {i+1}.")
            sys.exit(1)
        ciudades.append((x, y))
    return ciudades

def distancia(a, b):
    """
    Distancia euclidiana entre puntos a=(x1,y1) y b=(x2,y2).
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

def calcular_distancia_ruta(ruta, ciudades):
    """
    Dada ruta como lista de índices de ciudades, calcula la distancia total del ciclo (volver al inicio).
    """
    total = 0.0
    n = len(ruta)
    for i in range(n - 1):
        total += distancia(ciudades[ruta[i]], ciudades[ruta[i+1]])
    # volver al punto de inicio
    total += distancia(ciudades[ruta[-1]], ciudades[ruta[0]])
    return total

def generar_poblacion(ciudades):
    """
    Genera población inicial: lista de permutaciones aleatorias de índices [0..n-1].
    """
    n = len(ciudades)
    poblacion = []
    base = list(range(n))
    for _ in range(POP_SIZE):
        individuo = base[:]
        random.shuffle(individuo)
        poblacion.append(individuo)
    return poblacion

def seleccion_torneo(poblacion, fitnesses):
    """
    Selección por torneo: elige TOUR_SIZE individuos al azar, retorna el de menor fitness (mejor).
    """
    aspirantes = random.sample(range(len(poblacion)), TOUR_SIZE)
    mejor = aspirantes[0]
    for idx in aspirantes[1:]:
        if fitnesses[idx] < fitnesses[mejor]:
            mejor = idx
    return poblacion[mejor][:]  # devolver copia del mejor cromosoma

def cruza_pm(parent1, parent2):
    """
    Crossover Order 1 (PMX simplificado):
    Selecciona dos puntos de corte y cruza manteniendo orden relativo.
    """
    n = len(parent1)
    c1, c2 = sorted(random.sample(range(n), 2))
    hijo = [-1] * n
    # Copiar segmento medio
    for i in range(c1, c2 + 1):
        hijo[i] = parent1[i]
    # Rellenar resto con elementos de parent2 en orden
    pos = (c2 + 1) % n
    for i in range(n):
        idx = (c2 + 1 + i) % n
        if parent2[idx] not in hijo:
            hijo[pos] = parent2[idx]
            pos = (pos + 1) % n
    return hijo

def mutacion_swap(individuo):
    """
    Mutación por swap: con probabilidad MUT_RATE, elige dos posiciones aleatorias y las intercambia.
    """
    for i in range(len(individuo)):
        if random.random() < MUT_RATE:
            j = random.randrange(len(individuo))
            individuo[i], individuo[j] = individuo[j], individuo[i]

def ga_tsp(ciudades):
    """
    Algoritmo Genético para TSP:
      1. Generar población inicial.
      2. Para cada generación:
         a) Calcular fitness (distancia) de cada individuo.
         b) Generar nueva población:
            - Seleccionar pares con torneo, cruzar con cruza_pm, mutar con mutación_swap.
         c) Reemplazo completo.
      3. Retornar mejor individuo final y su distancia.
    """
    poblacion = generar_poblacion(ciudades)
    mejor_ind = None
    mejor_dist = float('inf')

    for gen in range(NUM_GEN):
        fitnesses = [calcular_distancia_ruta(ind, ciudades) for ind in poblacion]
        nueva_pop = []

        # Actualizar mejor global
        for i, fit in enumerate(fitnesses):
            if fit < mejor_dist:
                mejor_dist = fit
                mejor_ind = poblacion[i][:]

        # Crear nueva población
        while len(nueva_pop) < POP_SIZE:
            madre = seleccion_torneo(poblacion, fitnesses)
            padre = seleccion_torneo(poblacion, fitnesses)
            hijo = cruza_pm(madre, padre)
            mutacion_swap(hijo)
            nueva_pop.append(hijo)

        poblacion = nueva_pop

    return mejor_ind, mejor_dist

def main():
    """
    1. Leer coordenadas de ciudades.
    2. Ejecutar ga_tsp().
    3. Imprimir distancia con 6 decimales y secuencia de índices (n líneas + 1 con index 0).
    """
    ciudades = leer_ciudades()
    mejor_ruta, mejor_dist = ga_tsp(ciudades)
    print(f"{mejor_dist:.6f}")
    for idx in mejor_ruta:
        print(idx)
    # repetir índice 0 para cerrar ciclo
    print(mejor_ruta[0])

if __name__ == "__main__":
    main()
