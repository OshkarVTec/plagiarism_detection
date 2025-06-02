#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de un Algoritmo Genético (AG) para maximizar la función:
    f(x) = x * sin(10πx) + 1 (en rango [0,1])
Pasos:
  1. Representación binaria de la variable x en genes de longitud 16.
  2. Inicialización aleatoria de población.
  3. Evaluación de aptitud (fitness) según f(x).
  4. Selección por torneo, cruce (crossover) de un punto y mutación bit a bit.
  5. Iterar por un número dado de generaciones e imprimir mejor individuo.
"""

import random
import math
import sys

GENE_LENGTH = 16  # Bits para representar variable x (resolución)
POPULATION_SIZE = 50
GENERATIONS = 100
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01

def individual():
    """
    Genera un individuo (string binario) de longitud GENE_LENGTH.
    """
    return "".join(random.choice("01") for _ in range(GENE_LENGTH))

def decode(chromosome):
    """
    Decodifica el cromosoma binario a un valor x en [0, 1].
    """
    integer_value = int(chromosome, 2)
    max_int = 2**GENE_LENGTH - 1
    x = integer_value / max_int
    return x

def fitness(chromosome):
    """
    Función de aptitud: f(x) = x * sin(10πx) + 1
    Rango de x es [0,1].
    """
    x = decode(chromosome)
    return x * math.sin(10 * math.pi * x) + 1

def generate_population(size):
    """
    Crea una población inicial con 'size' individuos aleatorios.
    """
    return [individual() for _ in range(size)]

def tournament_selection(population, fitnesses):
    """
    Selección por torneo: elige TOURNAMENT_SIZE individuos aleatorios
    y retorna el mejor (máximo fitness).
    """
    participants = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    participants.sort(key=lambda x: x[1], reverse=True)
    return participants[0][0]

def crossover(parent1, parent2):
    """
    Cruce de un punto: intercambia bits entre dos padres con probabilidad CROSSOVER_RATE.
    Retorna dos hijos.
    """
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENE_LENGTH - 2)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(chromosome):
    """
    Mutación bit a bit: invierte cada bit con probabilidad MUTATION_RATE.
    """
    new_chromo = []
    for bit in chromosome:
        if random.random() < MUTATION_RATE:
            new_chromo.append('1' if bit == '0' else '0')
        else:
            new_chromo.append(bit)
    return "".join(new_chromo)

def evolve(population):
    """
    Realiza un ciclo de evolución:
      1. Calcula aptitudes.
      2. Nuevo conjunto por selección, cruce y mutación.
      3. Retorna nueva población y mejor fitness de esta generación.
    """
    fitnesses = [fitness(ind) for ind in population]
    new_population = []

    # Elitismo: conservar el mejor individuo
    best_idx = fitnesses.index(max(fitnesses))
    best_individual = population[best_idx]
    best_fit = fitnesses[best_idx]
    new_population.append(best_individual)

    # Generar resto de la población
    while len(new_population) < POPULATION_SIZE:
        # Selección
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)
        # Cruce
        child1, child2 = crossover(p1, p2)
        # Mutación
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    return new_population, best_individual, best_fit
