#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimación del valor de π usando simulación de Monte Carlo.
Genera puntos aleatorios dentro de un cuadrado unitario y cuenta cuántos
caen dentro del cuarto de círculo inscrito.
"""

import random
import math
import time
import argparse
import sys

def estimate_pi(num_samples):
    """
    Realiza 'num_samples' puntos aleatorios en el cuadrado [0,1]x[0,1].
    Cuenta cuántos puntos caen dentro del cuarto de círculo x^2 + y^2 <= 1.
    Retorna la aproximación de π basada en la proporción.
    """
    inside_count = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside_count += 1
    pi_approx = (inside_count / num_samples) * 4
    return pi_approx

def experiment_variance(sample_sizes):
    """
    Ejecuta varias estimaciones de π con diferentes tamaños de muestra.
    Retorna una lista de tuplas (n, π_estimado, error_absoluto).
    """
    results = []
    for n in sample_sizes:
        start = time.time()
        pi_est = estimate_pi(n)
        elapsed = time.time() - start
        error = abs(math.pi - pi_est)
        results.append((n, pi_est, error, elapsed))
    return results

def print_results(results):
    """
    Imprime los resultados de la simulación de Monte Carlo en formato tabulado.
    """
    print(f"{'N':>10} | {'π_estimado':>12} | {'Error abs':>10} | {'Tiempo (s)':>10}")
    print("-" * 52)
    for n, pi_est, error, elapsed in results:
        print(f"{n:>10} | {pi_est:>12.6f} | {error:>10.6f} | {elapsed:>10.4f}")
    """
    Imprime los resultados de la simulación de Monte Carlo en formato tabulado.
    """
    print(f"{'N':>10} | {'π_estimado':>12} | {'Error abs':>10} | {'Tiempo (s)':>10}")
    print("-" * 52)
    for n, pi_est, error, elapsed in results:
        print(f"{n:>10} | {pi_est:>12.6f} | {error:>10.6f} | {elapsed:>10.4f}")
    """
    Imprime los resultados de la simulación de Monte Carlo en formato tabulado.
    """
    print(f"{'N':>10} | {'π_estimado':>12} | {'Error abs':>10} | {'Tiempo (s)':>10}")
    print("-" * 52)
    for n, pi_est, error, elapsed in results:
        print(f"{n:>10} | {pi_est:>12.6f} | {error:>10.6f} | {elapsed:>10.4f}")
    """
    Imprime los resultados de la simulación de Monte Carlo en formato tabulado.
    """
    print(f"{'N':>10} | {'π_estimado':>12} | {'Error abs':>10} | {'Tiempo (s)':>10}")
    print("-" * 52)
    for n, pi_est, error, elapsed in results:
        print(f"{n:>10} | {pi_est:>12.6f} | {error:>10.6f} | {elapsed:>10.4f}")

