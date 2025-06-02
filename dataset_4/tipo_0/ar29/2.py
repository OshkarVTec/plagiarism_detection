#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encuentra todos los números primos en un rango [2, N] utilizando procesamiento paralelo.
- Divide el rango en pedazos y asigna a varios procesos.
- Cada proceso comprueba primalidad por división hasta sqrt(n).
- Al final, combina resultados y los ordena.
- Imprime la cuenta total y guarda la lista en un archivo de texto.
"""

import math
import sys
import argparse
from multiprocessing import Process, Queue, cpu_count

def is_prime(n):
    """
    Verifica si n es primo: prueba divisores impares hasta sqrt(n).
    Asume n > 1.
    """
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True

def worker(start, end, queue):
    """
    Proceso trabajador: revisa todos los números en [start, end) y pone los primos en la cola.
    """
    primes = []
    for num in range(start, end):
        if num > 1 and is_prime(num):
            primes.append(num)
    queue.put(primes)

def parallel_primes(n, num_workers):
    """
    Divide el rango [2, n+1) en num_workers segmentos casi iguales.
    Lanza procesos y recoge resultados por colas.
    Retorna lista de primos ordenada.
    """
    queue = Queue()
    processes = []
    # Calcular tamaño de segmento
    segment_size = (n - 1) // num_workers + 1
    for i in range(num_workers):
        start = 2 + i * segment_size
        end = min(2 + (i + 1) * segment_size, n + 1)
        p = Process(target=worker, args=(start, end, queue))
        processes.append(p)
        p.start()

    all_primes = []
    for p in processes:
        primes_segment = queue.get()
        all_primes.extend(primes_segment)

    for p in processes:
        p.join()

    return sorted(all_primes)

def save_primes(primes, output_file):
    """
    Guarda la lista de primos en un archivo de texto, uno por línea.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for p in primes:
                f.write(f"{p}\n")
        print(f"[INFO] Primos guardados en '{output_file}'.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar archivo: {e}")
