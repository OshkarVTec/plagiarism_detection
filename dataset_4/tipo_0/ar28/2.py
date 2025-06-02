#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación de una cola M/M/1 (llegadas y servicios exponenciales):
  - Parámetros: tasa de llegada (lambda), tasa de servicio (mu), tiempo total de simulación.
  - Genera tiempos de llegada y servicio usando distribución exponencial.
  - Mantiene reloj de eventos (próxima llegada o fin de servicio).
  - Calcula métricas: tiempo promedio en cola, longitud promedio de la cola, utilización del servidor.
"""

import random
import math
import sys
import argparse

def exponential(rate):
    """
    Retorna un número aleatorio según distribución exponencial con parámetro `rate`.
    """
    return -math.log(1.0 - random.random()) / rate

def simulate_mm1(lambda_rate, mu_rate, sim_time):
    """
    Simulación básica de cola M/M/1:
      - Eventos: llegada, fin de servicio.
      - Variables de estado: reloj, número en sistema, próximo fin de servicio.
      - Acumula estadísticas para calcular promedios.
    Retorna diccionario con métricas clave.
    """
    # Inicialización
    clock = 0.0
    next_arrival = exponential(lambda_rate)
    next_departure = float('inf')
    num_in_system = 0

    # Estadísticas
    area_num_in_system = 0.0
    last_event_time = 0.0
    total_delay = 0.0
    num_served = 0

    while clock < sim_time:
        # Determinar próximo evento
        if next_arrival < next_departure:
            event_time = next_arrival
            clock = event_time
            area_num_in_system += num_in_system * (clock - last_event_time)
            num_in_system += 1
            last_event_time = clock

            # Programar próxima llegada
            next_arrival = clock + exponential(lambda_rate)

            # Si servidor estaba ocioso, iniciar servicio inmediatamente
            if num_in_system == 1:
                next_departure = clock + exponential(mu_rate)
        else:
            event_time = next_departure
            clock = event_time
            area_num_in_system += num_in_system * (clock - last_event_time)
            num_in_system -= 1
            last_event_time = clock

            total_delay += (clock - (next_departure - exponential(mu_rate)))
            num_served += 1

            if num_in_system > 0:
                next_departure = clock + exponential(mu_rate)
            else:
                next_departure = float('inf')

    # Cálculo de métricas
    avg_num_in_system = area_num_in_system / sim_time
    utilization = (area_num_in_system - 0) / sim_time * (1 / mu_rate)  # aproximación
    avg_delay = total_delay / num_served if num_served > 0 else 0.0

    return {
        'avg_num_in_system': avg_num_in_system,
        'utilization': utilization,
        'avg_delay': avg_delay,
        'num_served': num_served
    }

