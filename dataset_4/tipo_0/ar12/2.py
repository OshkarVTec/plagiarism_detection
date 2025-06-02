#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de recursos del sistema usando psutil.
Recolecta uso de CPU, memoria, disco y red en intervalos regulares y escribe
los datos en un archivo CSV.
"""

import psutil
import csv
import time
import os
from datetime import datetime

# Archivo de salida para registrar métricas
OUTPUT_CSV = "system_monitor.csv"
# Intervalo (en segundos) entre cada medición
INTERVALO_MONITOREO = 5

def initialize_csv(file_path):
    """
    Crea (o sobrescribe) el archivo CSV con encabezados.
    """
    encabezados = [
        "timestamp",
        "cpu_percent",
        "memory_used_mb",
        "memory_total_mb",
        "disk_used_gb",
        "disk_total_gb",
        "net_bytes_sent_kb",
        "net_bytes_recv_kb"
    ]
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(encabezados)
    print(f"[INFO] Archivo CSV inicializado: {file_path}")

def monitor_system(interval, file_path):
    """
    Bucle principal de monitoreo:
    1. Recoge métricas de CPU, memoria, disco y red.
    2. Escribe una línea en el CSV cada intervalo.
    3. Se detiene al presionar Ctrl+C.
    """
    # Inicializar CSV al inicio
    initialize_csv(file_path)

    try:
        print(f"[INFO] Iniciando monitoreo cada {interval} segundos. Presiona Ctrl+C para detener.")
        # Variables iniciales para calcular diferencia de red
        prev_counters = psutil.net_io_counters()
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_percent = psutil.cpu_percent(interval=None)
            virtual_mem = psutil.virtual_memory()
            memory_used_mb = virtual_mem.used / (1024 ** 2)
            memory_total_mb = virtual_mem.total / (1024 ** 2)

            disk_usage = psutil.disk_usage("/")
            disk_used_gb = disk_usage.used / (1024 ** 3)
            disk_total_gb = disk_usage.total / (1024 ** 3)

            # Contadores de red totales
            net_counters = psutil.net_io_counters()
            bytes_sent_kb = (net_counters.bytes_sent - prev_counters.bytes_sent) / 1024
            bytes_recv_kb = (net_counters.bytes_recv - prev_counters.bytes_recv) / 1024
            prev_counters = net_counters

            # Escribir en CSV
            with open(file_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    f"{cpu_percent:.2f}",
                    f"{memory_used_mb:.2f}",
                    f"{memory_total_mb:.2f}",
                    f"{disk_used_gb:.2f}",
                    f"{disk_total_gb:.2f}",
                    f"{bytes_sent_kb:.2f}",
                    f"{bytes_recv_kb:.2f}"
                ])

            # Mostrar en consola (opcional)
            print(f"[{timestamp}] CPU: {cpu_percent:.2f}% | Memoria usada: {memory_used_mb:.2f}MB/{memory_total_mb:.2f}MB | "
                  f"Disco usado: {disk_used_gb:.2f}GB/{disk_total_gb:.2f}GB | Red -> Sent: {bytes_sent_kb:.2f}KB, Recv: {bytes_recv_kb:.2f}KB")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[INFO] Monitoreo detenido por el usuario.")
