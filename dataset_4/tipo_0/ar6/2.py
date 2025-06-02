#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descargador de archivos concurrente usando hilos (threading).
Lee una lista de URLs desde un archivo de texto y las descarga en paralelo.
"""

import threading
import requests
import os
import sys
import time

# Número máximo de hilos simultáneos
MAX_THREADS = 5

# Carpeta donde se guardarán las descargas
DOWNLOAD_DIR = "downloads"


def download_file(url, index):
    """
    Descarga el contenido de la URL dada y lo guarda en un archivo local.
    El nombre del archivo se genera a partir del índice proporcionado.
    """
    try:
        print(f"[Hilo {index}] Iniciando descarga: {url}")
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        # Obtener el nombre de archivo de la URL o asignar uno por defecto
        filename = url.split("/")[-1] or f"file_{index}"
        # Asegurarse de que la carpeta de descargas exista
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[Hilo {index}] Descarga finalizada: {filename}")
    except requests.RequestException as e:
        print(f"[Hilo {index}] Error al descargar {url}: {e}")


def worker_thread(queue, lock):
    """
    Función que ejecuta cada hilo.
    Mientras haya URLs en la cola, las descarga.
    """
    while True:
        lock.acquire()
        if not queue:
            lock.release()
            break
        url, idx = queue.pop(0)
        lock.release()
        download_file(url, idx)

print()
print()
print()
print()