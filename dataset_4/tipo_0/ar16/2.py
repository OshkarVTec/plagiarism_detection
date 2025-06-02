#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliente HTTP asíncrono para obtener múltiples URLs en paralelo.
Muestra código de estado HTTP y tamaño del contenido de cada respuesta.
"""

import asyncio
import aiohttp
import argparse
import sys
import time

# Número máximo de peticiones simultáneas
MAX_CONCURRENT = 10
# Timeout para cada petición (en segundos)
REQUEST_TIMEOUT = 10

async def fetch_url(session, url, semaphore):
    """
    Descarga la URL usando aiohttp, respetando el semáforo de concurrencia.
    Retorna tupla (url, status_code, content_length).
    """
    async with semaphore:
        try:
            async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
                status = response.status
                content = await response.read()
                length = len(content)
                print(f"[OK] {url} -> Status: {status}, Tamaño: {length} bytes")
                return (url, status, length)
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] La petición a {url} excedió {REQUEST_TIMEOUT} segundos.")
            return (url, None, 0)
        except Exception as e:
            print(f"[ERROR] Falló al obtener {url}: {e}")
            return (url, None, 0)

async def run_client(urls):
    """
    Crea sesión aiohttp y dispara tareas asíncronas para cada URL.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch_url(session, url, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

def load_urls(file_path):
    """
    Carga lista de URLs desde un archivo de texto (una URL por línea).
    Omite líneas vacías o que empiecen con '#'.
    """

    urls = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls

def save_results(results, output_file):
    """
    Guarda resultados en un archivo CSV con columnas:
    URL, status_code, content_length
    """
    try:
        import csv
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["URL", "Status", "ContentLength"])
            for url, status, length in results:
                writer.writerow([url, status if status else "ERROR", length])
        print(f"[INFO] Resultados guardados en '{output_file}'.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar CSV: {e}")
