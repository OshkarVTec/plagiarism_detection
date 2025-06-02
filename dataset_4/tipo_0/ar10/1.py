#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web scraper asíncrono que:
1. Lee URLs desde un archivo 'urls.txt' (una URL por línea).
2. Descarga cada página web de forma concurrente usando aiohttp.
3. Extrae el contenido de la etiqueta <title> con BeautifulSoup.
4. Guarda en 'titles.csv' el par (URL, Título).
"""

import asyncio
import aiohttp
import csv
import os
import sys
from bs4 import BeautifulSoup

# Nombre del archivo con la lista de URLs (una URL por línea)
URLS_FILE = "urls.txt"
# Nombre del archivo CSV donde se guardarán los resultados
OUTPUT_CSV = "titles.csv"
# Número máximo de peticiones simultáneas
MAX_CONCURRENT_REQUESTS = 10
# Timeout global para cada petición (en segundos)
REQUEST_TIMEOUT = 15


async def fetch(session, url, semaphore):
    """
    Descarga el contenido HTML de la URL proporcionada usando aiohttp.
    Se limita la concurrencia mediante un semáforo.
    """
    async with semaphore:
        try:
            # Realizar la petición GET con timeout
            async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
                # Verificar que el estatus HTTP sea 200 OK
                if response.status == 200:
                    html = await response.text()
                    return html
                else:
                    print(f"[ERROR] {url} devolvió estatus HTTP {response.status}")
                    return None
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] La petición a {url} excedió los {REQUEST_TIMEOUT} segundos.")
            return None
        except aiohttp.ClientError as e:
            print(f"[CLIENT ERROR] No se pudo descargar {url}: {e}")
            return None


def extract_title(html, url):
    """
    Dado el contenido HTML, extrae el texto de la etiqueta <title>.
    Si no se encuentra, retorna 'Sin título'.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        else:
            return "Sin título"
    except Exception as e:
        print(f"[ERROR] Al parsear HTML de {url}: {e}")
        return "Sin título"


async def process_url(session, url, semaphore):
    """
    Gestiona la descarga y extracción de título para una única URL.
    Retorna una tupla (url, título).
    """
    html = await fetch(session, url, semaphore)
    if html:
        title = extract_title(html, url)
    else:
        title = "Error al descargar"
    return url, title

