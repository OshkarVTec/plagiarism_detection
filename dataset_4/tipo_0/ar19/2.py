#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web crawler recursivo que:
  - Toma una URL base y explora enlaces internos hasta cierta profundidad.
  - Solo sigue enlaces dentro del mismo dominio.
  - Imprime título de cada página visitada y su URL.
  - Evita visitar la misma página dos veces.
"""

import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import sys
import time

# Profundidad máxima de recursión
MAX_DEPTH = 2
# Tiempo de espera entre peticiones (segundos)
DELAY = 1

visited = set()  # Conjunto global para URLs ya visitadas

def is_same_domain(url, base_domain):
    """
    Verifica si la URL pertenece al mismo dominio que 'base_domain'.
    """
    try:
        domain = urlparse(url).netloc
        return domain == base_domain
    except Exception:
        return False

def crawl(url, base_domain, depth):
    """
    Función recursiva para:
    1. Descargar contenido HTML de 'url'.
    2. Extraer y mostrar título.
    3. Encontrar enlaces internos y recursar hasta depth < MAX_DEPTH.
    """
    if depth > MAX_DEPTH:
        return
    if url in visited:
        return
    visited.add(url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"[ERROR] No se pudo acceder a {url}: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Sin título"
    print(f"[Profundidad {depth}] {title} -> {url}")

    # Espera para no sobrecargar servidor
    time.sleep(DELAY)

    # Extraer todos los enlaces <a href="...">
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # Formar URL absoluta
        joined = urljoin(url, href)
        # Limitar a mismo dominio y no visitar protocolos extraños
        if joined.startswith("http") and is_same_domain(joined, base_domain):
            crawl(joined, base_domain, depth + 1)
