#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web scraper básico:
  - Lee desde stdin una URL (http o https).
  - Descarga la página usando requests.
  - Analiza el HTML con BeautifulSoup (lxml).
  - Extrae todos los títulos de nivel 2 (<h2>) y enlaces (<a href="...">).
  - Imprime en stdout:
      * Primero todos los textos de <h2>, uno por línea, precedidos por "H2: ".
      * Luego todos los enlaces absolutos (hace join con la URL base si es relativo), uno por línea, "LINK: url".
Se requieren las librerías: requests, bs4.
"""

import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def leer_url():
    """
    Lee una línea con la URL. Valida que comience con http:// o https://.
    """
    url = sys.stdin.readline().strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        print("[ERROR] URL inválida. Debe comenzar con http:// o https://")
        sys.exit(1)
    return url

def descargar_pagina(url):
    """
    Descarga el contenido de la URL y retorna el texto HTML.
    Maneja errores HTTP u otros.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[ERROR] Al descargar {url}: {e}")
        sys.exit(1)

def analizar_html(html, base_url):
    """
    Usa BeautifulSoup para parsear HTML.
    Extrae:
      - Lista de textos de todas las etiquetas <h2>.
      - Lista de enlaces: obtiene atributo href de cada <a>.
        Si es relativo, lo convierte a absoluto con urljoin.
    Retorna (h2_texts, enlaces).
    """
    soup = BeautifulSoup(html, "lxml")
    h2_texts = []
    for tag in soup.find_all("h2"):
        texto = tag.get_text(strip=True)
        if texto:
            h2_texts.append(texto)

    enlaces = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        abs_href = urljoin(base_url, href)
        enlaces.append(abs_href)

    return h2_texts, enlaces

def main():
    """
    1. Leer URL.
    2. Descargar HTML.
    3. Extraer h2 y enlaces.
    4. Imprimir resultados con formato:
       H2: texto
       LINK: url
    """
    url = leer_url()
    html = descargar_pagina(url)
    h2s, links = analizar_html(html, url)

    for txt in h2s:
        print(f"H2: {txt}")
    for link in links:
        print(f"LINK: {link}")

if __name__ == "__main__":
    main()
