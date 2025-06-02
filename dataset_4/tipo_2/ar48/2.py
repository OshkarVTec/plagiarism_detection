#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de Web Scraper:
  - Lee una URL desde stdin.
  - Usa requests para obtener el HTML.
  - Con BeautifulSoup (parser “lxml”), extrae:
      * Textos de todas las etiquetas <h2> (strip).
      * URLs de todos los <a href="...">, convertidas a absolutas con urllib.parse.urljoin.
  - Salida en stdout:
      “H2: texto” por cada h2, luego “LINK: enlace” por cada enlace.
Requiere requests y bs4.
"""

import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def obtener_url():
    """
    Lee una línea con la URL. Debe empezar por http:// o https://
    """
    linea = sys.stdin.readline().strip()
    if not (linea.startswith("http://") or linea.startswith("https://")):
        print("[ERROR] Formato de URL incorrecto.")
        sys.exit(1)
    return linea

def fetch_html(website):
    """
    Realiza GET a website y retorna contenido HTML como texto.
    Si hay error, sale con mensaje.
    """
    try:
        respuesta = requests.get(website, timeout=10)
        respuesta.raise_for_status()
        return respuesta.text
    except Exception as ex:
        print(f"[ERROR] No se pudo obtener {website}: {ex}")
        sys.exit(1)

def extraer_datos(html, base):
    """
    Parse con BeautifulSoup:
      - h2s: lista de get_text(strip=True) para cada <h2>.
      - links: lista de href absolutos (relativos convertidos con urljoin).
    Retorna (h2s, links).
    """
    bs = BeautifulSoup(html, "lxml")
    h2s = []
    for h in bs.find_all("h2"):
        contenido = h.get_text(strip=True)
        if contenido:
            h2s.append(contenido)

    links = []
    for a in bs.find_all("a", href=True):
        url_rel = a["href"]
        abs_url = urljoin(base, url_rel)
        links.append(abs_url)

    return h2s, links

def main():
    """
    1. Obtener URL con obtener_url().
    2. Descargar HTML con fetch_html().
    3. Extraer textos h2 y enlaces con extraer_datos().
    4. Imprimir:
       H2: texto
       LINK: url
    """
    sitio = obtener_url()
    html_txt = fetch_html(sitio)
    titulos, enlaces = extraer_datos(html_txt, sitio)

    for t in titulos:
        print(f"H2: {t}")
    for l in enlaces:
        print(f"LINK: {l}")

if __name__ == "__main__":
    main()
