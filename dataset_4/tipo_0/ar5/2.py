#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web scraper para extraer titulares de noticias desde un sitio web.
Ejemplo usando requests y BeautifulSoup.
"""

import requests
from bs4 import BeautifulSoup
import time
import csv

# URL de ejemplo (puede ser un sitio de noticias como CNN en español)
BASE_URL = "https://cnnespanol.cnn.com/"

# Encabezados para simular un navegador y evitar bloqueos
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Archivo CSV donde se guardarán los titulares
OUTPUT_CSV = "headlines.csv"


def fetch_homepage():
    """Descarga el HTML de la página principal."""
    try:
        response = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error al descargar la página: {e}")
        return None


def parse_headlines(html):
    """
    Extrae los titulares del HTML.
    Se asume que los titulares están en elementos <h3> con clase específica.
    """
    soup = BeautifulSoup(html, "html.parser")
    headlines = []

    # Ejemplo: titulares dentro de <h3 class="container__title">  
    # Ajusta esto según la estructura real del sitio.
    for h3 in soup.find_all("h3"):
        title = h3.get_text(strip=True)
        if title:
            headlines.append(title)

    return headlines


def save_to_csv(headlines):
    """Guarda la lista de titulares en un archivo CSV."""
    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Titular", "Fecha de extracción"])
        fecha = time.strftime("%Y-%m-%d %H:%M:%S")
        for title in headlines:
            writer.writerow([title, fecha])
    print(f"Titulares guardados en {OUTPUT_CSV}")

