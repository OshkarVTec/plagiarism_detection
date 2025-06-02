#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de sitio estático:
1. Lee todos los archivos Markdown (.md) en la carpeta 'content/'.
2. Convierte cada archivo Markdown a HTML usando la librería 'markdown'.
3. Envuelve el contenido HTML en una plantilla básica (header+footer).
4. Genera los archivos HTML resultantes en 'output/'.
5. Crea un 'index.html' con enlaces a cada página generada.
"""

import os
import sys
import shutil
import datetime
from markdown import markdown

# Directorio donde están los archivos Markdown
CONTENT_DIR = "content"
# Directorio de salida para los archivos HTML generados
OUTPUT_DIR = "output"
# Nombre del archivo índice
INDEX_FILE = "index.html"


# Plantilla base HTML (se puede personalizar)
# Usa placeholders: {{ title }}, {{ content }}, {{ date }}
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{ title }}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px; }}
        header {{ border-bottom: 1px solid #ccc; margin-bottom: 20px; }}
        footer {{ border-top: 1px solid #ccc; margin-top: 20px; font-size: 0.9em; color: #555; }}
        nav ul {{ list-style: none; padding: 0; }}
        nav ul li {{ margin: 5px 0; }}
        nav ul li a {{ text-decoration: none; color: #0066cc; }}
    </style>
</head>
<body>
    <header>
        <h1>{{ title }}</h1>
        <p><em>Fecha de generación: {{ date }}</em></p>
        <nav>
            <a href="index.html">Volver al índice</a>
        </nav>
    </header>
    <main>
        {{ content }}
    </main>
    <footer>
        <p>Generado por StaticSiteGen &copy; {{ year }}</p>
    </footer>
</body>
</html>
"""


def ensure_directories():
    """
    Verifica si existen los directorios CONTENT_DIR y OUTPUT_DIR.
    - Si no existe CONTENT_DIR, informa y sale.
    - Si existe OUTPUT_DIR, lo elimina (para regenerar desde cero) y lo recrea.
    """
    if not os.path.isdir(CONTENT_DIR):
        print(f"[ERROR] No se encontró el directorio '{CONTENT_DIR}'.")
        sys.exit(1)

    # Si existe OUTPUT_DIR, eliminarlo para limpiar contenido anterior
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Directorio de salida '{OUTPUT_DIR}' listo.")


def get_markdown_files():
    """
    Retorna una lista de rutas absolutas a archivos que terminen en '.md' dentro de CONTENT_DIR.
    Solo busca en el nivel superior (no recursivo).
    """
    files = []
    for entry in os.listdir(CONTENT_DIR):
        path = os.path.join(CONTENT_DIR, entry)
        if os.path.isfile(path) and entry.lower().endswith(".md"):
            files.append(path)
    return files


def convert_markdown_to_html(md_text):
    """
    Convierte texto Markdown a HTML usando la librería 'markdown'.
    Activa extensiones básicas como 'fenced_code' para bloques de código.
    """
    html = markdown(md_text, extensions=["fenced_code", "tables", "toc"])
    return html