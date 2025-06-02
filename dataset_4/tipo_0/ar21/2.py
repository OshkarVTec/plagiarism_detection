#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa para realizar OCR (Reconocimiento Óptico de Caracteres) en imágenes.
Utiliza pytesseract (wrapper de Tesseract) y PIL (Pillow) para procesar la imagen.
Permite:
  - Cargar una imagen desde disco.
  - Preprocesarla (convertir a escala de grises, umbralizar).
  - Extraer texto mediante pytesseract.
  - Guardar el texto extraído en un archivo .txt.
"""

import os
import sys
import argparse
from PIL import Image, ImageFilter, ImageOps
import pytesseract

def preprocess_image(image_path, output_path=None):
    """
    1. Carga la imagen desde `image_path`.
    2. Convierte a escala de grises.
    3. Aplica filtro de nitidez (optional).
    4. Aplica umbralización (binarización) para mejorar contraste.
    5. Guarda (opcional) la imagen preprocesada en `output_path`.
    Retorna el objeto PIL.Image preprocesado.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró la imagen: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] No se pudo abrir la imagen: {e}")
        sys.exit(1)

    # Convertir a escala de grises
    gray = ImageOps.grayscale(img)
    # Aplicar filtro de nitidez (opcional, mejora bordes)
    sharpened = gray.filter(ImageFilter.SHARPEN)
    # Aplicar umbralización para convertir a blanco y negro
    threshold = 128
    bw = sharpened.point(lambda x: 0 if x < threshold else 255, '1')

    if output_path:
        try:
            bw.save(output_path)
            print(f"[INFO] Imagen preprocesada guardada en: {output_path}")
        except Exception as e:
            print(f"[WARN] No se pudo guardar la imagen preprocesada: {e}")

    return bw

def perform_ocr(image_obj, lang='spa'):
    """
    Realiza OCR sobre el objeto `image_obj` usando pytesseract.
    El parámetro `lang` define el idioma para Tesseract (por defecto 'spa' para español).
    Retorna el texto extraído como cadena.
    """
    try:
        text = pytesseract.image_to_string(image_obj, lang=lang)
        return text
    except pytesseract.TesseractNotFoundError:
        print("[ERROR] Tesseract no está instalado o no se encuentra en PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Ocurrió un error haciendo OCR: {e}")
        sys.exit(1)

def save_text(text, output_txt_path):
    """
    Guarda la cadena `text` en un archivo de texto en `output_txt_path`.
    """
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[INFO] Texto extraído guardado en: {output_txt_path}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el texto: {e}")
