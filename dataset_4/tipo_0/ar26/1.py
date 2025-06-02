#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de informes PDF usando ReportLab:
  - Crea un documento PDF con:
      * Título en portada.
      * Sección de texto con párrafos.
      * Tabla de datos.
      * Gráfica(s) como imagen(es) incrustadas.
  - Usa reportlab.lib para estilos, colores y formatos.
  - El archivo PDF se genera en la ruta especificada.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import pandas as pd

def create_sample_plot(output_path):
    """
    Genera una gráfica de ejemplo con Matplotlib:
      - Gráfica de barras de ventas por categoría.
      - Guarda la figura en `output_path`.
    """
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 12, 30]

    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color='skyblue')
    plt.title('Ventas por Categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Ventas ($)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Gráfica de ejemplo guardada en: {output_path}")

def generate_pdf(output_pdf):
    """
    Genera el informe PDF en la ruta `output_pdf`:
      1. Agrega título en portadilla.
      2. Sección de texto con párrafos.
      3. Tabla de datos de ejemplo (creada con pandas).
      4. Inserta la gráfica creada por `create_sample_plot`.
    """
    # Estilos básicos
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['BodyText']

    # Crear documento
    doc = SimpleDocTemplate(output_pdf, pagesize=LETTER)
    elements = []

    # Portada con título
    elements.append(Paragraph("Informe de Ventas Mensuales", title_style))
    elements.append(Spacer(1, 0.5 * inch))

    # Texto introductorio
    intro_text = """
    Este informe contiene un análisis de las ventas mensuales por categoría.
    Se incluyen una tabla con los datos recopilados y una gráfica resumen.
    """
    elements.append(Paragraph(intro_text, normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Sección de la tabla de datos
    elements.append(Paragraph("Tabla de Datos de Ventas", heading_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Crear DataFrame de ejemplo
    data = {
        'Mes': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo'],
        'Categoría A': [1500, 1800, 1200, 1700, 1600],
        'Categoría B': [800, 950, 700, 1020, 1100],
        'Categoría C': [400, 450, 380, 420, 390]
    }
    df = pd.DataFrame(data)
    # Convertir DataFrame a lista de listas para reportlab Table
    table_data = [df.columns.tolist()] + df.values.tolist()

    # Crear tabla en reportlab
    table = Table(table_data, hAlign='LEFT')
    # Definir estilo de la tabla
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
    ])
    table.setStyle(style)

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    # Sección de la gráfica
    elements.append(Paragraph("Gráfica de Ventas por Categoría", heading_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Generar gráfica y guardarla en archivo temporal
    plot_path = "temp_plot.png"
    create_sample_plot(plot_path)

    # Insertar la imagen
    img = Image(plot_path, width=6 * inch, height=4 * inch)
    elements.append(img)

    # Construir documento
    try:
        doc.build(elements)
        print(f"[INFO] Informe PDF generado en: {output_pdf}")
    except Exception as e:
        print(f"[ERROR] No se pudo generar el PDF: {e}")
    finally:
        # Eliminar archivo temporal de gráfica
        if os.path.exists(plot_path):
            os.remove(plot_path)
