#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa que lee un archivo CSV con datos de ventas y genera estadísticas:
- Total de ventas por producto.
- Gráfica de barras de ventas.
- Cálculo de media y desviación estándar de ingresos mensuales.
"""

import csv
import sys
from collections import defaultdict
from datetime import datetime
import statistics
import matplotlib.pyplot as plt

# Verifica que se proporcione el archivo CSV como argumento
if len(sys.argv) != 2:
    print("Uso: python data_analysis.py <archivo_ventas.csv>")
    sys.exit(1)

csv_file = sys.argv[1]

# Estructuras para acumular datos
ventas_por_producto = defaultdict(float)
ingresos_por_mes = defaultdict(float)

# Suponemos que el CSV tiene columnas: date,product,price,quantity
# Ejemplo de encabezado: 2025-05-01,Widget A,19.99,3
try:
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fecha = datetime.strptime(row["date"], "%Y-%m-%d")
                producto = row["product"]
                precio = float(row["price"])
                cantidad = int(row["quantity"])
                ingreso = precio * cantidad

                ventas_por_producto[producto] += ingreso
                mes_key = fecha.strftime("%Y-%m")
                ingresos_por_mes[mes_key] += ingreso
            except Exception as e:
                print(f"Error al procesar fila: {row} -> {e}")
except FileNotFoundError:
    print(f"No se encontró el archivo: {csv_file}")
    sys.exit(1)

# Mostrar resumen de ventas por producto
print("\nTotal de ingresos por producto:")
for prod, total in ventas_por_producto.items():
    print(f"  - {prod}: ${total:.2f}")

# Mostrar estadísticas de ingresos mensuales
meses = sorted(ingresos_por_mes.keys())
valores_mensual = [ingresos_por_mes[mes] for mes in meses]
media_mensual = statistics.mean(valores_mensual) if valores_mensual else 0
desv_mensual = statistics.pstdev(valores_mensual) if len(valores_mensual) > 1 else 0

print(f"\nEstadísticas de ingresos mensuales:")
print(f"  Meses analizados: {len(meses)}")
print(f"  Media mensual: ${media_mensual:.2f}")
print(f"  Desviación estándar: ${desv_mensual:.2f}")

# Gráfica de barras: ventas por producto
productos = list(ventas_por_producto.keys())
valores = [ventas_por_producto[p] for p in productos]

plt.figure(figsize=(10, 6))
plt.bar(productos, valores)
plt.title("Ingresos por producto")
plt.xlabel("Producto")
plt.ylabel("Ingresos ($)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("ventas_por_producto.png")
print("\nGráfica 'ventas_por_producto.png' generada.")

# Gráfica de línea: ingresos mensuales
plt.figure(figsize=(10, 6))
plt.plot(meses, valores_mensual, marker="o")
plt.title("Ingresos mensuales")
plt.xlabel("Mes")
plt.ylabel("Ingresos ($)")
plt.grid(True)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("ingresos_mensuales.png")
print("Gráfica 'ingresos_mensuales.png' generada.")
