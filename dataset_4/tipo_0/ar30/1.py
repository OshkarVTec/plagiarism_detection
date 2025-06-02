#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realiza clustering K-Means sobre un conjunto de datos CSV:
  - Lee un archivo CSV con columnas numéricas (ej. características de flores, clientes, etc.).
  - Escala las características (Normalización Min-Max).
  - Aplica K-Means con un número de clusters definido por el usuario.
  - Muestra en consola:
      * Centros de cada cluster.
      * Cantidad de muestras en cada cluster.
  - Guarda un CSV de salida con una columna adicional 'cluster' indicando etiqueta.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_file):
    """
    Carga un CSV en un DataFrame de pandas y verifica que todas las columnas sean numéricas.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Falló al leer CSV: {e}")
        sys.exit(1)

    # Verificar que haya al menos una columna
    if df.shape[1] < 1:
        print("[ERROR] El CSV debe tener al menos una columna de datos.")
        sys.exit(1)

    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 1:
        print("[ERROR] No se encontraron columnas numéricas para clustering.")
        sys.exit(1)

    return df, numeric_df

def scale_features(num_df):
    """
    Escala características numéricas al rango [0,1] usando MinMaxScaler.
    Retorna el array escalado y el scaler usado.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(num_df.values)
    return scaled, scaler

def perform_kmeans(data, k, seed):
    """
    Aplica K-Means con k clusters y aleatoriedad controlada por seed.
    Retorna el objeto KMeans ajustado.
    """
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
    return kmeans

def summarize_clusters(kmeans, original_df):
    """
    Imprime en consola:
      - Centros de cada cluster (en espacio escalado).
      - Tamaño de cada cluster.
    """
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = pd.Series(labels).value_counts().sort_index()

    print("\nCentros de los clusters (en espacio escalado):")
    for idx, center in enumerate(centers):
        center_str = ", ".join(f"{val:.4f}" for val in center)
        print(f"  Cluster {idx}: [{center_str}]")

    print("\nCantidad de muestras por cluster:")
    for idx, count in counts.items():
        print(f"  Cluster {idx}: {count} muestras")

def save_output_csv(original_df, labels, output_file):
    """
    Agrega columna 'cluster' al DataFrame original y la guarda en output_file.
    """
    df_out = original_df.copy()
    df_out['cluster'] = labels
    try:
        df_out.to_csv(output_file, index=False)
        print(f"[INFO] CSV de salida guardado en '{output_file}'.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar CSV de salida: {e}")

