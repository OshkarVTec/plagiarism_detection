#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para subir archivos o carpetas enteras a un bucket de AWS S3 usando boto3.
Permite:
  - Autenticar usando credenciales configuradas en AWS CLI o variables de entorno.
  - Subir un único archivo o todos los archivos bajo un directorio (recursivo).
  - Mostrar barra de progreso en consola durante la carga.
  - Configurar el bucket y prefijo (carpeta en S3) destino.
"""

import os
import sys
import threading
import argparse
import boto3
from botocore.exceptions import ClientError

class ProgressPercentage:
    """
    Muestra en consola el progreso de subida de un archivo a S3.
    Calcula porcentaje basado en bytes transferidos.
    """

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # bytes_amount es cantidad transferida en el último bloque
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                f"\rSubiendo {self._filename}: {self._seen_so_far:.0f}/{self._size:.0f} bytes ({percentage:.2f}%)"
            )
            sys.stdout.flush()
            if self._seen_so_far >= self._size:
                print()  # Saltar línea cuando termine

def upload_file(s3_client, file_path, bucket, key_prefix):
    """
    Sube `file_path` a S3 en `bucket` bajo la clave `key_prefix/<nombre archivo>`.
    Muestra progreso usando ProgressPercentage.
    """
    filename = os.path.basename(file_path)
    s3_key = f"{key_prefix}/{filename}" if key_prefix else filename
    try:
        s3_client.upload_file(
            file_path, bucket, s3_key,
            Callback=ProgressPercentage(file_path)
        )
        print(f"[OK] Subido a s3://{bucket}/{s3_key}")
    except ClientError as e:
        print(f"[ERROR] Falló al subir {file_path}: {e}")

def upload_directory(s3_client, dir_path, bucket, key_prefix):
    """
    Recorre recursivamente `dir_path` y sube cada archivo encontrado.
    Mantiene estructura relativa dentro del prefijo `key_prefix`.
    """
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Determinar ruta relativa para crear key en S3
            rel_dir = os.path.relpath(root, dir_path)
            rel_key = os.path.join(key_prefix, rel_dir) if key_prefix else rel_dir
            upload_file(s3_client, local_path, bucket, rel_key)
