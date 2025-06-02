#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramienta para migrar datos tablas específicas de una base de datos SQLite a otra.
Permite:
  - Definir tablas fuente y destino (nombre igual o diferente).
  - Crear la tabla destino si no existe, copiando esquema desde la original.
  - Copiar filas en lotes para no saturar memoria.
  - Mostrar avance de la migración.
"""

import sqlite3
import argparse
import sys
import os

def connect_db(db_path):
    """
    Abre conexión a la base de datos SQLite indicada en `db_path`.
    """
    if not os.path.exists(db_path):
        print(f"[ERROR] No se encontró la base de datos: {db_path}")
        sys.exit(1)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"[ERROR] No se pudo conectar a la base de datos: {e}")
        sys.exit(1)

def get_table_schema(conn, table_name):
    """
    Obtiene el esquema de la tabla `table_name` (instrucción CREATE TABLE).
    Retorna la cadena SQL correspondiente.
    """
    cursor = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    row = cursor.fetchone()
    if row:
        return row['sql']
    else:
        return None

def create_table(conn_dest, create_table_sql):
    """
    Ejecuta la instrucción `create_table_sql` en la base destino.
    """
    try:
        conn_dest.execute(create_table_sql)
        conn_dest.commit()
        print("[INFO] Tabla creada en la base destino.")
    except Exception as e:
        print(f"[ERROR] No se pudo crear la tabla en destino: {e}")
        sys.exit(1)

def migrate_table(conn_src, conn_dest, src_table, dest_table, batch_size=1000):
    """
    Copia datos de `src_table` en `conn_src` a `dest_table` en `conn_dest`:
      1. Obtiene columnas de la tabla fuente.
      2. Inserta filas en lotes de `batch_size`.
      3. Muestra progreso.
    """
    # Obtener columnas
    cursor = conn_src.execute(f"PRAGMA table_info({src_table});")
    cols = [row['name'] for row in cursor.fetchall()]
    col_list = ", ".join(cols)
    placeholders = ", ".join(["?"] * len(cols))

    # Contar filas totales
    cursor = conn_src.execute(f"SELECT COUNT(*) as cnt FROM {src_table};")
    total_rows = cursor.fetchone()['cnt']
    print(f"[INFO] Filas totales en '{src_table}': {total_rows}")

    offset = 0
    migrated = 0

    while migrated < total_rows:
        cursor = conn_src.execute(
            f"SELECT {col_list} FROM {src_table} LIMIT {batch_size} OFFSET {offset};"
        )
        rows = cursor.fetchall()
        if not rows:
            break

        # Preparar datos para inserción
        data = [tuple(row[col] for col in cols) for row in rows]
        try:
            conn_dest.executemany(
                f"INSERT INTO {dest_table} ({col_list}) VALUES ({placeholders});",
                data
            )
            conn_dest.commit()
        except Exception as e:
            print(f"[ERROR] Falló inserción en lote: {e}")
            sys.exit(1)

        migrated += len(data)
        offset += len(data)
        print(f"[INFO] Migradas {migrated}/{total_rows} filas.")

    print(f"[INFO] Migración de tabla '{src_table}' completada.")
