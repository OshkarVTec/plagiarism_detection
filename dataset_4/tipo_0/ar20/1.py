#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de logs en formato común (Common Log Format).
Ejemplo de línea de CLF:
127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
El script:
  - Lee archivo de log de texto.
  - Cuenta peticiones por IP.
  - Cuenta peticiones por recurso (ruta solicitada).
  - Muestra top 10 IPs y top 10 recursos más solicitados.
"""

import re
import sys
from collections import Counter

# Expresión regular para parsear CLF
LOG_PATTERN = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+)\s+'         # IP del cliente
    r'\S+\s+'                                # Identd (omitido)
    r'(?P<user>\S+)\s+'                      # Usuario autenticado (omitido)
    r'\[(?P<time>.+?)\]\s+'                  # Timestamp
    r'"(?P<request>.+?)"\s+'                 # Petición (método, ruta, protocolo)
    r'(?P<status>\d{3})\s+'                  # Código de estado
    r'(?P<size>\S+)'                         # Tamaño de respuesta
)

def parse_log_line(line):
    """
    Parsea una línea de log según LOG_PATTERN.
    Retorna un diccionario con campos: ip, user, time, request, status, size.
    Si no coincide, retorna None.
    """
    match = LOG_PATTERN.match(line)
    if not match:
        return None
    return match.groupdict()

def extract_resource(request_str):
    """
    Extrae la ruta (recurso) de la cadena de petición, e.g. "GET /index.html HTTP/1.1".
    Retorna '/index.html'. Si no se puede extraer, retorna '-'.
    """
    parts = request_str.split()
    if len(parts) >= 2:
        return parts[1]
    return "-"

def analyze_log(file_path):
    """
    Recorre el archivo de log línea por línea:
    - Conteo de IPs en Counter ip_counter
    - Conteo de recursos solicitados en Counter res_counter
    Retorna ambos contadores.
    """
    ip_counter = Counter()
    res_counter = Counter()
    total_lines = 0
    parsed_lines = 0

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                entry = parse_log_line(line)
                if entry:
                    parsed_lines += 1
                    ip = entry["ip"]
                    resource = extract_resource(entry["request"])
                    ip_counter[ip] += 1
                    res_counter[resource] += 1
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de log: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Falló al leer el archivo: {e}")
        sys.exit(1)

    print(f"[INFO] Líneas procesadas: {total_lines}, Líneas válidas: {parsed_lines}")
    return ip_counter, res_counter

def print_top(counter, top_n, descriptor):
    """
    Imprime las 'top_n' entradas más comunes en un Counter.
    """
    print(f"\nTop {top_n} {descriptor}:")
    print(f"{descriptor:<30} {'Conteo':>10}")
    print("-" * 42)
    for item, count in counter.most_common(top_n):
        print(f"{item:<30} {count:>10}")
    print("")
