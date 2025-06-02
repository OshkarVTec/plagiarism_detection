#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini intérprete de un lenguaje muy sencillo que soporta:
  - Variables (números enteros).
  - Asignaciones: nombre = expresión.
  - Impresión por pantalla: print expresión.
  - Estructuras condicionales: if condición: ... else: ...
  - Bucle while: while condición: ...
  - Comentarios con #
  - REPL interactivo y ejecución de scripts desde archivo.
Gramática simplificada (no oficial, implementación ad hoc):
  statement    -> assignment | print_stmt | if_stmt | while_stmt | pass
  assignment   -> IDENTIFIER '=' expr
  print_stmt   -> 'print' expr
  if_stmt      -> 'if' expr ':' suite [ 'else' ':' suite ]
  while_stmt   -> 'while' expr ':' suite
  suite        -> INDENT statement+ DEDENT | statement
  expr (eval)  -> se usa eval() de Python en un entorno restringido
"""

import sys
import readline  # Para historial en REPL
import re

VARIABLES = {}  # Diccionario global de variables

def eval_expr(expr):
    """
    Evalúa una expresión aritmético-lógica en un entorno restringido.
    Solo permitimos operadores básicos y variables definidas en VARIABLES.
    """
    # Solo dejar caracteres seguros (dígitos, letras, operadores y espacios)
    if not re.match(r'^[\d\w\s\+\-\*\/\%\(\)<>=!&|]+$', expr):
        raise ValueError("Expresión contiene caracteres no permitidos.")
    # Evaluar en un entorno donde los nombres se resuelven en VARIABLES
    try:
        return eval(expr, {}, VARIABLES)
    except Exception as e:
        raise ValueError(f"Error al evaluar expresión: {e}")

def parse_assignment(line):
    """
    Detecta y ejecuta asignación: nombre = expr
    Retorna True si coincide y asigna, False si no.
    """
    if '=' not in line:
        return False
    parts = line.split('=', 1)
    var = parts[0].strip()
    expr = parts[1].strip()
    if not re.match(r'^[a-zA-Z_]\w*$', var):
        raise ValueError(f"Nombre de variable inválido: '{var}'")
    value = eval_expr(expr)
    VARIABLES[var] = value
    return True

def parse_print(line):
    """
    Detecta y ejecuta print expr
    Retorna True si es una instrucción print válida.
    """
    if not line.startswith('print '):
        return False
    expr = line[len('print '):].strip()
    value = eval_expr(expr)
    print(value)
    return True

def run_block(lines, i=0):
    """
    Ejecuta una lista de líneas (un bloque), detectando indentaciones.
    Devuelve el índice donde terminó el bloque.
    """
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Saltar líneas vacías o comentarios
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        # Si hay indent (bloque interior), retornar al llamador
        if indent > 0:
            return i

        # if
        if stripped.startswith('if '):
            # Extraer condición
            cond_part = stripped[len('if '):].rstrip(':').strip()
            cond = eval_expr(cond_part)
            # Encontrar bloque if
            i += 1
            # Ejecutar sub-bloque de líneas con indent > 0
            if_block = []
            else_block = []
            # Recorrer mientras indent >= 1
            while i < n and (len(lines[i]) - len(lines[i].lstrip())) > 0:
                if_block.append(lines[i][4:])  # quitar 4 espacios
                i += 1
            # Si siguiente es else
            if i < n and lines[i].lstrip().startswith('else:'):
                i += 1
                while i < n and (len(lines[i]) - len(lines[i].lstrip())) > 0:
                    else_block.append(lines[i][4:])
                    i += 1
            # Ejecutar block correspondiente
            if cond:
                run_block(if_block, 0)
            else:
                run_block(else_block, 0)
            continue

        # while
        if stripped.startswith('while '):
            cond_part = stripped[len('while '):].rstrip(':').strip()
            # Capturar bloque
            i += 1
            while_block = []
            while i < n and (len(lines[i]) - len(lines[i].lstrip())) > 0:
                while_block.append(lines[i][4:])
                i += 1
            # Ejecutar mientras condición sea True
            while eval_expr(cond_part):
                run_block(while_block, 0)
            continue

        # parse print
        if parse_print(stripped):
            i += 1
            continue

        # parse assignment
        if parse_assignment(stripped):
            i += 1
            continue

        # Si no coincide con ningún comando, error
        raise ValueError(f"Línea no reconocida: '{stripped}'")
    return i

def run_script(filepath):
    """
    Carga un archivo de texto con nuestro mini-lenguaje y lo ejecuta.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo: {filepath}")
        return

    # Normalizar líneas a 4 espacios como indent de bloque
    lines = [line.rstrip('\n') for line in raw_lines]
    try:
        run_block(lines, 0)
    except Exception as e:
        print(f"[ERROR] Al ejecutar script: {e}")

def repl():
    """
    Modo interactivo: lee líneas, mantiene indentación para bloques.
    """
    print("Mini intérprete. Escribe 'exit' o 'quit' para salir.")
    buffer = []
    while True:
        prompt = '... ' if buffer else '>>> '
        try:
            line = input(prompt)
        except EOFError:
            print()
            break

        if not buffer and line.strip() in ('exit', 'quit'):
            break

        # Si la línea termina en ':', esperamos un bloque indentado
        if line.rstrip().endswith(':'):
            buffer.append(line)
            continue

        # Si tenemos un bloque pendiente y la línea no está indentada, procesar bloque
        if buffer:
            if line.startswith('    '):
                buffer.append(line)
                continue
            else:
                # Ejecutar bloque
                buffer.append(line)
                try:
                    run_block(buffer, 0)
                except Exception as e:
                    print(f"[ERROR] {e}")
                buffer = []
                continue

        # Fuera de bloques
        if not line.strip():
            continue
        try:
            if parse_print(line):
                continue
            if parse_assignment(line):
                continue
            print(f"[ERROR] Comando no reconocido: '{line}'")
        except Exception as e:
            print(f"[ERROR] {e}")
