#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resuelve un Sudoku de 9x9 usando algoritmo de backtracking:
  - Representa el tablero como una lista de listas (9x9).
  - Usa función recursiva para probar números del 1 al 9 en cada celda vacía.
  - Verifica que la inserción cumpla con reglas de fila, columna y bloque 3x3.
  - Imprime la solución en consola.
"""

import sys

def print_board(board):
    """
    Imprime el tablero de Sudoku en un formato legible.
    Usa líneas divisorias cada 3 filas/columnas.
    """
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        row = ''
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row += "| "
            row += f"{board[i][j]} " if board[i][j] != 0 else ". "
        print(row)
    print()

def find_empty(board):
    """
    Busca la siguiente celda vacía (valor 0) y retorna (fila, col).
    Si no hay vacías, retorna None.
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board, num, pos):
    """
    Verifica si `num` puede colocarse en `board[pos[0]][pos[1]]`:
      1. No debe repetirse en la misma fila.
      2. No debe repetirse en la misma columna.
      3. No debe repetirse en el bloque 3x3 correspondiente.
    """
    row, col = pos

    # Verificar fila
    for j in range(9):
        if board[row][j] == num and j != col:
            return False

    # Verificar columna
    for i in range(9):
        if board[i][col] == num and i != row:
            return False

    # Verificar bloque 3x3
    box_x = (col // 3) * 3
    box_y = (row // 3) * 3
    for i in range(box_y, box_y + 3):
        for j in range(box_x, box_x + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def solve(board):
    """
    Algoritmo recursivo de backtracking:
    1. Busca una celda vacía.
    2. Si no hay, el tablero está completo (solución encontrada).
    3. Prueba números del 1 al 9:
       - Si es válido, coloca y llama recursivamente.
       - Si la llamada retorna True, propagamos True.
       - Si no, resetea la celda a 0 (backtrack) y sigue probando.
    4. Si ningún número funciona, retorna False (backtrack).
    """
    find = find_empty(board)
    if not find:
        return True
    row, col = find

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num

            if solve(board):
                return True

            board[row][col] = 0

    return False

def read_board_from_file(file_path):
    """
    Lee un archivo de texto con 9 líneas, cada línea 9 dígitos (0 para vacío).
    Retorna el tablero como lista de listas de enteros.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Falló la lectura del archivo: {e}")
        sys.exit(1)

    if len(lines) != 9:
        print("[ERROR] El archivo debe tener 9 líneas con 9 caracteres cada una.")
        sys.exit(1)

    board = []
    for idx, line in enumerate(lines):
        if len(line) != 9 or any(c not in '0123456789' for c in line):
            print(f"[ERROR] Línea {idx+1} inválida: debe contener 9 dígitos (0-9).")
            sys.exit(1)
        row = [int(c) for c in line]
        board.append(row)

    return board
