#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Juego del gato (tic-tac-toe) en modo consola:
- Jugador humano vs IA usando algoritmo minimax.
"""

import math
import time

# Tablero de 3x3 representado como lista de listas
board = [[" " for _ in range(3)] for _ in range(3)]
HUMAN = "X"
AI = "O"


def print_board():
    """Imprime el estado actual del tablero."""
    print("\n")
    print("    0   1   2")
    print("  -------------")
    for i in range(3):
        print(f"{i} | {' | '.join(board[i])} |")
        print("  -------------")
    print("\n")


def is_moves_left():
    """Retorna True si quedan espacios vacíos en el tablero."""
    for row in board:
        if " " in row:
            return True
    return False


def evaluate():
    """
    Evalúa el tablero:
    - Retorna +10 si la IA (O) gana.
    - Retorna -10 si el humano (X) gana.
    - Retorna 0 si empate o partida en curso.
    """
    # Verificar filas
    for row in board:
        if row[0] == row[1] == row[2] != " ":
            return 10 if row[0] == AI else -10

    # Verificar columnas
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != " ":
            return 10 if board[0][col] == AI else -10

    # Verificar diagonales
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return 10 if board[0][0] == AI else -10
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return 10 if board[0][2] == AI else -10

    # No hay ganador aún
    return 0


def minimax(depth, is_maximizing):
    """
    Algoritmo minimax recursivo.
    - depth: profundidad actual.
    - is_maximizing: True si es el turno de la IA, False si es el humano.
    Retorna el puntaje óptimo para este nodo.
    """
    score = evaluate()

    # Si la IA ganó
    if score == 10:
        return score - depth  # Restar depth para ganar antes
    # Si el humano ganó
    if score == -10:
        return score + depth  # Sumar depth para retrasar derrota

    # Empate
    if not is_moves_left():
        return 0

    if is_maximizing:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = AI
                    val = minimax(depth + 1, False)
                    best = max(best, val)
                    board[i][j] = " "
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = HUMAN
                    val = minimax(depth + 1, True)
                    best = min(best, val)
                    board[i][j] = " "
        return best


def find_best_move():
    """
    Encuentra la mejor jugada para la IA.
    Retorna una tupla (fila, columna).
    """
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = AI
                move_val = minimax(0, False)
                board[i][j] = " "
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move


def check_winner():
    """Verifica si ya hay ganador o empate. Retorna True si el juego termina."""
    score = evaluate()
    if score == 10:
        print("¡Gana la IA (O)!")
        return True
    if score == -10:
        print("¡Gana el humano (X)!")
        return True
    if not is_moves_left():
        print("Empate.")
        return True
    return False


def human_move():
    """Solicita al humano que ingrese su jugada (fila y columna)."""
    while True:
        try:
            user_input = input("Ingresa tu jugada (fila,columna): ")
            if "," not in user_input:
                raise ValueError("Formato inválido.")
            i_str, j_str = user_input.split(",")
            i, j = int(i_str), int(j_str)
            if i < 0 or i > 2 or j < 0 or j > 2 or board[i][j] != " ":
                raise ValueError("Movimiento inválido.")
            board[i][j] = HUMAN
            break
        except ValueError as e:
            print(f"Error: {e}. Intenta de nuevo.")

