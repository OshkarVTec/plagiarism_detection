#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 del algoritmo de LCS:
Misma lógica para obtener la subsecuencia común más larga entre dos cadenas,
con renombrado de funciones y variables, reescritura de comentarios y reorganización mínima.
Lee dos cadenas desde stdin (sin línea vacía intermedia), calcula la LCS
e imprime su longitud y la propia subsecuencia.
"""

import sys

def obtener_entradas():
    """
    Lee dos líneas consecutivas desde stdin y las retorna como (str1, str2).
    Si no se reciben dos líneas válidas, sale con error.
    """
    try:
        linea1 = sys.stdin.readline().rstrip('\n')
        linea2 = sys.stdin.readline().rstrip('\n')
        if linea1 == '' and linea2 == '':
            raise ValueError("No se leyeron cadenas suficientes.")
        return linea1, linea2
    except Exception as exc:
        print(f"[ERROR] Fallo al leer cadenas: {exc}")
        sys.exit(1)

def crear_matriz(a, b):
    """
    Crea y devuelve la matriz dp de dimensiones (len(a)+1) x (len(b)+1).
    dp[i][j] contendrá la longitud de la LCS de a[0..i-1] y b[0..j-1].
    """
    len_a = len(a)
    len_b = len(b)
    # Inicialización con ceros
    dp_mat = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                dp_mat[i][j] = dp_mat[i - 1][j - 1] + 1
            else:
                # Tomar el máximo entre el valor superior o el izquierdo
                arriba = dp_mat[i - 1][j]
                izquierda = dp_mat[i][j - 1]
                dp_mat[i][j] = arriba if arriba >= izquierda else izquierda
    return dp_mat

def reconstruir_lcs(dp_mat, cadenaA, cadenaB):
    """
    A partir de la matriz dp_mat, reconstruye la LCS:
      - Empieza en (i=len(cadenaA), j=len(cadenaB)).
      - Si los caracteres coinciden, añade a resultado y mueve diagonalmente.
      - Si no, se desplaza donde el valor sea mayor (arriba o izquierda).
    Devuelve la subsecuencia resultante.
    """
    i = len(cadenaA)
    j = len(cadenaB)
    resultado = []

    while i > 0 and j > 0:
        if cadenaA[i - 1] == cadenaB[j - 1]:
            resultado.append(cadenaA[i - 1])
            i -= 1
            j -= 1
        else:
            if dp_mat[i - 1][j] >= dp_mat[i][j - 1]:
                i -= 1
            else:
                j -= 1

    # Invertir la lista de caracteres
    resultado.reverse()
    return ''.join(resultado)

def mostrar_salida(lcs_str):
    """
    Imprime en stdout:
      - Línea 1: longitud de la subsecuencia lcs_str.
      - Línea 2: la subsecuencia lcs_str (si no es vacía).
      Si es vacía, imprime línea en blanco tras la longitud 0.
    """
    lon = len(lcs_str)
    print(lon)
    if lon > 0:
        print(lcs_str)
    else:
        print()

def main():
    """
    1. Leer dos cadenas usando obtener_entradas().
    2. Generar la matriz dp con crear_matriz().
    3. Reconstruir la LCS con reconstruir_lcs().
    4. Mostrar longitud y subsecuencia resultante con mostrar_salida().
    """
    # Leer entradas
    str1, str2 = obtener_entradas()

    # Construir tabla de DP
    tabla = crear_matriz(str1, str2)

    # Obtener la subsecuencia común más larga
    lcs_final = reconstruir_lcs(tabla, str1, str2)

    # Mostrar resultado
    mostrar_salida(lcs_final)

if __name__ == "__main__":
    main()
