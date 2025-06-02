#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación del algoritmo de Longest Common Subsequence (LCS)
usando programación dinámica. Lee dos cadenas desde stdin (una por línea),
calcula la subsecuencia común más larga y la imprime junto con su longitud.
"""

import sys

def leer_cadenas():
    """
    Lee dos líneas desde stdin, cada una representando una cadena.
    Retorna una tupla (cadena1, cadena2).
    """
    try:
        s1 = sys.stdin.readline().rstrip('\n')
        s2 = sys.stdin.readline().rstrip('\n')
        if s1 is None or s2 is None:
            raise ValueError("No se recibieron dos cadenas válidas.")
        return s1, s2
    except Exception as e:
        print(f"[ERROR] Al leer cadenas: {e}")
        sys.exit(1)

def construir_tabla_dp(a, b):
    """
    Construye la tabla dp de tamaño (len(a)+1) x (len(b)+1), donde
    dp[i][j] es la longitud de la LCS de a[:i] y b[:j].
    """
    n = len(a)
    m = len(b)
    # Inicializar tabla con ceros
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp

def backtrack_lcs(dp, a, b):
    """
    Reconstruye una LCS a partir de la tabla dp, recorriéndola de abajo a arriba.
    Retorna la subsecuencia común más larga encontrada.
    """
    i = len(a)
    j = len(b)
    lcs_chars = []

    # Retroceder hasta llegar a dp[0][*] o dp[*][0]
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            # Caracter coincide: parte de la LCS
            lcs_chars.append(a[i - 1])
            i -= 1
            j -= 1
        else:
            # Mover hacia la dirección de mayor valor
            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

    # Como construimos la LCS de atrás hacia adelante, hay que invertirla
    lcs_chars.reverse()
    return ''.join(lcs_chars)

def imprimir_resultado(lcs):
    """
    Imprime en stdout:
      - Primera línea: longitud de la LCS.
      - Segunda línea: la LCS en sí misma (cadena).
    Si la LCS está vacía, imprime longitud 0 y línea en blanco.
    """
    longitud = len(lcs)
    print(longitud)
    if longitud > 0:
        print(lcs)
    else:
        print()

def main():
    """
    Flujo principal:
      1. Leer dos cadenas desde stdin.
      2. Construir la tabla dp para LCS.
      3. Reconstruir la LCS usando backtracking.
      4. Imprimir longitud y la LCS.
    """
    # Paso 1: leer las dos cadenas
    cadena1, cadena2 = leer_cadenas()

    # Paso 2: construir tabla dp
    dp_table = construir_tabla_dp(cadena1, cadena2)

    # Paso 3: backtracking para obtener la LCS
    subsecuencia = backtrack_lcs(dp_table, cadena1, cadena2)

    # Paso 4: imprimir resultados
    imprimir_resultado(subsecuencia)

if __name__ == "__main__":
    main()
