#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuickSort en Python para ordenar una lista de enteros.
Lee desde stdin una línea con números separados por espacios,
los ordena usando QuickSort y luego imprime el resultado en stdout.
"""

import sys
import threading

sys.setrecursionlimit(10**7)


def particionar(arr, inicio, fin):
    """
    Toma el último elemento como pivote, coloca el pivote en su posición correcta
    en el arreglo ordenado, y coloca todos los menores a la izquierda y mayores a la derecha.
    Retorna el índice final del pivote.
    """
    pivote = arr[fin]
    i = inicio - 1  # Índice del último elemento menor que el pivote

    for j in range(inicio, fin):
        if arr[j] <= pivote:
            i += 1
            # Intercambiar arr[i] y arr[j]
            arr[i], arr[j] = arr[j], arr[i]

    # Colocar el pivote en la posición correcta
    arr[i + 1], arr[fin] = arr[fin], arr[i + 1]
    return i + 1


def quicksort_rec(arr, inicio, fin):
    """
    Implementación recursiva de QuickSort:
      - Si inicio < fin, particiona el arreglo.
      - Ordena recursivamente la parte izquierda y la parte derecha.
    """
    if inicio < fin:
        # part_index es el índice donde quedó el pivote
        part_index = particionar(arr, inicio, fin)
        # Ordenar sublista izquierda
        quicksort_rec(arr, inicio, part_index - 1)
        # Ordenar sublista derecha
        quicksort_rec(arr, part_index + 1, fin)


def quicksort(arr):
    """
    Función wrapper que llama a la versión recursiva,
    ordena 'arr' in-place y no retorna nada.
    """
    n = len(arr)
    quicksort_rec(arr, 0, n - 1)


def leer_lista_entrada():
    """
    Lee una línea de stdin, separa los tokens por espacios, convierte cada token a entero.
    Si un token no es convertible a entero, muestra advertencia y lo omite.
    Retorna una lista de enteros.
    """
    linea = sys.stdin.readline().strip()
    if not linea:
        return []
    tokens = linea.split()
    numeros = []
    for tok in tokens:
        try:
            valor = int(tok)
            numeros.append(valor)
        except ValueError:
            print(f"[WARN] Ignorando token no numérico: '{tok}'")
    return numeros


def main():
    """
    Flujo principal:
      1. Solicita al usuario que introduzca números separados por espacios.
      2. Usa 'leer_lista_entrada' para obtener la lista de enteros.
      3. Si la lista está vacía, informa y termina.
      4. Muestra la lista original.
      5. Llama a 'quicksort' para ordenar la lista in-place.
      6. Imprime la lista ordenada en una sola línea con espacios.
    """
    print("Introduce una lista de números separados por espacios y presiona Enter:")
    arr = leer_lista_entrada()

    if not arr:
        print("[INFO] No se ingresaron números válidos. Saliendo.")
        return

    print(f"[INFO] Lista antes de ordenar: {arr}")

    quicksort(arr)

    resultado = " ".join(str(x) for x in arr)
    print(f"[RESULTADO] Lista ordenada: {resultado}")


if __name__ == "__main__":
    # Ejecutar en un hilo para manejar recursión profunda si la lista es muy grande
    threading.Thread(target=main).start()
