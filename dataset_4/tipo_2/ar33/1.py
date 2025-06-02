#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Sort en Python para ordenar una lista de enteros.  
Lee desde stdin una serie de números separados por espacios,  
los ordena usando Merge Sort y los imprime en stdout separados por espacios.
"""

import sys

def merge(left, right):
    """
    Combina dos listas ordenadas (left y right) en una única lista ordenada.
    """
    i = 0
    j = 0
    merged = []
    # Mientras haya elementos en ambas sublistas
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Si quedó algún resto en left, agregarlo
    while i < len(left):
        merged.append(left[i])
        i += 1

    # Si quedó algo en right, agregarlo
    while j < len(right):
        merged.append(right[j])
        j += 1

    return merged

def merge_sort(arr):
    """
    Implementación recursiva de Merge Sort:
      - Si el arreglo tiene longitud <= 1, ya está ordenado (caso base).
      - Dividir arr en dos mitades, ordenar recursivamente y luego mezclar.
    Retorna una nueva lista ordenada.
    """
    n = len(arr)
    # Caso base: lista vacía o con un solo elemento
    if n <= 1:
        return arr[:]

    # Encontrar punto medio
    mid = n // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Ordenar recursivamente cada mitad
    sorted_left = merge_sort(left_half)
    sorted_right = merge_sort(right_half)

    # Mezclar ambas mitades
    return merge(sorted_left, sorted_right)

def leer_lista_desde_entrada():
    """
    Lee una línea de stdin, espera números separados por espacios,
    los convierte a enteros y retorna la lista resultante.
    """
    linea = sys.stdin.readline().strip()
    if not linea:
        return []
    partes = linea.split()
    numeros = []
    for token in partes:
        try:
            num = int(token)
            numeros.append(num)
        except ValueError:
            print(f"[WARN] Ignorando token no numérico: '{token}'")
    return numeros

def main():
    """
    1. Leer lista de enteros desde stdin.
    2. Si la lista está vacía, imprimir mensaje y salir.
    3. Llamar a merge_sort para obtener lista ordenada.
    4. Imprimir la lista ordenada en una sola línea separada por espacios.
    """
    print("Introduce una lista de números separados por espacios (y presiona Enter):")
    datos = leer_lista_desde_entrada()

    if not datos:
        print("[INFO] No se recibieron números. Saliendo.")
        return

    print(f"[INFO] Lista original: {datos}")

    ordenados = merge_sort(datos)

    # Formatear resultado para imprimir
    resultado = " ".join(str(x) for x in ordenados)
    print(f"[RESULTADO] Lista ordenada: {resultado}")

if __name__ == "__main__":
    main()
