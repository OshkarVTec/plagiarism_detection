#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de Heap Sort en Python para ordenar una lista de enteros.
Lee desde stdin una línea con números separados por espacios,  
construye un Max-Heap, luego extrae sucesivamente el máximo para  
producir la lista ordenada (en orden ascendente) e imprime el resultado.
"""

import sys

def leer_numeros():
    """
    Lee una línea de stdin, espera números separados por espacios.
    Convierte cada token a entero, ignorando tokens no válidos.
    Retorna la lista de enteros resultante.
    """
    linea = sys.stdin.readline().strip()
    if not linea:
        return []
    partes = linea.split()
    arr = []
    for tok in partes:
        try:
            num = int(tok)
            arr.append(num)
        except ValueError:
            print(f"[WARN] Token no numérico ignorado: '{tok}'")
    return arr

def heapify(arr, n, i):
    """
    Mantiene la propiedad de Max-Heap en el subárbol con raíz en índice i,
    asumiendo que los subárboles izquierdo y derecho ya son Heaps válidos.
    - arr: lista que contiene el heap.
    - n: tamaño del heap (porción de arr que consideramos).
    - i: índice de la raíz del subárbol a ajustar.
    """
    largest = i       # Inicializamos como el propio nodo
    left = 2 * i + 1  # Índice del hijo izquierdo
    right = 2 * i + 2 # Índice del hijo derecho

    # Si el hijo izquierdo existe y es mayor que arr[largest], actualizar largest
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Si el hijo derecho existe y es mayor que arr[largest], actualizar largest
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Si el mayor no es el propio nodo, intercambiar y ajustar recursivamente
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def construir_max_heap(arr):
    """
    Convierte la lista arr en un Max-Heap in-place.
    Recorre los nodos no hoja desde la mitad hacia atrás.
    """
    n = len(arr)
    # El último nodo no hoja está en índice (n//2 - 1)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

def heap_sort(arr):
    """
    Ordena la lista arr in-place usando Heap Sort:
      1. Construir Max-Heap de arr.
      2. Intercambiar arr[0] (máximo) con arr[n-1], reducir tamaño de heap en 1.
      3. Ajustar el nuevo arr[0] con heapify.
      4. Repetir hasta que el heap tenga tamaño 1.
    Retorna la misma lista arr ahora ordenada en orden ascendente.
    """
    n = len(arr)
    construir_max_heap(arr)

    # Extraer elementos del heap uno por uno
    for i in range(n - 1, 0, -1):
        # Mover el actual máximo (arr[0]) al final
        arr[0], arr[i] = arr[i], arr[0]
        # Llamar a heapify en la raíz, considerando heap de tamaño i
        heapify(arr, i, 0)
    return arr

def imprimir_lista(arr):
    """
    Imprime los elementos de arr en una sola línea, separados por espacios.
    """
    if arr:
        print(" ".join(str(x) for x in arr))
    else:
        print("[INFO] Lista vacía, nada que imprimir.")

def main():
    """
    Flujo principal:
      1. Solicita al usuario ingresar una línea de números separados por espacios.
      2. Lee y convierte a lista de enteros.
      3. Si la lista está vacía, finaliza.
      4. Aplica heap_sort para ordenarla.
      5. Imprime la lista resultante.
    """
    print("Introduce una lista de números separados por espacios y presiona Enter:")
    arr = leer_numeros()

    if not arr:
        print("[INFO] No se ingresaron números válidos. Saliendo.")
        return

    print(f"[INFO] Lista original: {arr}")
    arr_ordenada = heap_sort(arr.copy())  # Usamos copia para no modificar la original
    print(f"[RESULTADO] Lista ordenada: {arr_ordenada}")

if __name__ == "__main__":
    main()
