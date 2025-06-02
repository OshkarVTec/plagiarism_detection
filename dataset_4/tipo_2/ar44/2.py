#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heap Sort reescrito (Tipo 2) para ordenar una lista de enteros.
Lee de stdin una línea de números separados por espacios,  
construye un Max-Heap (empujando abajo los subárboles) y luego  
extrae en bucle el máximo para producir la lista ordenada.  
El código renombra variables, funciones y comentarios.
"""

import sys

def leer_entrada():
    """
    Lee una línea desde sys.stdin, separa tokens por espacios, convierte
    cada uno a entero si es posible. Ignora tokens no convertibles
    con un mensaje de advertencia. Retorna lista de enteros.
    """
    texto = sys.stdin.readline().strip()
    if not texto:
        return []
    trozos = texto.split()
    lista = []
    for tok in trozos:
        try:
            valor = int(tok)
            lista.append(valor)
        except ValueError:
            print(f"[AVISO] Se omite token no entero: '{tok}'")
    return lista

def ajustar_heap(heap, tam, raiz):
    """
    Corrige la posición del nodo en índice 'raiz' en un Max-Heap de tamaño 'tam'
    dentro de la lista 'heap'. Ajusta recursivamente hacia abajo si es necesario.
    """
    mayor = raiz
    izquierdo = 2 * raiz + 1
    derecho = 2 * raiz + 2

    # Comprobar si el hijo izquierdo es mayor
    if izquierdo < tam and heap[izquierdo] > heap[mayor]:
        mayor = izquierdo

    # Comprobar si el hijo derecho es mayor
    if derecho < tam and heap[derecho] > heap[mayor]:
        mayor = derecho

    # Si mayor cambió, intercambiar y ajustar recursivamente
    if mayor != raiz:
        heap[raiz], heap[mayor] = heap[mayor], heap[raiz]
        ajustar_heap(heap, tam, mayor)

def construir_heap_max(lista):
    """
    Convierte la lista en un Max-Heap in-place. Recorre los nodos desde
    (len(lista)//2 - 1) hasta 0, llamando a ajustar_heap.
    Retorna la lista convertida en heap.
    """
    tam = len(lista)
    for idx in range(tam // 2 - 1, -1, -1):
        ajustar_heap(lista, tam, idx)
    return lista

def ordenar_por_heap(lista):
    """
    Realiza la ordenación por heap:
      1. Construir Max-Heap de la lista.
      2. Intercambiar el primer elemento con el último sin ordenar.
      3. Reducir el tamaño del heap en 1 y llamar a ajustar_heap en la raíz.
      4. Repetir hasta que quede un solo elemento.
    Devuelve la lista ordenada en orden ascendente.
    """
    n = len(lista)
    construir_heap_max(lista)
    for i in range(n - 1, 0, -1):
        lista[0], lista[i] = lista[i], lista[0]
        ajustar_heap(lista, i, 0)
    return lista

def mostrar_resultado(lista):
    """
    Imprime los elementos de 'lista' en una sola línea separado por espacios,
    o un mensaje si está vacía.
    """
    if not lista:
        print("[INFO] No hay elementos para mostrar.")
    else:
        print(" ".join(str(x) for x in lista))

def main():
    """
    Flujo principal:
      1. Lee lista de enteros desde stdin con leer_entrada().
      2. Si la lista está vacía, informa y termina.
      3. Imprime lista original.
      4. Llama a ordenar_por_heap para obtener la lista ordenada.
      5. Muestra la lista ordenada.
    """
    print("Por favor ingresa números separados por espacios y presiona Enter:")
    datos = leer_entrada()

    if not datos:
        print("[INFO] Lista ingresada vacía o inválida. Terminando ejecución.")
        return

    print(f"[INFO] Lista antes de ordenar: {datos}")
    ordenados = ordenar_por_heap(datos.copy())
    print(f"[SALIDA] Lista ordenada: {ordenados}")

if __name__ == "__main__":
    main()
