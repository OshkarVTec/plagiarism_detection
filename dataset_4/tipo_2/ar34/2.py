#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuickSort reescrito (Tipo 2):
Misma lógica de ordenamiento, con renombrado de funciones y variables,
cambio de comentarios y formato. Lee entrada de números, los ordena
usando QuickSort y muestra el resultado.
"""

import sys
import threading

# Aumentamos límite de recursión para listas muy grandes
sys.setrecursionlimit(10**7)


def dividir(lista, bajo, alto):
    """
    Divide 'lista' en torno a un pivote (último elemento).
    Reubica elementos menores o iguales al pivote a la izquierda,
    y mayores a la derecha. Devuelve el índice del pivote final.
    """
    piv = lista[alto]
    indice = bajo - 1  # Índice hasta donde hemos colocado elementos <= pivote

    for k in range(bajo, alto):
        if lista[k] <= piv:
            indice += 1
            # Intercambiar lista[indice] con lista[k]
            lista[indice], lista[k] = lista[k], lista[indice]

    # Poner el pivote en la posición correcta
    lista[indice + 1], lista[alto] = lista[alto], lista[indice + 1]
    return indice + 1


def quick_recursivo(vec, bajo, alto):
    """
    Implementación recursiva de QuickSort:
      - Si 'bajo' es menor que 'alto', se partitiona y luego se ordenan
        recursivamente las mitades izquierda (bajo a piv-1) y derecha (piv+1 a alto).
    """
    if bajo < alto:
        piv_idx = dividir(vec, bajo, alto)
        quick_recursivo(vec, bajo, piv_idx - 1)
        quick_recursivo(vec, piv_idx + 1, alto)


def quicksort_inplace(vec):
    """
    Ordena la lista 'vec' usando QuickSort in-place.
    No retorna la lista (se modifica directamente).
    """
    tam = len(vec)
    quick_recursivo(vec, 0, tam - 1)


def leer_enteros():
    """
    Lee una línea de entrada estándar, separa por espacios y convierte cada token en entero.
    Si no es convertible, muestra un aviso y omite el token.
    Devuelve la lista de enteros leídos.
    """
    linea_input = sys.stdin.readline().strip()
    if not linea_input:
        return []

    partes = linea_input.split()
    arr_enteros = []
    for elemento in partes:
        try:
            arr_enteros.append(int(elemento))
        except ValueError:
            print(f"[AVISO] Token no válido: '{elemento}', será descartado")
    return arr_enteros


def main():
    """
    1. Imprime mensaje para ingresar números separados por espacios.
    2. Usa 'leer_enteros' para obtener la lista de enteros.
    3. Si la lista queda vacía, informa y finaliza.
    4. Muestra la lista original antes de ordenar.
    5. Llama a 'quicksort_inplace' para ordenar la lista.
    6. Imprime la lista ordenada.
    """
    print("Por favor ingresa varios números separados por espacios y presiona Enter:")
    vec = leer_enteros()

    if not vec:
        print("[INFO] No se ingresaron valores numéricos. Terminando ejecución.")
        return

    print(f"[INFO] Arreglo sin ordenar: {vec}")

    quicksort_inplace(vec)

    salida = " ".join(str(num) for num in vec)
    print(f"[SALIDA] Arreglo ordenado: {salida}")


if __name__ == "__main__":
    # Ejecutar en un hilo para prevenir problemas de recursión excesiva
    threading.Thread(target=main).start()
