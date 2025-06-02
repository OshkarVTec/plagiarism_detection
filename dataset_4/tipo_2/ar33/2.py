#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión modificada (Tipo 2) de Merge Sort:  
Lee números desde stdin, aplica Merge Sort y escribe la lista ordenada.  
Se renombraron variables, se reestructuraron comentarios y se cambió el formato de algunas funciones,
pero la lógica subyacente sigue siendo idéntica al original.
"""

import sys

def mezclar(subizq, subder):
    """
    Dadas dos listas ya ordenadas (subizq y subder), retorna una lista combinada ordenada.
    """
    idx_izq = 0
    idx_der = 0
    fusionada = []

    # Mientras ambas listas tengan elementos
    while idx_izq < len(subizq) and idx_der < len(subder):
        if subizq[idx_izq] <= subder[idx_der]:
            fusionada.append(subizq[idx_izq])
            idx_izq += 1
        else:
            fusionada.append(subder[idx_der])
            idx_der += 1

    # Agregar los elementos restantes de subizq, si los hay
    while idx_izq < len(subizq):
        fusionada.append(subizq[idx_izq])
        idx_izq += 1

    # Agregar los elementos restantes de subder, si los hay
    while idx_der < len(subder):
        fusionada.append(subder[idx_der])
        idx_der += 1

    return fusionada

def orden_merge(lista):
    """
    Ordena recursivamente la lista usando Merge Sort:
      - Si el tamaño es 0 o 1, retorna copia de lista (ya está ordenada).
      - Divide en dos segmentos: izquierda y derecha.
      - Ordena cada uno recursivamente y luego los fusiona.
    Retorna la nueva lista ordenada.
    """
    longitud = len(lista)
    # Caso base: lista de tamaño 0 o 1
    if longitud <= 1:
        return lista[:]

    medio = longitud // 2
    izquierda = lista[:medio]
    derecha = lista[medio:]

    # Llamadas recursivas
    izq_ordenada = orden_merge(izquierda)
    der_ordenada = orden_merge(derecha)

    # Fusionar las mitades ordenadas
    return mezclar(izq_ordenada, der_ordenada)

def leer_numeros():
    """
    Lee una línea desde sys.stdin, separa por espacios y convierte cada token a entero.
    Si un token no es convertible, muestra advertencia y lo omite.
    Retorna la lista resultante de enteros.
    """
    entrada = sys.stdin.readline().strip()
    if not entrada:
        return []
    partes = entrada.split()
    lista_enteros = []
    for tok in partes:
        try:
            lista_enteros.append(int(tok))
        except ValueError:
            print(f"[AVISO] Token no válido ignorado: '{tok}'")
    return lista_enteros

def main():
    """
    Flujo principal:
    - Solicita al usuario que ingrese números separados por espacios.
    - Utiliza leer_numeros() para obtener la lista de enteros.
    - Si la lista está vacía, informa y termina.
    - Muestra la lista original y aplica orden_merge(lista).
    - Imprime la lista ordenada en una sola línea.
    """
    print("Por favor ingresa números separados por espacios y presiona Enter:")
    arr = leer_numeros()

    if not arr:
        print("[INFO] Lista vacía o no válida. Fin del programa.")
        return

    print(f"[INFO] Lista antes de ordenar: {arr}")

    arr_ordenada = orden_merge(arr)

    salida = " ".join(str(num) for num in arr_ordenada)
    print(f"[SALIDA] Lista ordenada: {salida}")

if __name__ == "__main__":
    main()
