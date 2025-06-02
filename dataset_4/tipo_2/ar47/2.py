#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión Tipo 2 de Regresión Lineal con Gradiente Descendente:
Lee:
  - Línea 1: N (número de datos) y η (tasa de aprendizaje)
  - N líneas con “X Y”.
Parámetros:
  - iter_max = 1000
Salida:
  - Valores de β0 β1 con 6 decimales.
Cada 100 iteraciones, imprimir el valor del costo al stdout.
"""

import sys

def cargar_datos():
    """
    Lee N y η, luego N pares X Y. Retorna lista de (X,Y) y η.
    """
    try:
        cab = sys.stdin.readline().strip().split()
        if len(cab) != 2:
            raise ValueError("Primera línea requiere 2 valores (N η).")
        N = int(cab[0])
        eta = float(cab[1])
        if N <= 0 or eta <= 0:
            raise ValueError("N > 0 y η > 0.")
    except Exception as e:
        print(f"[ERROR] Leer N y η: {e}")
        sys.exit(1)

    arr = []
    for i in range(N):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            print(f"[ERROR] Línea {i+2} inválida: se necesitan 2 valores.")
            sys.exit(1)
        try:
            X = float(parts[0])
            Y = float(parts[1])
        except:
            print(f"[ERROR] Valores no numéricos en línea {i+2}.")
            sys.exit(1)
        arr.append((X, Y))
    return arr, eta

def funcion_costo(b0, b1, datos):
    """
    Calcula Costo (1/2N) Σ (h(X_i) - Y_i)^2 para parámetros b0, b1.
    """
    N = len(datos)
    acc = 0.0
    for X, Y in datos:
        h = b0 + b1 * X
        acc += (h - Y) ** 2
    return acc / (2 * N)

def gradiente(datos, eta, iter_max=1000):
    """
    Gradiente descendente:
      - b0 := b0 - (η/N) Σ (h - Y)
      - b1 := b1 - (η/N) Σ (h - Y)*X
Imprime cada 100 iteraciones el costo. Retorna b0, b1.
    """
    N = len(datos)
    b0 = 0.0
    b1 = 0.0

    for e in range(1, iter_max + 1):
        sum_b0 = 0.0
        sum_b1 = 0.0
        for X, Y in datos:
            h = b0 + b1 * X
            err = h - Y
            sum_b0 += err
            sum_b1 += err * X
        b0 -= (eta / N) * sum_b0
        b1 -= (eta / N) * sum_b1

        if e % 100 == 0:
            costo = funcion_costo(b0, b1, datos)
            print(f"Itr {e}, Costo: {costo:.6f}")

    return b0, b1

def main():
    """
    1. Cargar datos y η.
    2. Ejecutar gradiente().
    3. Mostrar β0 β1 con 6 decimales.
    """
    datos, eta = cargar_datos()
    b0, b1 = gradiente(datos, eta)
    print(f"{b0:.6f} {b1:.6f}")

if __name__ == "__main__":
    main()
