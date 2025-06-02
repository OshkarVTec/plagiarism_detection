#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de Re
gression Lineal simple (un predictor) usando gradiente descendente.
Lee desde stdin:
  - Primera línea: m (número de muestras) y α (learning rate), separados por espacio.
  - Siguientes m líneas: cada una “x y” (floats), donde x es la variable independiente y y la dependiente.
Parámetros internos:
  - num_iters = 1000
Salida:
  - θ0 θ1 (dos valores con 6 decimales) que minimizan el costo (ecm).
También imprime, cada 100 iteraciones, el valor de la función de costo actual al stdout.
"""

import sys

def leer_datos():
    """
    Lee m y α, luego m pares (x, y). Retorna lista de tuplas (x,y) y α.
    """
    try:
        linea = sys.stdin.readline().strip().split()
        if len(linea) != 2:
            raise ValueError("Se esperaban 2 valores en la primera línea (m y alpha).")
        m = int(linea[0])
        alpha = float(linea[1])
        if m <= 0 or alpha <= 0:
            raise ValueError("m > 0 y α > 0.")
    except Exception as e:
        print(f"[ERROR] Leer m y α: {e}")
        sys.exit(1)

    datos = []
    for i in range(m):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            print(f"[ERROR] Línea {i+2} inválida: se requieren 2 valores.")
            sys.exit(1)
        try:
            x = float(parts[0])
            y = float(parts[1])
        except:
            print(f"[ERROR] Datos no numéricos en línea {i+2}.")
            sys.exit(1)
        datos.append((x, y))
    return datos, alpha

def costo_ecm(theta0, theta1, datos):
    """
    Calcula el error cuadrático medio (ECM) para parámetros θ0, θ1 con las muestras.
    ECM = (1/2m) Σ (pred - y)^2
    """
    m = len(datos)
    total = 0.0
    for x, y in datos:
        pred = theta0 + theta1 * x
        total += (pred - y) ** 2
    return total / (2 * m)

def gradiente_descendente(datos, alpha, num_iters=1000):
    """
    Aplica gradiente descendente para ajustar θ0 y θ1.
    Itera num_iters veces:
      - θ0 := θ0 - (α/m) Σ (h(x_i) - y_i)
      - θ1 := θ1 - (α/m) Σ (h(x_i) - y_i) * x_i
Imprime cada 100 iteraciones el valor del costo. Retorna θ0, θ1.
    """
    m = len(datos)
    theta0 = 0.0
    theta1 = 0.0

    for it in range(1, num_iters + 1):
        sum0 = 0.0
        sum1 = 0.0
        for x, y in datos:
            pred = theta0 + theta1 * x
            error = pred - y
            sum0 += error
            sum1 += error * x
        theta0 -= (alpha / m) * sum0
        theta1 -= (alpha / m) * sum1

        if it % 100 == 0:
            cost = costo_ecm(theta0, theta1, datos)
            print(f"Iter {it}, Costo: {cost:.6f}")

    return theta0, theta1

def main():
    """
    1. Leer datos y α.
    2. Llamar a gradiente_descendente().
    3. Imprimir θ0 y θ1 con 6 decimales.
    """
    datos, alpha = leer_datos()
    theta0, theta1 = gradiente_descendente(datos, alpha)
    print(f"{theta0:.6f} {theta1:.6f}")

if __name__ == "__main__":
    main()
