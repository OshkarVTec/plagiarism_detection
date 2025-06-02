#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 del algoritmo de mochila 0/1:
Misma funcionalidad que el original, con nombres de variables cambiados,
comentarios reescritos y reorganización mínima del flujo de programa.
Lee desde stdin:
  - Línea 1: n (número de objetos) y C (capacidad).
  - Las siguientes n líneas: cada objeto con (peso costo) (peso y valor).
Calcula el valor máximo que puede llevarse en la mochila sin exceder C
y determina qué objetos forman parte de la solución óptima.
Imprime en stdout:
  - Valor máximo alcanzado.
  - Lista de índices (0 a n-1) de los objetos seleccionados, en orden ascendente.
"""

import sys

def cargar_datos():
    """
    Lee los datos de entrada:
      - n: cantidad de objetos
      - C: capacidad máxima
      - luego n líneas con dos enteros: peso_i y valor_i
    Devuelve tupla (n, C, lista_pesos, lista_valores).
    """
    try:
        linea1 = sys.stdin.readline().strip().split()
        if len(linea1) != 2:
            raise ValueError("Primera línea debe tener dos enteros (n y C).")
        n = int(linea1[0])
        C = int(linea1[1])
    except Exception as e:
        print(f"[ERROR] Datos inválidos en primera línea: {e}")
        sys.exit(1)

    lista_pesos = []
    lista_val = []
    for idx in range(n):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            print(f"[ERROR] Línea {idx+2} inválida: se requieren dos enteros.")
            sys.exit(1)
        try:
            w_i = int(parts[0])
            v_i = int(parts[1])
            if w_i < 0 or v_i < 0:
                raise ValueError("Peso o valor negativo no permitido.")
        except Exception as ex:
            print(f"[ERROR] Error en línea {idx+2}: {ex}")
            sys.exit(1)
        lista_pesos.append(w_i)
        lista_val.append(v_i)

    return n, C, lista_pesos, lista_val

def resolver_mochila(n, C, pesos, valores):
    """
    Resuelve el problema de la mochila 0/1 usando programación dinámica:
      - Tabla dp_tab de dimensión (n+1) x (C+1).
      - dp_tab[i][w] = valor máximo usando los primeros i objetos con capacidad w.
    Luego reconstruye la lista de objetos escogidos.
    Retorna (mejor_valor, objetos_elegidos).
    """
    # Inicializar tabla con ceros
    dp_tab = [[0] * (C + 1) for _ in range(n + 1)]

    # Llenado de la tabla
    for i in range(1, n + 1):
        peso_i = pesos[i - 1]
        valor_i = valores[i - 1]
        for cap_actual in range(C + 1):
            # No incluir el objeto (i-1)
            dp_tab[i][cap_actual] = dp_tab[i - 1][cap_actual]
            # Intentar incluirlo si cabe
            if peso_i <= cap_actual:
                posible = dp_tab[i - 1][cap_actual - peso_i] + valor_i
                if posible > dp_tab[i][cap_actual]:
                    dp_tab[i][cap_actual] = posible

    mejor_valor = dp_tab[n][C]

    # Backtracking para hallar índices de objetos seleccionados
    seleccion = []
    cap_restante = C
    for i in range(n, 0, -1):
        if dp_tab[i][cap_restante] != dp_tab[i - 1][cap_restante]:
            seleccion.append(i - 1)
            cap_restante -= pesos[i - 1]
        if cap_restante == 0:
            break

    seleccion.reverse()
    return mejor_valor, seleccion

def mostrar_resultado(valor_max, elementos):
    """
    Muestra por stdout:
      - Primera línea: valor máximo de la mochila.
      - Segunda línea: índices de objetos seleccionados separados por espacios.
    Si no hay elementos, imprime vacío tras la primera línea.
    """
    print(valor_max)
    if elementos:
        print(" ".join(str(idx) for idx in elementos))
    else:
        print()

def main():
    """
    Flujo principal:
      1. Cargar n, C, listas de pesos y valores con cargar_datos().
      2. Resolver con resolver_mochila().
      3. Mostrar resultado con mostrar_resultado().
    """
    n, C, lista_pesos, lista_val = cargar_datos()
    valor_optimo, seleccionados = resolver_mochila(n, C, lista_pesos, lista_val)
    mostrar_resultado(valor_optimo, seleccionados)

if __name__ == "__main__":
    main()
