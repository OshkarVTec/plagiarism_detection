#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carrito de compras CLI.
Permite:
  - Listar productos disponibles.
  - Agregar productos al carrito.
  - Ver contenido del carrito.
  - Eliminar artículos del carrito.
  - Finalizar compra (calcular total).
Los productos están almacenados en un archivo 'products.json'.
El carrito se guarda en 'cart.json' entre ejecuciones.
"""

import json
import os
import sys
from datetime import datetime

# Archivos de almacenamiento
PRODUCTS_FILE = "products.json"
CART_FILE = "cart.json"

def load_products(file_path):
    """
    Carga lista de productos desde un archivo JSON.
    Cada producto: { "id": int, "name": str, "price": float }
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] No se encontró productos en '{file_path}'.")
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("products", [])
    except Exception as e:
        print(f"[ERROR] Falló lectura de productos: {e}")
        sys.exit(1)

def save_cart(cart_items, file_path):
    """
    Guarda el contenido del carrito en un archivo JSON.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"cart": cart_items}, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el carrito: {e}")

def load_cart(file_path):
    """
    Carga carrito desde archivo JSON. Si no existe, retorna lista vacía.
    """
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("cart", [])
    except Exception:
        return []

def list_products(products):
    """
    Muestra la lista de productos disponibles con sus IDs y precios.
    """
    print("\nProductos disponibles:")
    print(f"{'ID':<5} {'Nombre':<30} {'Precio ($)':>10}")
    print("-" * 50)
    for prod in products:
        print(f"{prod['id']:<5} {prod['name']:<30} {prod['price']:>10.2f}")
    print("")

def view_cart(cart_items, products):
    """
    Muestra contenido del carrito con cantidad y subtotal por ítem.
    """
    if not cart_items:
        print("\n[INFO] El carrito está vacío.\n")
        return

    print("\nContenido del carrito:")
    print(f"{'ID':<5} {'Nombre':<30} {'Cant.':<5} {'Precio':>10} {'Subtotal':>10}")
    print("-" * 70)
    total = 0.0
    for item in cart_items:
        prod = next((p for p in products if p['id'] == item['product_id']), None)
        if prod:
            subtotal = prod['price'] * item['quantity']
            total += subtotal
            print(f"{prod['id']:<5} {prod['name']:<30} {item['quantity']:<5} {prod['price']:>10.2f} {subtotal:>10.2f}")
    print("-" * 70)
    print(f"{'Total a pagar':<50} {total:>10.2f}\n")

def add_to_cart(cart_items, products):
    """
    Solicita ID de producto y cantidad, verifica existencia y agrega item al carrito.
    """
    try:
        prod_id = int(input("Ingresa ID de producto a agregar: ").strip())
        quantity = int(input("Cantidad: ").strip())
    except ValueError:
        print("[ERROR] Debes ingresar números enteros para ID y cantidad.")
        return

    prod = next((p for p in products if p['id'] == prod_id), None)
    if not prod:
        print(f"[ERROR] Producto con ID {prod_id} no existe.")
        return

    if quantity <= 0:
        print("[ERROR] La cantidad debe ser mayor a cero.")
        return

    # Verificar si ya existe en carrito
    existing = next((item for item in cart_items if item['product_id'] == prod_id), None)
    if existing:
        existing['quantity'] += quantity
    else:
        cart_items.append({"product_id": prod_id, "quantity": quantity})
    save_cart(cart_items, CART_FILE)
    print(f"[OK] Agregado {quantity} unidad(es) de '{prod['name']}' al carrito.")

def remove_from_cart(cart_items, products):
    """
    Solicita ID de producto a quitar, verifica y remueve completamente del carrito.
    """
    try:
        prod_id = int(input("Ingresa ID de producto a eliminar del carrito: ").strip())
    except ValueError:
        print("[ERROR] Debes ingresar un número entero para ID.")
        return

    existing = next((item for item in cart_items if item['product_id'] == prod_id), None)
    if not existing:
        print(f"[ERROR] No hay un producto con ID {prod_id} en el carrito.")
        return

    cart_items.remove(existing)
    save_cart(cart_items, CART_FILE)
    prod = next((p for p in products if p['id'] == prod_id), None)
    nombre = prod['name'] if prod else str(prod_id)
    print(f"[OK] Producto '{nombre}' eliminado del carrito.")

def checkout(cart_items, products):
    """
    Muestra el total y “finaliza” la compra.
    Después vacía el carrito y guarda cambios.
    """
    if not cart_items:
        print("[INFO] El carrito está vacío. Nada que pagar.")
        return

    total = 0.0
    for item in cart_items:
        prod = next((p for p in products if p['id'] == item['product_id']), None)
        if prod:
            total += prod['price'] * item['quantity']

    print(f"\nTotal a pagar: ${total:.2f}")
    confirm = input("¿Confirmar compra? (s/n): ").strip().lower()
    if confirm == "s":
        print("[INFO] Compra realizada el", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cart_items.clear()
        save_cart(cart_items, CART_FILE)
    else:
        print("[INFO] Compra cancelada.")

def print_menu():
    """
    Muestra el menú de opciones disponibles al usuario.
    """
    menu = """
=== Carrito de Compras CLI ===
1. Listar productos disponibles
2. Ver carrito
3. Agregar producto al carrito
4. Eliminar producto del carrito
5. Finalizar compra (checkout)
6. Salir
"""
    print(menu)

