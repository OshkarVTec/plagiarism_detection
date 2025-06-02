#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de autenticación básico en consola usando SQLite.
Permite registrar usuarios con contraseña (hash SHA-256) y luego iniciar sesión.
"""

import sqlite3
import sys
import os
import getpass
import hashlib

DB_FILE = "auth.db"


def get_db_connection():
    """Obtiene (o crea) la conexión a la base de datos SQLite."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inicializa la tabla de usuarios si no existe."""
    if not os.path.exists(DB_FILE):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()
        print("Base de datos inicializada.")


def hash_password(password):
    """Hash de la contraseña usando SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user():
    """Registra un nuevo usuario con contraseña."""
    conn = get_db_connection()
    cursor = conn.cursor()

    username = input("Usuario: ").strip()
    if not username:
        print("El nombre de usuario no puede estar vacío.")
        conn.close()
        return

    # Verificar si el usuario ya existe
    cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
    if cursor.fetchone():
        print("El usuario ya existe.")
        conn.close()
        return

    while True:
        password = getpass.getpass("Contraseña: ").strip()
        password_confirm = getpass.getpass("Confirmar contraseña: ").strip()
        if not password:
            print("La contraseña no puede estar vacía.")
        elif password != password_confirm:
            print("Las contraseñas no coinciden.")
        else:
            break

    pwd_hash = hash_password(password)
    cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?);", (username, pwd_hash))
    conn.commit()
    conn.close()
    print(f"Usuario '{username}' registrado correctamente.")


def login_user():
    """Solicita credenciales y verifica en la base de datos."""
    conn = get_db_connection()
    cursor = conn.cursor()

    username = input("Usuario: ").strip()
    password = getpass.getpass("Contraseña: ").strip()
    pwd_hash = hash_password(password)

    cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
    row = cursor.fetchone()
    conn.close()

    if row and row["password_hash"] == pwd_hash:
        print(f"¡Bienvenido, {username}!")
    else:
        print("Credenciales inválidas.")


def print_help():
    """Muestra las opciones disponibles."""
    help_text = """
Sistema de autenticación (SQLite)
Uso:
    python auth_system.py register
        Registra un nuevo usuario.
    python auth_system.py login
        Inicia sesión con usuario y contraseña.
    python auth_system.py help
        Muestra esta ayuda.
"""
    print(help_text)
