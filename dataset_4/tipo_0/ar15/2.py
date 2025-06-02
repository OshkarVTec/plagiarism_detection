#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestor de biblioteca en consola usando SQLite.
Permite:
  - Registrar nuevos libros (Título, Autor, Año, ISBN).
  - Listar todos los libros.
  - Buscar libros por título o autor.
  - Registrar préstamos y devoluciones.
  - Listar libros prestados y disponibles.
Las tablas:
  - books(id, title, author, year, isbn, available)
  - loans(id, book_id, borrower, loan_date, return_date)
"""

import sqlite3
import sys
import os
from datetime import datetime

DB_FILE = "library.db"

def get_connection():
    """
    Retorna una conexión a la base de datos SQLite.
    Si el archivo no existe, lo crea automáticamente.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """
    Crea tablas 'books' y 'loans' si no existen.
    """
    if not os.path.exists(DB_FILE):
        print("[INFO] Creando base de datos y tablas iniciales...")
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        year INTEGER NOT NULL,
        isbn TEXT UNIQUE NOT NULL,
        available INTEGER DEFAULT 1
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER NOT NULL,
        borrower TEXT NOT NULL,
        loan_date TEXT NOT NULL,
        return_date TEXT,
        FOREIGN KEY(book_id) REFERENCES books(id)
    );
    """)
    conn.commit()
    conn.close()

def print_help():
    """
    Muestra opciones disponibles en el gestor de biblioteca.
    """
    help_text = """
=== Menú Biblioteca ===
1. Agregar nuevo libro
2. Listar todos los libros
3. Buscar libros (título/autor)
4. Registrar préstamo
5. Registrar devolución
6. Listar préstamos activos
7. Listar libros disponibles
8. Salir
"""
    print(help_text)

def add_book():
    """
    Solicita datos del libro y lo inserta en la tabla 'books'.
    """
    title = input("Título: ").strip()
    author = input("Autor: ").strip()
    try:
        year = int(input("Año de publicación: ").strip())
    except ValueError:
        print("[ERROR] Año inválido.")
        return
    isbn = input("ISBN: ").strip()

    if not title or not author or not isbn:
        print("[ERROR] Título, Autor e ISBN no pueden estar vacíos.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO books(title, author, year, isbn)
        VALUES (?, ?, ?, ?);
        """, (title, author, year, isbn))
        conn.commit()
        print(f"[OK] Libro agregado: '{title}' de {author}.")
    except sqlite3.IntegrityError:
        print("[ERROR] ISBN duplicado. Ya existe un libro con ese ISBN.")
    finally:
        conn.close()

def list_all_books():
    """
    Muestra todos los libros (incluyendo si están disponibles o no).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM books;")
    rows = cursor.fetchall()
    if not rows:
        print("[INFO] No hay libros en la biblioteca.")
        conn.close()
        return

    print(f"\n{'ID':<4} {'Título':<30} {'Autor':<20} {'Año':<5} {'ISBN':<15} {'Disp.':<5}")
    print("-" * 85)
    for row in rows:
        disponible = "Sí" if row["available"] == 1 else "No"
        print(f"{row['id']:<4} {row['title']:<30} {row['author']:<20} {row['year']:<5} {row['isbn']:<15} {disponible:<5}")
    print("")
    conn.close()

def search_books():
    """
    Busca libros por coincidencia parcial en título o autor.
    """
    term = input("Ingrese término de búsqueda (título o autor): ").strip().lower()
    if not term:
        print("[ERROR] El término no puede estar vacío.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM books
    WHERE LOWER(title) LIKE ? OR LOWER(author) LIKE ?;
    """, (f"%{term}%", f"%{term}%"))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[INFO] No se encontraron libros que coincidan.")
        return

    print(f"\nResultados de búsqueda para '{term}':")
    print(f"{'ID':<4} {'Título':<30} {'Autor':<20} {'Año':<5} {'ISBN':<15} {'Disp.':<5}")
    print("-" * 85)
    for row in rows:
        disponible = "Sí" if row["available"] == 1 else "No"
        print(f"{row['id']:<4} {row['title']:<30} {row['author']:<20} {row['year']:<5} {row['isbn']:<15} {disponible:<5}")
    print("")

def register_loan():
    """
    Registra un préstamo de libro:
    - Solicita ID de libro y nombre del prestatario.
    - Verifica disponibilidad y crea entrada en 'loans', actualiza 'books'.
    """
    try:
        book_id = int(input("ID del libro a prestar: ").strip())
    except ValueError:
        print("[ERROR] ID inválido.")
        return
    borrower = input("Nombre del prestatario: ").strip()
    if not borrower:
        print("[ERROR] El nombre del prestatario no puede estar vacío.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    # Verificar existencia y disponibilidad
    cursor.execute("SELECT * FROM books WHERE id = ?;", (book_id,))
    book = cursor.fetchone()
    if not book:
        print("[ERROR] No existe libro con ese ID.")
        conn.close()
        return
    if book["available"] == 0:
        print("[INFO] El libro no está disponible para préstamo.")
        conn.close()
        return

    loan_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Insertar en loans y marcar libro como no disponible
        cursor.execute("""
        INSERT INTO loans(book_id, borrower, loan_date)
        VALUES (?, ?, ?);
        """, (book_id, borrower, loan_date))
        cursor.execute("UPDATE books SET available = 0 WHERE id = ?;", (book_id,))
        conn.commit()
        print(f"[OK] Préstamo registrado: Libro ID {book_id} prestado a '{borrower}'.")
    except Exception as e:
        print(f"[ERROR] Falló el registro de préstamo: {e}")
    finally:
        conn.close()

def register_return():
    """
    Registra la devolución de un libro:
    - Solicita ID del préstamo o ID de libro.
    - Marca return_date y actualiza disponibilidad en 'books'.
    """
    try:
        book_id = int(input("ID del libro a devolver: ").strip())
    except ValueError:
        print("[ERROR] ID inválido.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    # Buscar préstamo activo sin return_date para ese libro
    cursor.execute("""
    SELECT * FROM loans
    WHERE book_id = ? AND return_date IS NULL;
    """, (book_id,))
    loan = cursor.fetchone()
    if not loan:
        print("[INFO] No hay préstamo activo para el libro ID {book_id}.")
        conn.close()
        return

    return_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
        UPDATE loans SET return_date = ?
        WHERE id = ?;
        """, (return_date, loan["id"]))
        cursor.execute("UPDATE books SET available = 1 WHERE id = ?;", (book_id,))
        conn.commit()
        print(f"[OK] Devolución registrada para libro ID {book_id}.")
    except Exception as e:
        print(f"[ERROR] No se pudo registrar la devolución: {e}")
    finally:
        conn.close()

def list_active_loans():
    """
    Lista todos los préstamos que no han sido devueltos (return_date IS NULL).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT l.id, b.title, l.borrower, l.loan_date
    FROM loans l
    JOIN books b ON l.book_id = b.id
    WHERE l.return_date IS NULL;
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[INFO] No hay préstamos activos.")
        return

    print(f"\n{'Loan ID':<8} {'Título':<30} {'Prestatario':<20} {'Fecha préstamo':<20}")
    print("-" * 80)
    for row in rows:
        print(f"{row['id']:<8} {row['title']:<30} {row['borrower']:<20} {row['loan_date']:<20}")
    print("")

def list_available_books():
    """
    Lista todos los libros que están marcados como disponibles (available = 1).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM books WHERE available = 1;")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[INFO] No hay libros disponibles actualmente.")
        return

    print(f"\n{'ID':<4} {'Título':<30} {'Autor':<20} {'Año':<5} {'ISBN':<15}")
    print("-" * 80)
    for row in rows:
        print(f"{row['id']:<4} {row['title']:<30} {row['author']:<20} {row['year']:<5} {row['isbn']:<15}")
    print("")

