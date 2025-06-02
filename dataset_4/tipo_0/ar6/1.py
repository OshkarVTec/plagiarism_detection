#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST básica con Flask para gestionar notas (CRUD).
Utiliza SQLite como base de datos local.
"""

from flask import Flask, request, jsonify, g
import sqlite3
import os

# Nombre del archivo de la base de datos SQLite
DATABASE = "notes.db"

app = Flask(__name__)


def get_db():
    """Obtiene la conexión a la base de datos SQLite, almacenada en el contexto 'g'."""
    db = getattr(g, "_database", None)
    if db is None:
        db = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # Para obtener resultados como diccionarios
        g._database = db
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Cierra la conexión a la base de datos al finalizar el request."""
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    """Inicializa la tabla de notas si no existe."""
    if not os.path.exists(DATABASE):
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
        print("Base de datos inicializada.")


@app.route("/notes", methods=["GET"])
def get_all_notes():
    """Devuelve todas las notas en formato JSON."""
    db = get_db()
    cursor = db.execute("SELECT * FROM notes ORDER BY created_at DESC;")
    rows = cursor.fetchall()
    notes = [dict(row) for row in rows]
    return jsonify(notes), 200


@app.route("/notes/<int:note_id>", methods=["GET"])
def get_note(note_id):
    """Devuelve una nota específica por su ID."""
    db = get_db()
    cursor = db.execute("SELECT * FROM notes WHERE id = ?;", (note_id,))
    row = cursor.fetchone()
    if row is None:
        return jsonify({"error": "Nota no encontrada"}), 404
    return jsonify(dict(row)), 200


@app.route("/notes", methods=["POST"])
def create_note():
    """Crea una nueva nota. Espera 'title' y 'content' en JSON del body."""
    data = request.get_json()
    if not data or "title" not in data or "content" not in data:
        return jsonify({"error": "Faltan campos 'title' o 'content'."}), 400

    title = data["title"].strip()
    content = data["content"].strip()
    if not title or not content:
        return jsonify({"error": "Los campos no pueden estar vacíos."}), 400

    db = get_db()
    cursor = db.execute(
        "INSERT INTO notes (title, content) VALUES (?, ?);",
        (title, content)
    )
    db.commit()
    new_id = cursor.lastrowid
    return jsonify({"message": "Nota creada", "id": new_id}), 201


@app.route("/notes/<int:note_id>", methods=["PUT"])
def update_note(note_id):
    """Actualiza una nota existente. Se pueden actualizar title y/o content."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Body vacío"}), 400

    db = get_db()
    cursor = db.execute("SELECT * FROM notes WHERE id = ?;", (note_id,))
    if cursor.fetchone() is None:
        return jsonify({"error": "Nota no encontrada"}), 404

    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    if not title and not content:
        return jsonify({"error": "Al menos 'title' o 'content' debe proporcionarse."}), 400

    if title:
        db.execute("UPDATE notes SET title = ? WHERE id = ?;", (title, note_id))
    if content:
        db.execute("UPDATE notes SET content = ? WHERE id = ?;", (content, note_id))
    db.commit()

    return jsonify({"message": "Nota actualizada"}), 200


@app.route("/notes/<int:note_id>", methods=["DELETE"])
def delete_note(note_id):
    """Elimina una nota existente por su ID."""
    db = get_db()
    cursor = db.execute("SELECT * FROM notes WHERE id = ?;", (note_id,))
    if cursor.fetchone() is None:
        return jsonify({"error": "Nota no encontrada"}), 404

    db.execute("DELETE FROM notes WHERE id = ?;", (note_id,))
    db.commit()
    return jsonify({"message": "Nota eliminada"}), 200
