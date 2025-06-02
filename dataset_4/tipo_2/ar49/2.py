#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variante Tipo 2 de CRUD API con Flask (tareas en memoria):
Cada “tarea” es un dict con:
  - identificador ‘id’ (incremental)
  - ‘titulo’ (string)
  - ‘completada’ (booleano)
Endpoints:
  - GET   /todos          → lista todas las tareas en JSON
  - GET   /todos/<int:id> → tarea con ese id o 404
  - POST  /todos          → crea tarea (JSON: { "titulo": ..., "completada": false })
                             Retorna la tarea creada con “id”.
  - PUT   /todos/<int:id> → actualiza campos “titulo” y/o “completada”
  - DELETE /todos/<int:id>→ elimina tarea o 404
Para ejecutar: `FLASK_APP=archivo flask run` o `python archivo.py`.
"""

from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# “Base de datos” en memoria
lista_tareas = []
proximo_id = 1

def obtener_tarea(tarea_id):
    """
    Busca tarea en lista_tareas, retorna el dict o None si no existe.
    """
    for t in lista_tareas:
        if t['id'] == tarea_id:
            return t
    return None

@app.route('/todos', methods=['GET'])
def listar_tareas():
    """
    Devuelve JSON con lista de todas las tareas.
    """
    return jsonify(lista_tareas)

@app.route('/todos/<int:tarea_id>', methods=['GET'])
def obtener_una_tarea(tarea_id):
    """
    Retorna JSON de tarea con id=tarea_id o 404.
    """
    tarea = obtener_tarea(tarea_id)
    if tarea is None:
        abort(404)
    return jsonify(tarea)

@app.route('/todos', methods=['POST'])
def agregar_tarea():
    """
    Crea nueva tarea. JSON debe tener “titulo”. Campo “completada” opcional (default False).
    Valida tipos, asigna id. Retorna tarea creada con 201.
    """
    global proximo_id
    if not request.json or 'titulo' not in request.json:
        abort(400)
    titulo = request.json['titulo']
    completada = request.json.get('completada', False)
    if not isinstance(titulo, str) or not isinstance(completada, bool):
        abort(400)
    nueva = {
        'id': proximo_id,
        'titulo': titulo,
        'completada': completada
    }
    proximo_id += 1
    lista_tareas.append(nueva)
    return jsonify(nueva), 201

@app.route('/todos/<int:tarea_id>', methods=['PUT'])
def modificar_tarea(tarea_id):
    """
    Actualiza tarea con id=tarea_id. JSON puede contener “titulo” y “completada”.
    Valida tipos. Retorna tarea actualizada o 404/400.
    """
    tarea = obtener_tarea(tarea_id)
    if tarea is None:
        abort(404)
    if not request.json:
        abort(400)
    if 'titulo' in request.json and not isinstance(request.json['titulo'], str):
        abort(400)
    if 'completada' in request.json and not isinstance(request.json['completada'], bool):
        abort(400)
    tarea['titulo'] = request.json.get('titulo', tarea['titulo'])
    tarea['completada'] = request.json.get('completada', tarea['completada'])
    return jsonify(tarea)

@app.route('/todos/<int:tarea_id>', methods=['DELETE'])
def eliminar_tarea(tarea_id):
    """
    Elimina tarea con id=tarea_id o 404 si no existe. Retorna {}.
    """
    tarea = obtener_tarea(tarea_id)
    if tarea is None:
        abort(404)
    lista_tareas.remove(tarea)
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)
