#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST CRUD simple con Flask para manejar “tareas” en memoria.
Cada tarea tiene:
  - id (entero autoincremental)
  - title (string)
  - done (booleano)
Endpoints:
  - GET /tasks: retorna lista JSON de todas las tareas.
  - GET /tasks/<int:id>: retorna la tarea con ese id o 404.
  - POST /tasks: crea nueva tarea. JSON esperado: { "title": ..., "done": false }
    Retorna JSON de la tarea creada con campo “id”.
  - PUT /tasks/<int:id>: actualiza tarea. JSON puede contener “title” y/o “done”.
  - DELETE /tasks/<int:id>: elimina tarea o retorna 404 si no existe.
Para correr: `FLASK_APP=archivo flask run` o `python archivo.py`.
"""

from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# Base de datos en memoria: lista de dicts
tasks = []
next_id = 1

def find_task(task_id):
    """
    Busca tarea en tasks por id. Retorna tarea o None.
    """
    for t in tasks:
        if t['id'] == task_id:
            return t
    return None

@app.route('/tasks', methods=['GET'])
def get_tasks():
    """
    Retorna JSON con lista de todas las tareas.
    """
    return jsonify(tasks)

@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """
    Retorna JSON de la tarea con id=task_id o 404 si no existe.
    """
    task = find_task(task_id)
    if task is None:
        abort(404)
    return jsonify(task)

@app.route('/tasks', methods=['POST'])
def create_task():
    """
    Crea una nueva tarea. JSON debe tener “title” y opcionalmente “done”.
    Asigna id autoincremental. Retorna la tarea creada (201).
    """
    global next_id
    if not request.json or 'title' not in request.json:
        abort(400)
    title = request.json['title']
    done = request.json.get('done', False)
    if not isinstance(title, str) or not isinstance(done, bool):
        abort(400)
    task = {
        'id': next_id,
        'title': title,
        'done': done
    }
    next_id += 1
    tasks.append(task)
    return jsonify(task), 201

@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    """
    Actualiza tarea con id=task_id. JSON puede tener “title” y/o “done”.
    Valida tipos y retorna tarea actualizada o 404/400.
    """
    task = find_task(task_id)
    if task is None:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and not isinstance(request.json['title'], str):
        abort(400)
    if 'done' in request.json and not isinstance(request.json['done'], bool):
        abort(400)
    task['title'] = request.json.get('title', task['title'])
    task['done'] = request.json.get('done', task['done'])
    return jsonify(task)

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """
    Elimina tarea con id=task_id o retorna 404 si no existe. Retorna {}.
    """
    task = find_task(task_id)
    if task is None:
        abort(404)
    tasks.remove(task)
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)
