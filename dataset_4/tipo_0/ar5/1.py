#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestor de tareas en línea de comandos.
Permite agregar, listar, marcar como completada y eliminar tareas.
Las tareas se guardan en un archivo JSON.
"""

import json
import os
import sys
from datetime import datetime

# Ruta del archivo donde se almacenan las tareas
TASKS_FILE = "tasks.json"


def load_tasks():
    """Carga las tareas desde el archivo JSON. Si no existe, retorna lista vacía."""
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_tasks(tasks):
    """Guarda la lista de tareas en el archivo JSON."""
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4, ensure_ascii=False)


def list_tasks(tasks):
    """Imprime todas las tareas, indicando su estado (completa o pendiente)."""
    if not tasks:
        print("No hay tareas registradas.")
        return

    print("\nListado de tareas:")
    for idx, task in enumerate(tasks, start=1):
        status = "✅" if task["completed"] else "❌"
        date_str = task["created_at"]
        print(f"{idx}. [{status}] {task['title']} (creada: {date_str})")
    print("")


def add_task(tasks, title):
    """Agrega una nueva tarea con el título dado."""
    task = {
        "title": title,
        "completed": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tasks.append(task)
    save_tasks(tasks)
    print(f"Tarea agregada: {title}")


def complete_task(tasks, index):
    """Marca como completada la tarea en la posición index (1-based)."""
    try:
        task = tasks[index - 1]
        if task["completed"]:
            print("La tarea ya estaba marcada como completada.")
        else:
            task["completed"] = True
            save_tasks(tasks)
            print(f"Tarea marcada como completada: {task['title']}")
    except IndexError:
        print("Índice de tarea inválido.")


def delete_task(tasks, index):
    """Elimina la tarea en la posición index (1-based)."""
    try:
        task = tasks.pop(index - 1)
        save_tasks(tasks)
        print(f"Tarea eliminada: {task['title']}")
    except IndexError:
        print("Índice de tarea inválido.")


def print_help():
    """Muestra la ayuda sobre los comandos disponibles."""
    help_text = """
Gestor de tareas (To-Do List)
Uso:
    python task_manager.py list
        Lista todas las tareas.
    python task_manager.py add "Título de la tarea"
        Agrega una nueva tarea con el título especificado.
    python task_manager.py complete <número>
        Marca como completada la tarea con el índice dado.
    python task_manager.py delete <número>
        Elimina la tarea con el índice dado.
    python task_manager.py help
        Muestra esta ayuda.
    """
    print(help_text)


def main():
    tasks = load_tasks()

    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "list":
        list_tasks(tasks)
    elif command == "add":
        if len(sys.argv) < 3:
            print("Falta el título de la tarea.")
        else:
            title = " ".join(sys.argv[2:])
            add_task(tasks, title)
    elif command == "complete":
        if len(sys.argv) != 3 or not sys.argv[2].isdigit():
            print("Debes indicar el número de tarea a completar.")
        else:
            index = int(sys.argv[2])
            complete_task(tasks, index)
    elif command == "delete":
        if len(sys.argv) != 3 or not sys.argv[2].isdigit():
            print("Debes indicar el número de tarea a eliminar.")
        else:
            index = int(sys.argv[2])
            delete_task(tasks, index)
    elif command == "help":
        print_help()
    else:
        print(f"Comando desconocido: {command}")
        print_help()

