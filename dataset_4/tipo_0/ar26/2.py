#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de un directorio en tiempo real usando watchdog:
  - Observa eventos: creación, modificación, eliminación de archivos.
  - Ejecuta acciones al detectar cambios (por ejemplo, copiar nuevos archivos
    a un directorio de respaldo).
  - Registra eventos en un archivo de log.
"""

import sys
import os
import time
import argparse
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SyncEventHandler(FileSystemEventHandler):
    """
    Manejador de eventos que:
      - Al crear o modificar un archivo, lo copia al directorio de respaldo.
      - Al eliminar un archivo, elimina el correspondiente en respaldo.
      - Registra cada evento en un archivo de log.
    """

    def __init__(self, src_dir, backup_dir, log_file):
        self.src_dir = src_dir
        self.backup_dir = backup_dir
        self.log_file = log_file

    def log_event(self, event_type, src_path):
        """
        Registra en log_file la acción realizada con timestamp.
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        rel_path = os.path.relpath(src_path, self.src_dir)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {event_type}: {rel_path}\n")

    def on_created(self, event):
        """
        Evento creación de archivo/directorio.
        Si es archivo, lo copia a backup.
        """
        if not event.is_directory:
            rel_path = os.path.relpath(event.src_path, self.src_dir)
            dest_path = os.path.join(self.backup_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                shutil.copy2(event.src_path, dest_path)
                self.log_event('CREADO', event.src_path)
            except Exception as e:
                print(f"[ERROR] No se pudo copiar {event.src_path} a respaldo: {e}")

    def on_modified(self, event):
        """
        Evento modificación de archivo.
        Copia nuevo contenido al respaldo.
        """
        if not event.is_directory:
            rel_path = os.path.relpath(event.src_path, self.src_dir)
            dest_path = os.path.join(self.backup_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                shutil.copy2(event.src_path, dest_path)
                self.log_event('MODIFICADO', event.src_path)
            except Exception as e:
                print(f"[ERROR] No se pudo actualizar {event.src_path} en respaldo: {e}")

    def on_deleted(self, event):
        """
        Evento eliminación de archivo.
        Elimina archivo correspondiente en respaldo.
        """
        if not event.is_directory:
            rel_path = os.path.relpath(event.src_path, self.src_dir)
            dest_path = os.path.join(self.backup_dir, rel_path)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                self.log_event('ELIMINADO', event.src_path)
            except Exception as e:
                print(f"[ERROR] No se pudo eliminar {dest_path}: {e}")

