#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor de chat en tiempo real usando Flask y Flask-SocketIO.
Permite múltiples usuarios conectados simultáneamente.
Mensajes son emitidos a todos los clientes conectados.
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
import os

# Crear la aplicación Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # Clave de sesión para Flask
socketio = SocketIO(app)

# Ruta raíz que sirve la plantilla HTML para el chat
@app.route('/')
def index():
    return render_template('chat.html')

# Evento cuando un cliente se conecta
@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Cliente conectado: {os.urandom(4).hex()}")
    send({'msg': 'Un nuevo usuario ha entrado al chat.'}, broadcast=True)

# Evento cuando un cliente envía un mensaje
@socketio.on('message')
def handle_message(data):
    """
    `data` es un diccionario con:
      - 'username': nombre de usuario que envía.
      - 'msg': contenido del mensaje.
    Emite el mensaje a todos los clientes.
    """
    username = data.get('username', 'Anon')
    msg = data.get('msg', '')
    print(f"[{username}] dice: {msg}")
    emit('message', {'username': username, 'msg': msg}, broadcast=True)

# Evento cuando un cliente se desconecta
@socketio.on('disconnect')
def handle_disconnect():
    print("[INFO] Cliente desconectado.")
    send({'msg': 'Un usuario ha salido del chat.'}, broadcast=True)

@socketio.on('message')
def handle_message(data):
    """
    `data` es un diccionario con:
      - 'username': nombre de usuario que envía.
      - 'msg': contenido del mensaje.
    Emite el mensaje a todos los clientes.
    """
    username = data.get('username', 'Anon')
    msg = data.get('msg', '')
    print(f"[{username}] dice: {msg}")
    emit('message', {'username': username, 'msg': msg}, broadcast=True)

# Evento cuando un cliente se desconecta
@socketio.on('disconnect')
def handle_disconnect():
    print("[INFO] Cliente desconectado.")
    send({'msg': 'Un usuario ha salido del chat.'}, broadcast=True)

@socketio.on('message')
def handle_message(data):
    """
    `data` es un diccionario con:
      - 'username': nombre de usuario que envía.
      - 'msg': contenido del mensaje.
    Emite el mensaje a todos los clientes.
    """
    username = data.get('username', 'Anon')
    msg = data.get('msg', '')
    print(f"[{username}] dice: {msg}")
    emit('message', {'username': username, 'msg': msg}, broadcast=True)

# Evento cuando un cliente se desconecta
@socketio.on('disconnect')
def handle_disconnect():
    print("[INFO] Cliente desconectado.")
    send({'msg': 'Un usuario ha salido del chat.'}, broadcast=True)

@socketio.on('message')
def handle_message(data):
    """
    `data` es un diccionario con:
      - 'username': nombre de usuario que envía.
      - 'msg': contenido del mensaje.
    Emite el mensaje a todos los clientes.
    """
    username = data.get('username', 'Anon')
    msg = data.get('msg', '')
    print(f"[{username}] dice: {msg}")
    emit('message', {'username': username, 'msg': msg}, broadcast=True)

# Evento cuando un cliente se desconecta
@socketio.on('disconnect')
def handle_disconnect():
    print("[INFO] Cliente desconectado.")
    send({'msg': 'Un usuario ha salido del chat.'}, broadcast=True)

