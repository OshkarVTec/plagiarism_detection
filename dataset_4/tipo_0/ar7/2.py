#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de chat simple punto a punto usando sockets en Python.
Incluye servidor y cliente. El chat está basado en texto plano.
"""

import socket
import threading
import sys

# Puerto por defecto donde el servidor escuchará conexiones
DEFAULT_PORT = 5000
# Tamaño del buffer para recibir datos
BUFFER_SIZE = 1024


def broadcast(message, sender_socket, clients):
    """
    Envía el mensaje a todos los clientes conectados excepto al remitente.
    """
    for client in clients:
        if client != sender_socket:
            try:
                client.send(message)
            except Exception:
                client.close()
                clients.remove(client)


def handle_client(client_socket, clients):
    """
    Función que maneja la comunicación con un cliente conectado.
    Recibe mensajes y los retransmite (broadcast).
    """
    while True:
        try:
            message = client_socket.recv(BUFFER_SIZE)
            if not message:
                break
            broadcast(message, client_socket, clients)
        except Exception:
            break

    # Cuando el cliente se desconecta, lo eliminamos de la lista
    client_socket.close()
    if client_socket in clients:
        clients.remove(client_socket)


def start_server(host="0.0.0.0", port=DEFAULT_PORT):
    """
    Inicia el servidor de chat. Acepta múltiples clientes y crea un hilo por cada uno.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"[Servidor] Escuchando en {host}:{port}")

    clients = []

    try:
        while True:
            client_socket, addr = server.accept()
            print(f"[Conexión] Cliente conectado: {addr}")
            clients.append(client_socket)
            thread = threading.Thread(target=handle_client, args=(client_socket, clients))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print("\n[Servidor] Cerrando conexiones y terminando.")
    finally:
        for c in clients:
            c.close()
        server.close()


def start_client(server_ip, server_port):
    """
    Inicia el cliente de chat. Se conecta al servidor y lanza dos hilos:
    uno para enviar mensajes y otro para recibirlos.
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((server_ip, server_port))
    except Exception as e:
        print(f"No se pudo conectar al servidor: {e}")
        return

    print(f"Conectado al servidor {server_ip}:{server_port}")

    def receive_messages():
        """Hilo para recibir mensajes desde el servidor."""
        while True:
            try:
                message = client.recv(BUFFER_SIZE)
                if not message:
                    break
                print("\n" + message.decode("utf-8"))
            except Exception:
                break

    def send_messages():
        """Hilo para enviar mensajes al servidor."""
        while True:
            msg = input()
            if msg.lower() == "exit":
                client.close()
                break
            try:
                client.send(msg.encode("utf-8"))
            except Exception:
                break

    recv_thread = threading.Thread(target=receive_messages)
    recv_thread.daemon = True
    recv_thread.start()

    send_thread = threading.Thread(target=send_messages)
    send_thread.daemon = True
    send_thread.start()

    recv_thread.join()
    send_thread.join()
    print("Conexión cerrada.")

