#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conjunto de dos scripts (cliente y servidor) para transferir archivos mediante TCP:
  - Servidor:
      * Escucha en un puerto específico.
      * Al recibir conexión, espera recibir nombre de archivo.
      * Envía el archivo al cliente en bloques.
      * Cierra conexión al finalizar.
  - Cliente:
      * Se conecta al servidor con IP y puerto.
      * Envía nombre de archivo que desea descargar.
      * Recibe datos en bloques y los escribe localmente.
      * Cierra conexión al finalizar.
"""

# -----------------------------------------------------------------------
#                       Servidor TCP para archivos
# -----------------------------------------------------------------------

import os
import socket
import threading

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5001
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

def handle_client(client_socket, client_address):
    """
    Maneja la conexión con un cliente:
      - Recibe mensaje con formato: '<filename><SEPARATOR><filesize>'.
      - Envía el contenido del archivo en bloques de tamaño BUFFER_SIZE.
      - Cierra conexión al terminar.
    """
    print(f"[INFO] Cliente conectado: {client_address}")
    try:
        # Leer mensaje inicial
        received = client_socket.recv(BUFFER_SIZE).decode()
        filename, filesize = received.split(SEPARATOR)
        filename = os.path.basename(filename)
        filesize = int(filesize)
        print(f"[INFO] Cliente solicita archivo: {filename} ({filesize} bytes)")

        if not os.path.isfile(filename):
            error_msg = f"ERROR: El archivo '{filename}' no existe."
            client_socket.send(error_msg.encode())
            print(f"[WARN] {error_msg}")
        else:
            # Enviar archivo en bloques
            with open(filename, 'rb') as f:
                while True:
                    bytes_read = f.read(BUFFER_SIZE)
                    if not bytes_read:
                        break
                    client_socket.sendall(bytes_read)
            print(f"[INFO] Envío de '{filename}' completado.")
    except Exception as e:
        print(f"[ERROR] Falló la transferencia: {e}")
    finally:
        client_socket.close()
        print(f"[INFO] Conexión cerrada: {client_address}")

def start_file_server():
    """
    Inicia el servidor TCP en SERVER_HOST:SERVER_PORT.
    Para cada conexión, crea un hilo que llama a handle_client.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"[INFO] Servidor de archivos escuchando en {SERVER_HOST}:{SERVER_PORT}")

    try:
        while True:
            client_sock, client_addr = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_sock, client_addr)
            )
            client_thread.start()
    except KeyboardInterrupt:
        print("\n[INFO] Servidor interrumpido por el usuario.")
    finally:
        server_socket.close()

# -----------------------------------------------------------------------
#                       Cliente TCP para archivos
# -----------------------------------------------------------------------

def download_file(server_ip, server_port, filename):
    """
    Conecta al servidor y solicita `filename`.
    Recibe datos y los escribe en un nuevo archivo local.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, server_port))
    except Exception as e:
        print(f"[ERROR] No se pudo conectar al servidor: {e}")

    if not os.path.isfile(filename):
        print(f"[ERROR] El archivo '{filename}' no está disponible localmente. Solicitándolo al servidor.")
    filesize = os.path.getsize(filename) if os.path.exists(filename) else 0

    # Enviar solicitud: 'filename<SEPARATOR><filesize>'
    message = f"{filename}{SEPARATOR}{filesize}"
    client_socket.send(message.encode())

    # Abrir archivo de destino para escribir
    dest_filename = "descargado_" + filename
    with open(dest_filename, 'wb') as f:
        while True:
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:
                # Transferencia finalizada
                break
            f.write(bytes_read)

    print(f"[INFO] Archivo recibido y guardado como '{dest_filename}'")
    client_socket.close()
