#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor HTTP estático básico usando sockets.
Sirve archivos desde el directorio 'www' en el puerto especificado.
Si no encuentra el archivo solicitado, devuelve 404.
"""

import socket
import threading
import os
import sys
import mimetypes

# Carpeta raíz para archivos estáticos
WWW_ROOT = "www"
# Puerto por defecto para servidor HTTP
DEFAULT_PORT = 8080
# Tamaño del buffer para recibir solicitudes
BUFFER_SIZE = 1024

def handle_client_connection(client_socket, client_address):
    """
    Maneja la conexión de un cliente HTTP:
    1. Lee la solicitud.
    2. Parsea línea inicial para obtener método y ruta.
    3. Busca archivo en WWW_ROOT; si existe, lo envía con código 200.
    4. Si no existe, envía 404.
    Finalmente cierra la conexión.
    """
    try:
        request_data = client_socket.recv(BUFFER_SIZE).decode("utf-8", errors="ignore")
        if not request_data:
            client_socket.close()
            return

        # Obtener la primera línea, ejemplo: "GET /index.html HTTP/1.1"
        request_line = request_data.splitlines()[0]
        parts = request_line.split()
        if len(parts) < 2:
            client_socket.close()
            return

        method, path = parts[0], parts[1]
        print(f"[{client_address}] {method} {path}")

        if method != "GET":
            # Solo servimos GET
            response = "HTTP/1.1 405 Method Not Allowed\r\n"
            response += "Connection: close\r\n\r\n"
            client_socket.sendall(response.encode("utf-8"))
            client_socket.close()
            return

        # Normalizar ruta: si es '/', servir 'index.html'
        if path == "/":
            path = "/index.html"

        # Evitar rutas fuera de www
        safe_path = os.path.normpath(path).lstrip(os.sep)
        file_path = os.path.join(WWW_ROOT, safe_path)

        if os.path.isfile(file_path):
            # Determinar tipo MIME
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            with open(file_path, "rb") as f:
                content = f.read()

            response_headers = [
                "HTTP/1.1 200 OK",
                f"Content-Type: {mime_type}",
                f"Content-Length: {len(content)}",
                "Connection: close",
                "\r\n"
            ]
            header_data = "\r\n".join(response_headers).encode("utf-8")
            client_socket.sendall(header_data + content)
        else:
            # Archivo no encontrado
            body = "<h1>404 Not Found</h1><p>El recurso no existe.</p>"
            response_headers = [
                "HTTP/1.1 404 Not Found",
                "Content-Type: text/html; charset=utf-8",
                f"Content-Length: {len(body.encode('utf-8'))}",
                "Connection: close",
                "\r\n"
            ]
            header_data = "\r\n".join(response_headers).encode("utf-8")
            client_socket.sendall(header_data + body.encode("utf-8"))
    except Exception as e:
        print(f"[ERROR] al manejar la conexión: {e}")
    finally:
        client_socket.close()

def start_server(port):
    """
    Inicializa el socket servidor en el puerto especificado y
    maneja cada cliente en un hilo separado.
    """
    if not os.path.isdir(WWW_ROOT):
        print(f"[ERROR] No existe la carpeta '{WWW_ROOT}'. Crea archivos estáticos allí.")
        sys.exit(1)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Permitir reutilizar dirección para reinicios rápidos
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(5)
    print(f"[INFO] Servidor HTTP iniciado en el puerto {port}. Serviendo carpeta '{WWW_ROOT}'.")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            thread = threading.Thread(target=handle_client_connection,
                                      args=(client_socket, client_address))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print("\n[INFO] Servidor detenido por usuario.")
    finally:
        server_socket.close()

def print_usage():
    """
    Muestra instrucción de uso para el script.
    """
    print(f"Uso: python static_http_server.py [puerto]")
    print(f"Si no se especifica puerto, se usará {DEFAULT_PORT}.")

