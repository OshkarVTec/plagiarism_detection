#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliente FTP interactivo.
Permite conectar a un servidor FTP, listar directorios, cambiar carpetas,
descargar y subir archivos desde línea de comandos.
"""

import ftplib
import os
import sys
import getpass

# Fin de lista de comandos (para cerrar sesión y salir)
EXIT_COMMANDS = {"exit", "quit", "bye"}

def print_help():
    """
    Muestra la ayuda de los comandos disponibles en la consola FTP.
    """
    help_text = """
Comandos FTP interactivos:
    ls                   -> Lista archivos y carpetas en el directorio actual
    cd <directorio>      -> Cambia al directorio especificado en el servidor
    pwd                  -> Muestra el directorio actual en el servidor
    get <archivo>        -> Descarga archivo del servidor al local
    put <archivo>        -> Sube archivo local al servidor
    mkdir <directorio>   -> Crea un nuevo directorio en el servidor
    delete <archivo>     -> Elimina el archivo especificado en el servidor
    rmdir <directorio>   -> Elimina el directorio especificado en el servidor (debe estar vacío)
    help                 -> Muestra esta ayuda
    exit | quit | bye    -> Cierra sesión y sale del cliente FTP
    """
    print(help_text)

def connect_ftp(server, port, username, password):
    """
    Conecta al servidor FTP y retorna el objeto ftplib.FTP.
    """
    try:
        print(f"[INFO] Conectando a FTP {server}:{port}...")
        ftp = ftplib.FTP()
        ftp.connect(server, port, timeout=10)
        ftp.login(username, password)
        print(f"[OK] Autenticación exitosa. Bienvenido, {username}!")
        return ftp
    except ftplib.all_errors as e:
        print(f"[ERROR] No se pudo conectar/autenticar: {e}")
        sys.exit(1)

def interactive_ftp(ftp):
    """
    Bucle principal para recibir y ejecutar comandos del usuario.
    """
    print("Escribe 'help' para ver comandos disponibles.")
    while True:
        try:
            cmd = input("ftp> ").strip()
        except EOFError:
            # Ctrl+D finaliza el cliente
            print("\n[INFO] Cerrando sesión.")
            break

        if not cmd:
            continue

        parts = cmd.split()
        command = parts[0].lower()

        if command in EXIT_COMMANDS:
            print("[INFO] Saliendo del cliente FTP.")
            break

        elif command == "help":
            print_help()

        elif command == "ls":
            try:
                ftp.retrlines("LIST")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        elif command == "pwd":
            try:
                cwd = ftp.pwd()
                print(f"Directorio actual en servidor: {cwd}")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        elif command == "cd":
            if len(parts) != 2:
                print("[ERROR] Uso: cd <directorio>")
                continue
            try:
                ftp.cwd(parts[1])
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        elif command == "get":
            if len(parts) != 2:
                print("[ERROR] Uso: get <archivo>")
                continue
            remote_file = parts[1]
            local_file = os.path.basename(remote_file)
            try:
                with open(local_file, "wb") as f:
                    ftp.retrbinary(f"RETR {remote_file}", f.write)
                print(f"[OK] Archivo descargado: {local_file}")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")
            except Exception as e:
                print(f"[ERROR] No se pudo escribir localmente: {e}")

        elif command == "put":
            if len(parts) != 2:
                print("[ERROR] Uso: put <archivo>")
                continue
            local_file = parts[1]
            if not os.path.isfile(local_file):
                print(f"[ERROR] No existe el archivo local: {local_file}")
                continue
            remote_file = os.path.basename(local_file)
            try:
                with open(local_file, "rb") as f:
                    ftp.storbinary(f"STOR {remote_file}", f)
                print(f"[OK] Archivo subido: {remote_file}")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        elif command == "mkdir":
            if len(parts) != 2:
                print("[ERROR] Uso: mkdir <directorio>")
                continue
            try:
                ftp.mkd(parts[1])
                print(f"[OK] Directorio creado: {parts[1]}")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        elif command == "rmdir":
            if len(parts) != 2:
                print("[ERROR] Uso: rmdir <directorio>")
                continue
            try:
                ftp.rmd(parts[1])
                print(f"[OK] Directorio eliminado: {parts[1]}")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        elif command == "delete":
            if len(parts) != 2:
                print("[ERROR] Uso: delete <archivo>")
                continue
            try:
                ftp.delete(parts[1])
                print(f"[OK] Archivo eliminado: {parts[1]}")
            except ftplib.all_errors as e:
                print(f"[ERROR] {e}")

        else:
            print(f"[ERROR] Comando desconocido: {command}. Escribe 'help' para ayuda.")
