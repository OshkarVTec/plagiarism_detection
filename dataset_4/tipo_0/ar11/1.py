#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Envío masivo de correos electrónicos mediante SMTP.
Lee una lista de destinatarios desde un archivo, permite agregar adjuntos
y envía el mismo mensaje a todos los correos especificados.
"""

import smtplib
import os
import sys
import getpass
from email.message import EmailMessage
from email.utils import make_msgid
from mimetypes import guess_type

# Archivo que contiene la lista de destinatarios (una dirección por línea)
RECIPIENTS_FILE = "recipients.txt"
# Carpeta donde se guardan los adjuntos opcionales
ATTACHMENTS_DIR = "attachments"

def load_recipients(file_path):
    """
    Carga la lista de destinatarios desde un archivo de texto.
    Ignora líneas vacías o que empiecen con '#'.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] No se encontró el archivo: {file_path}")
        sys.exit(1)

    recipients = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                recipients.append(line)
    return recipients

def get_attachments(folder_path):
    """
    Retorna la lista de rutas de archivos en la carpeta de adjuntos.
    Si la carpeta no existe o está vacía, retorna lista vacía.
    """
    attachments = []
    if not os.path.isdir(folder_path):
        print(f"[INFO] No existe la carpeta de adjuntos: {folder_path}. Se omiten adjuntos.")
        return attachments

    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            attachments.append(full_path)
    return attachments

def create_email(sender, subject, body, attachments):
    """
    Crea un objeto EmailMessage configurado con remitente, asunto, cuerpo y adjuntos.
    Todos los correos usarán la misma plantilla.
    """
    msg = EmailMessage()
    msg["From"] = sender
    msg["Subject"] = subject
    # Para mensajería HTML básica, podemos inyectar un ID de imagen en línea
    msg.set_content(body)

    # Agregar identificadores para imágenes en línea (si se usaran)
    # ejemplo_cid = make_msgid(domain='example.com')

    for file_path in attachments:
        # Determinar tipo MIME según extensión
        mime_type, _ = guess_type(file_path)
        if mime_type is None:
            # Tipo genérico si no se detecta
            mime_type = "application/octet-stream"
        maintype, subtype = mime_type.split("/", 1)

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            filename = os.path.basename(file_path)
            msg.add_attachment(file_data,
                               maintype=maintype,
                               subtype=subtype,
                               filename=filename)
            print(f"[INFO] Adjuntado: {filename} ({mime_type})")
        except Exception as e:
            print(f"[WARN] No se pudo adjuntar {file_path}: {e}")

    return msg

def send_bulk_email(smtp_server, smtp_port, sender_email, password, recipients, subject, body, attachments):
    """
    Conecta al servidor SMTP y envía el mismo correo a todos los destinatarios en la lista.
    """
    try:
        print(f"[INFO] Conectando a SMTP {smtp_server}:{smtp_port}...")
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.ehlo()
        # Intentar iniciar TLS si el servidor lo soporta
        try:
            server.starttls()
            server.ehlo()
            print("[INFO] Conexión segura establecida con STARTTLS.")
        except Exception:
            print("[INFO] STARTTLS no soportado o fallo. Continuando sin TLS.")
        # Autenticación
        server.login(sender_email, password)
        print("[INFO] Autenticación exitosa.")

        base_msg = create_email(sender_email, subject, body, attachments)
        sent_count = 0
        for idx, recipient in enumerate(recipients, start=1):
            try:
                msg = base_msg
                msg["To"] = recipient
                server.send_message(msg)
                sent_count += 1
                print(f"[OK] Correo enviado a {recipient} ({idx}/{len(recipients)})")
                # Remover el campo 'To' para el siguiente ciclo
                del msg["To"]
            except Exception as e:
                print(f"[ERROR] No se pudo enviar a {recipient}: {e}")

        server.quit()
        print(f"[INFO] Envío completado. Total enviados: {sent_count}, Total destinatarios: {len(recipients)}")
    except Exception as e:
        print(f"[ERROR] Falló la conexión o autenticación SMTP: {e}")
        sys.exit(1)

def print_usage():
    """
    Muestra la ayuda de uso del script en consola.
    """
    help_text = f"""
Uso: python bulk_email.py <smtp_server> <smtp_port> <sender_email>
    <smtp_server>   Dirección del servidor SMTP (e.g., smtp.gmail.com)
    <smtp_port>     Puerto del servidor SMTP (e.g., 587)
    <sender_email>  Correo electrónico del remitente (debe corresponder con la cuenta)
El script solicitará la contraseña en tiempo de ejecución, luego leerá:
    - Lista de destinatarios desde '{RECIPIENTS_FILE}'
    - Mensaje y asunto ingresados por el usuario
    - Adjuntos desde la carpeta '{ATTACHMENTS_DIR}' (si existe)
"""
    print(help_text)
