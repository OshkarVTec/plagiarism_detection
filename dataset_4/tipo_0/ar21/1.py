#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encriptador/Desencriptador de archivos usando AES (CBC) con PyCryptodome.
Opciones:
  --encrypt <file>  : encripta archivo y genera '<file>.enc'
  --decrypt <file>  : desencripta archivo '.enc' y genera '<orig>_dec'
Clave:
  - Se solicita la contraseña al usuario, se genera clave de 256 bits
    mediante derivación PBKDF2 con sal aleatoria.
Formato de archivo encriptado:
  [16 bytes salt][16 bytes IV][cuerpo cifrado]
"""

import os
import sys
import getpass
import struct
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

# Parámetros KDF
KDF_SALT_BYTES = 16
KDF_ITERATIONS = 100000
KEY_LENGTH = 32  # 256 bits
AES_BLOCK_SIZE = 16
BUFFER_SIZE = 64 * 1024  # 64KB para leer de a pedazos

def derive_key(password, salt):
    """
    Deriva una clave de KEY_LENGTH bytes usando PBKDF2 con SHA256.
    """
    return PBKDF2(password, salt, dkLen=KEY_LENGTH, count=KDF_ITERATIONS)

def encrypt_file(input_file, password):
    """
    Encripta 'input_file' usando AES-CBC:
    1. Genera salt y deriva clave.
    2. Genera IV aleatorio de 16 bytes.
    3. Escribe [salt][IV] + contenido cifrado en 'input_file.enc'.
    Usa padding PKCS#7.
    """
    if not os.path.isfile(input_file):
        print(f"[ERROR] No existe el archivo: {input_file}")
        sys.exit(1)

    salt = get_random_bytes(KDF_SALT_BYTES)
    key = derive_key(password.encode("utf-8"), salt)
    iv = get_random_bytes(AES_BLOCK_SIZE)
    cipher = AES.new(key, AES.MODE_CBC, iv)

    output_file = input_file + ".enc"
    try:
        with open(input_file, "rb") as fin, open(output_file, "wb") as fout:
            # Escribir salt e IV al inicio
            fout.write(salt)
            fout.write(iv)
            # Leer por bloques y cifrar
            while True:
                chunk = fin.read(BUFFER_SIZE)
                if len(chunk) == 0:
                    break
                elif len(chunk) % AES_BLOCK_SIZE != 0:
                    # Padding PKCS#7
                    padding_len = AES_BLOCK_SIZE - (len(chunk) % AES_BLOCK_SIZE)
                    chunk += bytes([padding_len]) * padding_len
                    fout.write(cipher.encrypt(chunk))
                    break
                fout.write(cipher.encrypt(chunk))
        print(f"[OK] Archivo encriptado: {output_file}")
    except Exception as e:
        print(f"[ERROR] Falló encriptación: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        sys.exit(1)

def decrypt_file(input_file, password):
    """
    Desencripta 'input_file' asumiendo formato [salt][IV][cifrado].
    1. Lee salt e IV.
    2. Deriva clave y crea cipher AES-CBC.
    3. Descifra contenido y remueve padding.
    4. Guarda en '<orig>_dec' (sin extensión .enc).
    """
    if not os.path.isfile(input_file):
        print(f"[ERROR] No existe el archivo: {input_file}")
        sys.exit(1)

    try:
        with open(input_file, "rb") as fin:
            salt = fin.read(KDF_SALT_BYTES)
            iv = fin.read(AES_BLOCK_SIZE)
            key = derive_key(password.encode("utf-8"), salt)
            cipher = AES.new(key, AES.MODE_CBC, iv)

            # Preparar archivo de salida
            if input_file.endswith(".enc"):
                output_file = input_file[:-4] + "_dec"
            else:
                output_file = input_file + "_dec"

            with open(output_file, "wb") as fout:
                next_chunk = b''
                while True:
                    chunk = fin.read(BUFFER_SIZE)
                    if len(chunk) == 0:
                        # Último bloque: descifrar y quitar padding
                        decrypted = cipher.decrypt(next_chunk)
                        padding_len = decrypted[-1]
                        fout.write(decrypted[:-padding_len])
                        break
                    # Desencriptar bloque anterior (si existe)
                    if next_chunk:
                        fout.write(cipher.decrypt(next_chunk))
                    next_chunk = chunk
        print(f"[OK] Archivo desencriptado: {output_file}")
    except Exception as e:
        print(f"[ERROR] Falló desencriptación: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        sys.exit(1)

def print_usage():
    """
    Muestra ayuda de uso para el script.
    """
    help_text = """
Uso: python aes_file_crypto.py [--encrypt <archivo>] [--decrypt <archivo>]
Ejemplos:
  python aes_file_crypto.py --encrypt documento.txt
  python aes_file_crypto.py --decrypt documento.txt.enc
"""
    print(help_text)
