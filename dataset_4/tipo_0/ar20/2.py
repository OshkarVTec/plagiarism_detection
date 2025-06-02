#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor WebSocket básico que:
  - Acepta conexiones entrantes.
  - Envía de vuelta (echo) cualquier mensaje recibido.
  - Mantiene múltiple clientes concurrentes.
"""

import asyncio
import websockets
import sys

# Puerto por defecto
DEFAULT_PORT = 8765

async def echo(websocket, path):
    """
    Maneja una conexión WebSocket:
    - Recibe mensajes del cliente y le envía la misma cadena de vuelta.
    - Finaliza cuando el cliente cierra la conexión.
    """
    client = websocket.remote_address
    print(f"[INFO] Cliente conectado: {client}")
    try:
        async for message in websocket:
            print(f"[{client}] → {message}")
            response = f"Echo: {message}"
            await websocket.send(response)
            print(f"[{client}] ← {response}")
    except websockets.ConnectionClosed:
        print(f"[INFO] Cliente desconectado: {client}")


"""
1. Lee puerto desde args o usa DEFAULT_PORT.
2. Inicia servidor WebSocket que atiende echo() en cada conexión.
3. Corre bucle de eventos hasta Ctrl+C.
"""
port = DEFAULT_PORT
if len(sys.argv) == 2:
    if not sys.argv[1].isdigit():
        print("[ERROR] El puerto debe ser numérico.")
        sys.exit(1)
    port = int(sys.argv[1])
elif len(sys.argv) > 2:
    print(f"Uso: python websocket_server.py [puerto]")
    sys.exit(1)

start_server = websockets.serve(echo, "0.0.0.0", port)
print(f"[INFO] Servidor WebSocket iniciado en ws://0.0.0.0:{port}")

loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)
try:
    loop.run_forever()
except KeyboardInterrupt:
    print("\n[INFO] Servidor detenido por usuario.")



"""
1. Lee puerto desde args o usa DEFAULT_PORT.
2. Inicia servidor WebSocket que atiende echo() en cada conexión.
3. Corre bucle de eventos hasta Ctrl+C.
"""
port = DEFAULT_PORT
if len(sys.argv) == 2:
    if not sys.argv[1].isdigit():
        print("[ERROR] El puerto debe ser numérico.")
        sys.exit(1)
    port = int(sys.argv[1])
elif len(sys.argv) > 2:
    print(f"Uso: python websocket_server.py [puerto]")
    sys.exit(1)

start_server = websockets.serve(echo, "0.0.0.0", port)
print(f"[INFO] Servidor WebSocket iniciado en ws://0.0.0.0:{port}")

loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)
try:
    loop.run_forever()
except KeyboardInterrupt:
    print("\n[INFO] Servidor detenido por usuario.")

