#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliente HTTP que obtiene un token OAuth2 (grant type: client_credentials) y
consume un endpoint protegido que requiere Bearer token.
Permite:
  - Autenticar contra servidor OAuth2 (client_id y client_secret).
  - Obtener token de acceso.
  - Hacer solicitud GET a recurso protegido usando el token.
  - Mostrar respuesta JSON.
"""

import sys
import argparse
import requests

def get_oauth2_token(token_url, client_id, client_secret, scope=None):
    """
    Solicita token OAuth2 usando grant type client_credentials:
      - token_url: endpoint para solicitar token.
      - client_id y client_secret: credenciales del cliente.
      - scope: ámbito (opcional).
    Retorna el access_token.
    """
    data = {'grant_type': 'client_credentials'}
    if scope:
        data['scope'] = scope
    try:
        response = requests.post(
            token_url,
            data=data,
            auth=(client_id, client_secret),
            timeout=10
        )
        response.raise_for_status()
        token_json = response.json()
        access_token = token_json.get('access_token')
        if not access_token:
            print("[ERROR] No se obtuvo access_token en la respuesta.")
            sys.exit(1)
        return access_token
    except requests.RequestException as e:
        print(f"[ERROR] Falló al obtener token OAuth2: {e}")
        sys.exit(1)

def call_protected_api(api_url, token):
    """
    Realiza una solicitud GET a `api_url` usando el token Bearer.
    Retorna la respuesta JSON.
    """
    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        print(f"[ERROR] HTTP error: {http_err} - {response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Falló al llamar al API protegida: {e}")
        sys.exit(1)

