#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream de Twitter en tiempo real para análisis de sentimiento:
  - Usa Tweepy para conectarse a la API de streaming de Twitter (requiere credenciales).
  - Filtra tweets por palabras clave o hashtag.
  - Para cada tweet recibido, usa TextBlob para clasificar su sentimiento: positivo, neutro o negativo.
  - Imprime en consola el texto y su polaridad.
  - Opcionalmente guarda resultados en un archivo CSV.
"""

import sys
import os
import argparse
import csv
from textblob import TextBlob
import tweepy

# Clase para manejar los eventos del stream
class SentimentStreamListener(tweepy.StreamListener):
    """
    Escucha tweets en tiempo real y analiza sentimiento con TextBlob.
    """

    def __init__(self, csv_writer=None):
        super().__init__()
        self.csv_writer = csv_writer

    def on_status(self, status):
        """
        Se ejecuta cuando llega un nuevo tweet:
          - Extrae texto y crea objeto TextBlob.
          - Determina sentimiento según polarity.
          - Imprime en consola y escribe en CSV si se configuró.
        """
        # Omitir retweets
        if hasattr(status, 'retweeted_status'):
            return

        text = status.text
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment = "Positivo"
        elif polarity == 0:
            sentiment = "Neutro"
        else:
            sentiment = "Negativo"

        print(f"Tweet: {text}")
        print(f"Sentimiento: {sentiment} (Polaridad: {polarity:.2f})")
        print("-" * 50)

        if self.csv_writer:
            self.csv_writer.writerow([text, sentiment, polarity])

    def on_error(self, status_code):
        """
        Maneja errores del stream. Si el código es 420 (rate limit), detiene el stream.
        """
        print(f"[ERROR] Estado del stream: {status_code}")
        if status_code == 420:
            # Desconectar para evitar penalización
            return False

