#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot básico basado en reglas y procesamiento simple de lenguaje natural con NLTK.
El bot responde según palabras clave y patrones predefinidos.
"""

import nltk
import random
import string
import sys

# Asegurarse de tener los recursos necesarios de NLTK
# nltk.download('punkt')  # para tokenización
# nltk.download('wordnet')  # para lematización

from nltk.stem import WordNetLemmatizer

# Diccionario de respuestas predefinidas según temas
RESPUESTAS = {
    "saludos": ["¡Hola!", "¡Buen día!", "¡Hola! ¿En qué puedo ayudarte?"],
    "despedidas": ["¡Adiós!", "¡Hasta luego!", "¡Fue un gusto ayudarte!"],
    "agradecimientos": ["¡De nada!", "¡Con gusto!", "¡Un placer!"],
    "desconocido": ["Lo siento, no entendí eso.", "¿Podrías reformular tu pregunta?", "No estoy seguro de cómo responder a eso."]
}

# Palabras clave asociadas a cada tema
KEYWORDS = {
    "saludos": ["hola", "buenos", "buenas", "saludos", "hey"],
    "despedidas": ["adiós", "hasta", "chao", "nos vemos"],
    "agradecimientos": ["gracias", "muchas", "agradezco"]
}

def normalize_text(text):
    """
    Normaliza el texto de entrada:
    - Convierte a minúsculas.
    - Tokeniza en palabras.
    - Lematiza cada palabra.
    - Elimina signos de puntuación.
    Retorna lista de tokens lematizados.
    """
    lemmatizer = WordNetLemmatizer()
    # Quitar puntuación y convertir a minúsculas
    texto = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(texto)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

def detect_intent(tokens):
    """
    Detecta la intención del usuario basándose en la presencia de palabras clave.
    Retorna la etiqueta de intención correspondiente o 'desconocido'.
    """
    for intent, palabras in KEYWORDS.items():
        for palabra in palabras:
            if palabra in tokens:
                return intent
    return "desconocido"

def get_response(intent):
    """
    Devuelve una respuesta aleatoria basada en el diccionario RESPUESTAS.
    """
    respuestas = RESPUESTAS.get(intent, RESPUESTAS["desconocido"])
    return random.choice(respuestas)

def chatbot():
    """
    Bucle principal del chatbot:
    - Lee input del usuario.
    - Normaliza el texto y detecta intención.
    - Muestra respuesta correspondiente.
    - Termina si detecta una despedida.
    """
    print("=== Chatbot Sencillo (escribe 'salir' para terminar) ===")
    while True:
        try:
            user_input = input("Tú: ").strip()
        except EOFError:
            print("\n[INFO] Finalizando chatbot.")
            break

        if not user_input:
            continue

        # Si el usuario escribe 'salir', termina el chat
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Chatbot: ¡Hasta luego!")
            break

        tokens = normalize_text(user_input)
        intent = detect_intent(tokens)
        response = get_response(intent)

        print(f"Chatbot: {response}")
