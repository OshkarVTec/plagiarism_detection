#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconocimiento de voz desde micrófono usando la librería SpeechRecognition.
Convierte voz a texto en tiempo real (hasta que se diga 'salir').
"""

import speech_recognition as sr
import sys


"""
1. Inicializa el Recognizer y el micrófono.
2. En un bucle, escucha audio y transcribe mediante el recognizer de Google.
3. Imprime en consola el texto reconocido.
4. Si el usuario dice 'salir', termina el programa.
"""
recognizer = sr.Recognizer()

try:
    mic = sr.Microphone()
except Exception as e:
    print(f"[ERROR] No se pudo acceder al micrófono: {e}")
    sys.exit(1)

print("=== Reconocimiento de Voz (di 'salir' para terminar) ===")
with mic as source:
    # Ajustar nivel de ruido ambiental
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("[INFO] Calibración completa, comenzando escucha...")

while True:
    with mic as source:
        print("Escuchando...")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        # Usar servicio de reconocimiento de Google (requiere conexión)
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tú (voz): {text}")
        if text.strip().lower() == "salir":
            print("[INFO] Palabra 'salir' detectada. Terminando reconocimiento de voz.")
            break
    except sr.UnknownValueError:
        print("[WARN] No se entendió lo dicho.")
    except sr.RequestError as e:
        print(f"[ERROR] Error en el servicio de reconocimiento: {e}")
    except Exception as e:
        print(f"[ERROR] Excepción imprevista: {e}")

while True:
    with mic as source:
        print("Escuchando...")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        # Usar servicio de reconocimiento de Google (requiere conexión)
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tú (voz): {text}")
        if text.strip().lower() == "salir":
            print("[INFO] Palabra 'salir' detectada. Terminando reconocimiento de voz.")
            break
    except sr.UnknownValueError:
        print("[WARN] No se entendió lo dicho.")
    except sr.RequestError as e:
        print(f"[ERROR] Error en el servicio de reconocimiento: {e}")
    except Exception as e:
        print(f"[ERROR] Excepción imprevista: {e}")

while True:
    with mic as source:
        print("Escuchando...")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        # Usar servicio de reconocimiento de Google (requiere conexión)
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tú (voz): {text}")
        if text.strip().lower() == "salir":
            print("[INFO] Palabra 'salir' detectada. Terminando reconocimiento de voz.")
            break
    except sr.UnknownValueError:
        print("[WARN] No se entendió lo dicho.")
    except sr.RequestError as e:
        print(f"[ERROR] Error en el servicio de reconocimiento: {e}")
    except Exception as e:
        print(f"[ERROR] Excepción imprevista: {e}")

