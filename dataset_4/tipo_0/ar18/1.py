#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación de chat sencilla con GUI usando Tkinter.
Permite al usuario escribir mensajes y muestra respuestas generadas
por un pequeño bot basado en reglas (sin conexión a internet).
"""

import tkinter as tk
from tkinter import scrolledtext

# Palabras clave y respuestas del bot (muy básico)
KEYWORDS_RESPONSES = {
    "hola": "¡Hola! ¿Cómo estás?",
    "adiós": "¡Hasta luego! Que tengas un buen día.",
    "como": "Estoy bien, gracias. ¿Y tú?",
    "gracias": "¡De nada!",
    "tiempo": "No tengo información del clima, pero espero que sea bueno."
}

DEFAULT_RESPONSE = "Lo siento, no entiendo. ¿Podrías reformular?"

class ChatGUI:
    """
    Clase que define la interfaz de chat con Tkinter.
    Contiene un área de texto para mostrar conversación y un Entry para escribir.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("ChatBot GUI")
        self.master.geometry("500x600")
        self.master.resizable(False, False)

        # Frame para conversación
        self.conv_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.conv_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Área de texto desplazable para mostrar conversación
        self.text_area = scrolledtext.ScrolledText(
            self.conv_frame,
            wrap=tk.WORD,
            width=60,
            height=30,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.text_area.pack(padx=5, pady=5, fill="both", expand=True)

        # Frame para la entrada y botón
        self.entry_frame = tk.Frame(self.master)
        self.entry_frame.pack(padx=10, pady=(0,10), fill="x")

        # Campo de entrada de usuario
        self.entry_field = tk.Entry(self.entry_frame, font=("Arial", 12))
        self.entry_field.pack(side=tk.LEFT, padx=(0,5), fill="x", expand=True)
        self.entry_field.bind("<Return>", self.send_message)

        # Botón de enviar
        self.send_button = tk.Button(
            self.entry_frame,
            text="Enviar",
            command=self.send_message,
            width=10
        )
        self.send_button.pack(side=tk.RIGHT)

        # Mensaje de bienvenida
        self.display_message("Bot", "¡Hola! Soy un bot. Escribe algo y presiona Enter.")

    def display_message(self, sender, message):
        """
        Muestra un mensaje en el área de texto con formato:
        [sender]: message
        """
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"{sender}: {message}\n")
        self.text_area.configure(state=tk.DISABLED)
        # Auto-scroll al final
        self.text_area.yview(tk.END)

    def generate_bot_response(self, user_text):
        """
        Genera una respuesta basada en palabras clave del usuario.
        Si detecta alguna palabra clave, devuelve la respuesta asociada.
        Si no, devuelve DEFAULT_RESPONSE.
        """
        user_lower = user_text.lower()
        for keyword, response in KEYWORDS_RESPONSES.items():
            if keyword in user_lower:
                return response
        return DEFAULT_RESPONSE

    def send_message(self, event=None):
        """
        Método que se llama al presionar Enter o botón Enviar.
        Obtiene texto del entry, lo muestra como mensaje del usuario,
        luego genera respuesta del bot y la muestra.
        """
        user_text = self.entry_field.get().strip()
        if not user_text:
            return
        self.display_message("Tú", user_text)
        self.entry_field.delete(0, tk.END)

        # Generar respuesta del bot
        response = self.generate_bot_response(user_text)
        self.display_message("Bot", response)
