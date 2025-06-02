#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Editor básico de imágenes con interfaz gráfica (Tkinter) y Pillow.
Permite:
  - Cargar una imagen desde disco.
  - Aplicar transformaciones: rotar, voltear, escala de grises, recortar.
  - Ajustar brillo y contraste.
  - Guardar la imagen modificada.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance

class ImageEditorApp:
    """
    Clase principal que define la ventana y las funcionalidades del editor.
    Contiene:
      - Área para mostrar la imagen.
      - Botones para cargar, transformar y guardar.
      - Barras para ajustar brillo y contraste.
    """

    def __init__(self, master):
        self.master = master
        self.master.title("Editor de Imágenes - Tkinter + Pillow")
        self.master.geometry("800x600")
        self.master.configure(bg="#eee")

        # Atributos para la imagen original y la modificada
        self.original_image = None
        self.edited_image = None
        self.tk_image = None

        # Crear widgets
        self.create_widgets()

    def create_widgets(self):
        """
        Crea todos los widgets (botones, sliders, canvas) y los ubica en la ventana.
        """
        # Botón para cargar imagen
        self.btn_load = tk.Button(self.master, text="Cargar Imagen", command=self.load_image)
        self.btn_load.pack(pady=10)

        # Canvas para mostrar imagen
        self.canvas = tk.Canvas(self.master, width=700, height=400, bg="#ccc")
        self.canvas.pack(pady=10)

        # Frame para botones de transformaciones
        self.frame_ops = tk.Frame(self.master, bg="#eee")
        self.frame_ops.pack(pady=10)

        # Botones de operaciones
        self.btn_grayscale = tk.Button(self.frame_ops, text="Escala de Grises", command=self.apply_grayscale)
        self.btn_rotate = tk.Button(self.frame_ops, text="Rotar 90°", command=self.rotate_90)
        self.btn_flip_h = tk.Button(self.frame_ops, text="Voltear Horizontal", command=self.flip_horizontal)
        self.btn_flip_v = tk.Button(self.frame_ops, text="Voltear Vertical", command=self.flip_vertical)
        self.btn_crop = tk.Button(self.frame_ops, text="Recortar (centro)", command=self.crop_center)

        self.btn_grayscale.grid(row=0, column=0, padx=5)
        self.btn_rotate.grid(row=0, column=1, padx=5)
        self.btn_flip_h.grid(row=0, column=2, padx=5)
        self.btn_flip_v.grid(row=0, column=3, padx=5)
        self.btn_crop.grid(row=0, column=4, padx=5)

        # Sliders para brillo y contraste
        self.frame_adjust = tk.Frame(self.master, bg="#eee")
        self.frame_adjust.pack(pady=10)

        tk.Label(self.frame_adjust, text="Brillo").grid(row=0, column=0, padx=5)
        self.slider_brightness = tk.Scale(self.frame_adjust, from_=0.2, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, length=200)
        self.slider_brightness.set(1.0)
        self.slider_brightness.grid(row=0, column=1, padx=5)

        tk.Label(self.frame_adjust, text="Contraste").grid(row=0, column=2, padx=5)
        self.slider_contrast = tk.Scale(self.frame_adjust, from_=0.2, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, length=200)
        self.slider_contrast.set(1.0)
        self.slider_contrast.grid(row=0, column=3, padx=5)

        # Botón para aplicar ajustes de brillo/contraste
        self.btn_apply_adj = tk.Button(self.frame_adjust, text="Aplicar Ajustes", command=self.apply_brightness_contrast)
        self.btn_apply_adj.grid(row=0, column=4, padx=10)

        # Botón para guardar imagen
        self.btn_save = tk.Button(self.master, text="Guardar Imagen", command=self.save_image)
        self.btn_save.pack(pady=10)

    def load_image(self):
        """
        Abre diálogo para seleccionar un archivo de imagen.
        Carga con Pillow y muestra en el canvas.
        """
        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen: {e}")
            return

        self.original_image = img.copy()
        self.edited_image = img.copy()
        self.display_image(self.edited_image)

    def display_image(self, pil_image):
        """
        Escala la imagen PIL a un tamaño adecuado para el canvas y la muestra.
        """
        # Obtener tamaño del canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        # Escalar manteniendo proporción
        img_w, img_h = pil_image.size
        ratio = min(canvas_w/img_w, canvas_h/img_h)
        new_size = (int(img_w*ratio), int(img_h*ratio))
        resized = pil_image.resize(new_size, Image.ANTIALIAS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w/2, canvas_h/2, image=self.tk_image)

    def apply_grayscale(self):
        """
        Convierte la imagen actual a escala de grises.
        """
        if self.edited_image:
            self.edited_image = self.edited_image.convert("L").convert("RGB")
            self.display_image(self.edited_image)

    def rotate_90(self):
        """
        Rota la imagen 90 grados en sentido horario.
        """
        if self.edited_image:
            self.edited_image = self.edited_image.rotate(-90, expand=True)
            self.display_image(self.edited_image)

    def flip_horizontal(self):
        """
        Voltea la imagen horizontalmente.
        """
        if self.edited_image:
            self.edited_image = self.edited_image.transpose(Image.FLIP_LEFT_RIGHT)
            self.display_image(self.edited_image)

    def flip_vertical(self):
        """
        Voltea la imagen verticalmente.
        """
        if self.edited_image:
            self.edited_image = self.edited_image.transpose(Image.FLIP_TOP_BOTTOM)
            self.display_image(self.edited_image)

    def crop_center(self):
        """
        Recorta un rectángulo central (50% ancho y alto) de la imagen.
        """
        if self.edited_image:
            w, h = self.edited_image.size
            left = w * 0.25
            top = h * 0.25
            right = w * 0.75
            bottom = h * 0.75
            self.edited_image = self.edited_image.crop((left, top, right, bottom))
            self.display_image(self.edited_image)

    def apply_brightness_contrast(self):
        """
        Ajusta brillo y contraste según valores de los sliders.
        Reinicia desde la imagen original para evitar acumulación de cambios.
        """
        if self.original_image:
            brightness = self.slider_brightness.get()
            contrast = self.slider_contrast.get()
            img = self.original_image.copy()

            # Ajustar brillo
            enhancer_b = ImageEnhance.Brightness(img)
            img = enhancer_b.enhance(brightness)
            # Ajustar contraste
            enhancer_c = ImageEnhance.Contrast(img)
            img = enhancer_c.enhance(contrast)

            self.edited_image = img
            self.display_image(self.edited_image)

    def save_image(self):
        """
        Abre diálogo para guardar la imagen actualmente editada.
        """
        if not self.edited_image:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("GIF", "*.gif")],
            title="Guardar imagen como"
        )
        if not file_path:
            return

        try:
            self.edited_image.save(file_path)
            messagebox.showinfo("Éxito", f"Imagen guardada en: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la imagen: {e}")
