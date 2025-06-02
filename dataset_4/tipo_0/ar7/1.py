#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculadora básica con interfaz gráfica utilizando Tkinter.
Permite operaciones simples: suma, resta, multiplicación y división.
"""

import tkinter as tk
from tkinter import messagebox

# Configuración de la ventana principal
root = tk.Tk()
root.title("Calculadora Tkinter")
root.geometry("300x400")
root.resizable(False, False)

# Variable para mostrar la expresión y el resultado
expression = ""
input_text = tk.StringVar()

def btn_click(item):
    """Agrega el carácter presionado a la expresión."""
    global expression
    expression += str(item)
    input_text.set(expression)

def btn_clear():
    """Limpia la expresión y el campo de texto."""
    global expression
    expression = ""
    input_text.set("")

def btn_equal():
    """Evalúa la expresión y muestra el resultado."""
    global expression
    try:
        result = str(eval(expression))
        input_text.set(result)
        expression = result
    except Exception as e:
        messagebox.showerror("Error", f"Expresión inválida\n{e}")
        expression = ""
        input_text.set("")

# Marco para contener la pantalla de la calculadora
input_frame = tk.Frame(root, width=312, height=50, bd=0, highlightbackground="black", highlightcolor="black", highlightthickness=1)
input_frame.pack(side=tk.TOP)

# Pantalla de entrada
input_field = tk.Entry(input_frame, font=('arial', 18, 'bold'), textvariable=input_text, width=50, bg="#eee", bd=0, justify=tk.RIGHT)
input_field.grid(row=0, column=0)
input_field.pack(ipady=10)

# Marco para los botones
btns_frame = tk.Frame(root, width=312, height=272.5, bg="grey")
btns_frame.pack()

# Primera fila
clear = tk.Button(btns_frame, text="C", fg="black", width=32, height=3, bd=0, bg="#eee", cursor="hand2", command=lambda: btn_clear()).grid(row=0, column=0, columnspan=3, padx=1, pady=1)
divide = tk.Button(btns_frame, text="/", fg="black", width=10, height=3, bd=0, bg="#ffa500", cursor="hand2", command=lambda: btn_click("/")).grid(row=0, column=3, padx=1, pady=1)

# Segunda fila
seven = tk.Button(btns_frame, text="7", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("7")).grid(row=1, column=0, padx=1, pady=1)
eight = tk.Button(btns_frame, text="8", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("8")).grid(row=1, column=1, padx=1, pady=1)
nine = tk.Button(btns_frame, text="9", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("9")).grid(row=1, column=2, padx=1, pady=1)
multiply = tk.Button(btns_frame, text="*", fg="black", width=10, height=3, bd=0, bg="#ffa500", cursor="hand2", command=lambda: btn_click("*")).grid(row=1, column=3, padx=1, pady=1)

# Tercera fila
four = tk.Button(btns_frame, text="4", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("4")).grid(row=2, column=0, padx=1, pady=1)
five = tk.Button(btns_frame, text="5", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("5")).grid(row=2, column=1, padx=1, pady=1)
six = tk.Button(btns_frame, text="6", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("6")).grid(row=2, column=2, padx=1, pady=1)
subtract = tk.Button(btns_frame, text="-", fg="black", width=10, height=3, bd=0, bg="#ffa500", cursor="hand2", command=lambda: btn_click("-")).grid(row=2, column=3, padx=1, pady=1)

# Cuarta fila
one = tk.Button(btns_frame, text="1", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("1")).grid(row=3, column=0, padx=1, pady=1)
two = tk.Button(btns_frame, text="2", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("2")).grid(row=3, column=1, padx=1, pady=1)
three = tk.Button(btns_frame, text="3", fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("3")).grid(row=3, column=2, padx=1, pady=1)
add = tk.Button(btns_frame, text="+", fg="black", width=10, height=3, bd=0, bg="#ffa500", cursor="hand2", command=lambda: btn_click("+")).grid(row=3, column=3, padx=1, pady=1)

# Quinta fila
zero = tk.Button(btns_frame, text="0", fg="black", width=21, height=3, bd=0, bg="#fff", cursor="hand2", command=lambda: btn_click("0")).grid(row=4, column=0, columnspan=2, padx=1, pady=1)
point = tk.Button(btns_frame, text=".", fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2", command=lambda: btn_click(".")).grid(row=4, column=2, padx=1, pady=1)
equals = tk.Button(btns_frame, text="=", fg="black", width=10, height=3, bd=0, bg="#ffa500", cursor="hand2", command=lambda: btn_equal()).grid(row=4, column=3, padx=1, pady=1)

# Inicia el bucle principal de la interfaz
root.mainloop()
