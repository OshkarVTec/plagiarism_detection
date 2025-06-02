#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de emociones en texto (positivo/negativo) usando Naive Bayes.
Entrena un modelo simple con ejemplos de frases predefinidas, luego permite
al usuario ingresar texto para predecir si la emoción es positiva o negativa.
"""

import sys
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Datos de entrenamiento predefinidos (pequeño conjunto para demostración)
TRAIN_DATA = [
    ("Estoy feliz y contento", "positivo"),
    ("Me siento muy triste hoy", "negativo"),
    ("¡Qué alegría ver este día!", "positivo"),
    ("Odio cuando las cosas salen mal", "negativo"),
    ("Estoy tranquilo y relajado", "positivo"),
    ("Me molesta el ruido fuerte", "negativo"),
    ("¡Es un día maravilloso!", "positivo"),
    ("No soporto la presión", "negativo"),
    ("Me encanta pasar tiempo con amigos", "positivo"),
    ("Esto me pone de mal humor", "negativo")
]

MODEL_FILE = "emotion_model.json"

def train_model():
    """
    Entrena un clasificador Naive Bayes usando CountVectorizer sobre TRAIN_DATA.
    Retorna el pipeline entrenado.
    """
    texts = [t for t, _ in TRAIN_DATA]
    labels = [l for _, l in TRAIN_DATA]

    pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("clf", MultinomialNB())
    ])
    pipeline.fit(texts, labels)
    print("[INFO] Modelo entrenado con datos de muestra.")
    return pipeline

def save_model(pipeline, file_path):
    """
    Guarda vocabulario del vectorizador y parámetros del clasificador en JSON.
    Nota: Para simplicidad, solo guarda _feature_count_ y vocabulario.
    """
    vect = pipeline.named_steps["vect"]
    clf = pipeline.named_steps["clf"]

    model_data = {
        "vocabulary": vect.vocabulary_,
        "class_count": clf.class_count_.tolist(),
        "feature_count": clf.feature_count_.tolist(),
        "classes": clf.classes_.tolist()
    }
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Modelo guardado en '{file_path}'.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el modelo: {e}")

def load_model(file_path):
    """
    Carga parámetros guardados en JSON y reconstruye el pipeline manualmente.
    Retorna el pipeline listo para predecir.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] No existe el archivo de modelo: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el modelo: {e}")
        return None

    # Reconstruir CountVectorizer
    vect = CountVectorizer(vocabulary=model_data["vocabulary"])
    # Reconstruir MultinomialNB
    clf = MultinomialNB()
    clf.class_count_ = model_data["class_count"]
    clf.feature_count_ = model_data["feature_count"]
    clf.classes_ = model_data["classes"]
    # Refit no es necesario; se setean internamente los parámetros

    pipeline = Pipeline([
        ("vect", vect),
        ("clf", clf)
    ])
    print(f"[INFO] Modelo cargado desde '{file_path}'.")
    return pipeline

def predict_emotion(pipeline, text):
    """
    Usa el pipeline para predecir la etiqueta (positivo/negativo) de 'text'.
    """
    return pipeline.predict([text])[0]
