#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API REST con FastAPI para inferencia de un modelo de clasificación entrenado.
Permite:
  - Cargar un modelo entrenado (p. ej. pickle de scikit-learn).
  - Recibir peticiones POST con datos en JSON para clasificación.
  - Devolver la predicción y probabilidad.
  - Endpoint de salud (health check).
Ejecutar con: uvicorn ml_api:app --reload
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Ruta al modelo serializado (pickle)
MODEL_FILE = "model_clf.pkl"

# Definir esquema de datos de entrada (ajustar según atributos del modelo)
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Definir esquema de respuesta
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

app = FastAPI(title="API de Inferencia de Modelo", version="1.0")

# Cargar el modelo al iniciar la app
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print("[INFO] Modelo cargado exitosamente.")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo del modelo: {MODEL_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    sys.exit(1)

@app.get("/health")
def health_check():
    """
    Endpoint de salud. Retorna {'status': 'OK'} para indicar que el servicio está activo.
    """
    return {"status": "OK"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    """
    Recibe un JSON con los valores de las características.
    Realiza predicción y devuelve la etiqueta y probabilidad.
    """
    # Convertir datos a arreglo numpy en el orden esperado por el modelo
    try:
        features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
        probas = model.predict_proba(features)[0]
        pred = int(model.predict(features)[0])
        # Obtener probabilidad de la clase predicha
        probability = float(np.max(probas))
        return PredictionResponse(prediction=pred, probability=probability)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al realizar predicción: {e}")
