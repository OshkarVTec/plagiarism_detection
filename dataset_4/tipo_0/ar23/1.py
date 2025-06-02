#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de automatización web usando Selenium:
  - Abre navegador Chrome en modo automatizado.
  - Navega a un sitio de ejemplo que requiera login.
  - Rellena formulario de usuario y contraseña.
  - Hace clic en el botón de inicio de sesión.
  - Después de iniciar sesión, navega a una página protegida.
  - Extrae datos (por ejemplo, títulos de artículos) y los imprime.
  - Cierra el navegador al final.
Configura:
  - Se asume que ChromeDriver está instalado y en PATH.
  - Usa argumentos para usuario, contraseña y URL base.
"""

import sys
import argparse
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def configure_driver(headless=True):
    """
    Configura Selenium WebDriver para Chrome.
    Si headless=True, corre en modo sin ventana gráfica.
    Retorna la instancia del driver.
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # Para evitar detección de Selenium
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def perform_login(driver, login_url, username, password):
    """
    Navega a login_url, rellena usuario y contraseña, y hace clic en iniciar sesión.
    Espera a que la página siguiente cargue para confirmar éxito o lanza excepción.
    """
    driver.get(login_url)
    try:
        # Esperar a que el campo de usuario esté presente
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        # Encontrar elementos de formulario
        user_field = driver.find_element(By.NAME, "username")
        pass_field = driver.find_element(By.NAME, "password")
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")

        # Rellenar credenciales
        user_field.send_keys(username)
        pass_field.send_keys(password)
        login_button.click()

        # Esperar a que se redirija, por ejemplo, a /dashboard
        WebDriverWait(driver, 10).until(
            EC.url_contains("/dashboard")
        )
        print("[INFO] Login exitoso. Redirigido a dashboard.")
    except (NoSuchElementException, TimeoutException) as e:
        print(f"[ERROR] Falló el login o no se encontró un elemento: {e}")
        driver.quit()
        sys.exit(1)

def extract_data(driver, protected_url):
    """
    Navega a protected_url y extrae, por ejemplo, títulos de artículos:
    - Espera a que estén presentes elementos <h2 class="article-title">.
    - Imprime texto de cada título extraído.
    """
    driver.get(protected_url)
    try:
        # Esperar a que los títulos estén presentes
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "article-title"))
        )
        titles = driver.find_elements(By.CLASS_NAME, "article-title")
        print("[INFO] Títulos de artículos encontrados:")
        for idx, title_elem in enumerate(titles, start=1):
            print(f"  {idx}. {title_elem.text}")
    except (NoSuchElementException, TimeoutException):
        print("[WARN] No se encontraron títulos de artículos.")

