#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototipo de blockchain local muy sencillo:
  - Define una clase Block con:
      * índice, timestamp, datos, prev_hash, nonce, hash
      * método compute_hash() que calcula SHA-256 sobre los atributos.
  - Define una clase Blockchain que:
      * Inicializa con bloque génesis.
      * Permite agregar transacciones simples (strings).
      * Implementa proof-of-work con dificultad ajustable (número de ceros prefijo).
      * Verifica integridad del chain (hash correcto y prev_hash enlazado).
  - CLI básico para:
      * Añadir un nuevo bloque con datos personalizados.
      * Mostrar la cadena completa.
      * Verificar integridad.
"""
import time
import hashlib
import json
import sys
import argparse

class Block:
    """
    Representa un bloque en la cadena.
    """
    def __init__(self, index, timestamp, data, prev_hash, difficulty):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.prev_hash = prev_hash
        self.difficulty = difficulty  # número de ceros al inicio del hash
        self.nonce = 0
        self.hash = self.compute_proof_of_work()

    def compute_hash(self):
        """
        Calcula SHA-256 sobre los atributos index, timestamp, data, prev_hash y nonce.
        """
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'prev_hash': self.prev_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def compute_proof_of_work(self):
        """
        Incrementa nonce hasta que el hash resultante comience con '0' * difficulty.
        Retorna el hash válido.
        """
        prefix = '0' * self.difficulty
        self.nonce = 0
        computed_hash = self.compute_hash()
        while not computed_hash.startswith(prefix):
            self.nonce += 1
            computed_hash = self.compute_hash()
        return computed_hash

class Blockchain:
    """
    Maneja la cadena de bloques.
    """
    def __init__(self, difficulty=2):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Crea el bloque génesis con índice 0 y prev_hash '0'.
        """
        genesis_block = Block(0, time.time(), "Génesis", "0", self.difficulty)
        self.chain.append(genesis_block)

    @property
    def last_block(self):
        """
        Retorna el último bloque de la cadena.
        """
        return self.chain[-1]

    def add_block(self, data):
        """
        Crea un nuevo bloque con los datos proporcionados y lo añade a la cadena.
        """
        index = self.last_block.index + 1
        timestamp = time.time()
        prev_hash = self.last_block.hash
        new_block = Block(index, timestamp, data, prev_hash, self.difficulty)
        self.chain.append(new_block)

    def is_valid(self):
        """
        Verifica la integridad de la cadena:
          - Cada bloque debe tener prev_hash igual al hash del bloque anterior.
          - Cada hash debe satisfacer dificultad y coincidir con los datos.
        Retorna True si es válida, False en caso contrario.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]
            # Verificar enlace de hashes
            if current.prev_hash != prev.hash:
                print(f"[ERROR] Bloque {i}: prev_hash no coincide.")
                return False
            # Verificar hash recálculado
            recalculated = current.compute_hash()
            if current.hash != recalculated:
                print(f"[ERROR] Bloque {i}: hash almacenado no coincide con recálculo.")
                return False
            # Verificar dificultad
            if not current.hash.startswith('0' * current.difficulty):
                print(f"[ERROR] Bloque {i}: no satisface la dificultad.")
                return False
        return True

    def display_chain(self):
        """
        Muestra cada bloque en formato legible.
        """
        for block in self.chain:
            t_struct = time.localtime(block.timestamp)
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", t_struct)
            print(f"------ Bloque {block.index} ------")
            print(f"Timestamp  : {t_str}")
            print(f"Datos      : {block.data}")
            print(f"Prev Hash  : {block.prev_hash}")
            print(f"Nonce      : {block.nonce}")
            print(f"Hash       : {block.hash}")
            print(f"Dificultad : {block.difficulty}\n")
