#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación simple de una blockchain con proof-of-work.
Incluye creación de bloques, hashing y minería.
"""

import hashlib
import json
from time import time


class Block:
    """
    Representa un bloque en la cadena.
    Cada bloque almacena:
    - index: posición en la cadena.
    - timestamp: marca de tiempo de creación.
    - data: información que almacena (por ejemplo, transacciones).
    - previous_hash: hash del bloque anterior.
    - nonce: número usado para la prueba de trabajo.
    - hash: hash del bloque actual.
    """
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.timestamp = time()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        Calcula el hash SHA-256 del bloque a partir de su contenido.
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()

        return hashlib.sha256(block_string).hexdigest()

    def mine(self, difficulty):
        """
        Realiza la prueba de trabajo incrementando nonce hasta encontrar
        un hash que comience con una cantidad determinada de ceros.
        """
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()


class Blockchain:
    """
    Representa la cadena de bloques.
    - difficulty: nivel de dificultad de la prueba de trabajo.
    - chain: lista de bloques.
    """
    def __init__(self, difficulty=4):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Crea el bloque génesis (primer bloque) con valores por defecto.
        """
        genesis_block = Block(index=0, data="Bloque Génesis", previous_hash="0")
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    def latest_block(self):
        """Devuelve el último bloque de la cadena."""
        return self.chain[-1]

    def add_block(self, data):
        """Crea y mina un nuevo bloque con los datos proporcionados."""
        previous_hash = self.latest_block().hash
        new_block = Block(index=len(self.chain), data=data, previous_hash=previous_hash)
        print(f"Minería del bloque {new_block.index}...")
        new_block.mine(self.difficulty)
        self.chain.append(new_block)
        print(f"Bloque {new_block.index} añadido con hash: {new_block.hash}")

    def is_chain_valid(self):
        """
        Verifica la integridad de la cadena:
        - Recalcula el hash de cada bloque y lo compara.
        - Verifica que el previous_hash concuerde con el hash anterior.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]
            if current.hash != current.compute_hash():
                print(f"Hash inválido en bloque {i}")
                return False
            if current.previous_hash != prev.hash:
                print(f"Previous hash inválido en bloque {i}")
                return False
        return True
