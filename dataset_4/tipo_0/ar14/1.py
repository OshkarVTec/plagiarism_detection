#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador léxico y sintáctico (parser) para evaluar expresiones aritméticas
básicas que incluyen +, -, *, /, paréntesis y números enteros.
Implementa:
  - Lexer: convierte input en tokens (ENTERO, MÁS, MENOS, etc.).
  - Parser recursivo descendente para la gramática:
    expr   -> term ((+|-) term)*
    term   -> factor ((*|/) factor)*
    factor -> ENTERO | '(' expr ')'
"""

import sys
import re

# Definición de tipos de tokens
TOKENS = {
    'ENTERO': r'\d+',
    'MAS': r'\+',
    'MENOS': r'-',
    'MULT': r'\*',
    'DIV': r'/',
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'ESPACIO': r'\s+'
}

# Compilar expresiones regulares de tokens
token_regex = [(name, re.compile(pattern)) for name, pattern in TOKENS.items()]

class Token:
    """
    Representa un token con tipo y valor (texto original).
    """
    def __init__(self, tipo, valor):
        self.tipo = tipo
        self.valor = valor

    def __repr__(self):
        return f"Token({self.tipo}, '{self.valor}')"

def lexer(texto):
    """
    Convierte la cadena de entrada en una lista de tokens.
    Omite espacios y reporta error si encuentra símbolo no reconocido.
    """
    tokens = []
    i = 0
    while i < len(texto):
        match = None
        for tipo, regex in token_regex:
            match = regex.match(texto, i)
            if match:
                lexeme = match.group(0)
                if tipo != 'ESPACIO':
                    tokens.append(Token(tipo, lexeme))
                i = match.end(0)
                break
        if not match:
            print(f"[ERROR] Carácter no reconocido: '{texto[i]}'")
            sys.exit(1)
    tokens.append(Token('EOF', ''))
    return tokens

class Parser:
    """
    Parser recursivo descendente para la gramática:
      expr   -> term ((+|-) term)*
      term   -> factor ((*|/) factor)*
      factor -> ENTERO | '(' expr ')'
    Genera un árbol sintáctico y evalúa simultáneamente.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def eat(self, token_type):
        """
        Consume el token actual si coincide con token_type, y avanza.
        Caso contrario, arroja un error de sintaxis.
        """
        if self.current_token.tipo == token_type:
            # print(f"[DEBUG] Consumiendo {self.current_token}")
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            print(f"[ERROR] Se esperaba {token_type}, pero se encontró {self.current_token.tipo}")
            sys.exit(1)

    def factor(self):
        """
        factor -> ENTERO | '(' expr ')'
        Retorna el valor numérico de factor.
        """
        token = self.current_token
        if token.tipo == 'ENTERO':
            value = int(token.valor)
            self.eat('ENTERO')
            return value
        elif token.tipo == 'LPAREN':
            self.eat('LPAREN')
            result = self.expr()
            if self.current_token.tipo != 'RPAREN':
                print("[ERROR] Falta ')'")
                sys.exit(1)
            self.eat('RPAREN')
            return result
        else:
            print(f"[ERROR] Factor inválido: {token}")
            sys.exit(1)

    def term(self):
        """
        term -> factor ((*|/) factor)*
        Aplica multiplicación y división con precedencia.
        """
        result = self.factor()
        while self.current_token.tipo in ('MULT', 'DIV'):
            token = self.current_token
            if token.tipo == 'MULT':
                self.eat('MULT')
                result *= self.factor()
            elif token.tipo == 'DIV':
                self.eat('DIV')
                divisor = self.factor()
                if divisor == 0:
                    print("[ERROR] División por cero.")
                    sys.exit(1)
                result //= divisor  # División entera
        return result

    def expr(self):
        """
        expr -> term ((+|-) term)*
        Maneja suma y resta con menor precedencia que term.
        """
        result = self.term()
        while self.current_token.tipo in ('MAS', 'MENOS'):
            token = self.current_token
            if token.tipo == 'MAS':
                self.eat('MAS')
                result += self.term()
            elif token.tipo == 'MENOS':
                self.eat('MENOS')
                result -= self.term()
        return result

    def parse(self):
        """
        Punto de entrada del parser: retorna el valor evaluado de la expresión.
        Verifica que al final se encuentre EOF.
        """
        value = self.expr()
        if self.current_token.tipo != 'EOF':
            print("[ERROR] Símbolos extra después de la expresión.")
            sys.exit(1)
        return value

