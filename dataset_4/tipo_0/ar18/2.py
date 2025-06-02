#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Juego Snake clásico implementado con pygame.
Controles: flechas (arriba, abajo, izquierda, derecha).
El objetivo es comer la comida para crecer, evitando chocar con las paredes
o con el propio cuerpo.
"""

import pygame
import sys
import random

# Definición de colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)

# Tamaño de la ventana y tamaño de bloque (snake y comida)
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
BLOCK_SIZE = 20

# Velocidad inicial (bloques por segundo)
SPEED = 10

def draw_block(surface, color, position):
    """
    Dibuja un cuadrado de tamaño BLOCK_SIZE en la posición dada (x, y).
    """
    rect = pygame.Rect(position[0], position[1], BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(surface, color, rect)

def random_food_position():
    """
    Retorna una posición aleatoria (x, y) alineada a la cuadricula BLOCK_SIZE.
    """
    x = random.randint(0, (WINDOW_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    y = random.randint(0, (WINDOW_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
    return (x, y)


"""
Configura pygame, inicializa juego, y corre el bucle principal.
Maneja eventos de teclado, movimiento de la serpiente, colisiones,
puntuación y dibuja todo en pantalla.
"""
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

# Posición inicial de la serpiente (lista de bloques)
snake = [(WINDOW_WIDTH//2, WINDOW_HEIGHT//2)]
direction = (0, -BLOCK_SIZE)  # Inicialmente moviéndose hacia arriba

food_pos = random_food_position()
score = 0

game_over = False

while True:
    # Manejo de eventos (teclado, cierre de ventana)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # Cambiar dirección según flechas, evitando movimiento inverso directo
            if event.key == pygame.K_UP and direction != (0, BLOCK_SIZE):
                direction = (0, -BLOCK_SIZE)
            elif event.key == pygame.K_DOWN and direction != (0, -BLOCK_SIZE):
                direction = (0, BLOCK_SIZE)
            elif event.key == pygame.K_LEFT and direction != (BLOCK_SIZE, 0):
                direction = (-BLOCK_SIZE, 0)
            elif event.key == pygame.K_RIGHT and direction != (-BLOCK_SIZE, 0):
                direction = (BLOCK_SIZE, 0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # Cambiar dirección según flechas, evitando movimiento inverso directo
            if event.key == pygame.K_UP and direction != (0, BLOCK_SIZE):
                direction = (0, -BLOCK_SIZE)
            elif event.key == pygame.K_DOWN and direction != (0, -BLOCK_SIZE):
                direction = (0, BLOCK_SIZE)
            elif event.key == pygame.K_LEFT and direction != (BLOCK_SIZE, 0):
                direction = (-BLOCK_SIZE, 0)
            elif event.key == pygame.K_RIGHT and direction != (-BLOCK_SIZE, 0):
                direction = (BLOCK_SIZE, 0)

    if not game_over:
        # Mover la serpiente: nueva cabeza en la dirección actual
        head_x, head_y = snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])
        snake.insert(0, new_head)

        # Verificar colisión con comida
        if new_head == food_pos:
            score += 1
            food_pos = random_food_position()
        else:
            # Eliminar cola si no comió
            snake.pop()

        # Verificar colisión con paredes
        if (new_head[0] < 0 or new_head[0] >= WINDOW_WIDTH or
            new_head[1] < 0 or new_head[1] >= WINDOW_HEIGHT):
            game_over = True

        # Verificar colisión con sí misma (cabeza en otra parte del cuerpo)
        if new_head in snake[1:]:
            game_over = True

    # Dibujar fondo
    screen.fill(BLACK)

    # Dibujar comida
    draw_block(screen, RED, food_pos)

    # Dibujar serpiente
    for block in snake:
        draw_block(screen, GREEN, block)

    # Mostrar puntuación en esquina superior izquierda
    font = pygame.font.SysFont(None, 35)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    if game_over:
        # Mostrar mensaje de Game Over
        over_font = pygame.font.SysFont(None, 75)
        over_text = over_font.render("GAME OVER", True, RED)
        screen.blit(over_text, (WINDOW_WIDTH//2 - over_text.get_width()//2,
                                WINDOW_HEIGHT//2 - over_text.get_height()//2))

    pygame.display.flip()
    clock.tick(SPEED)

