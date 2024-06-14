import sys
from random import randint
import numpy as np
import pygame
from rede_neural import RedeNeural

pygame.init()

LARGURA_JANELA = 600
ALTURA_JANELA = 300
COR_DE_FUNDO = (255, 255, 255) # branco
GRAVIDADE = 1

janela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))

pygame.display.set_caption('rede neural')

class Agente:
    def __init__(self):
        self.LARGURA = 50
        self.ALTURA = 50
        self.COR = (255, 0, 0)
        self.x = janela.get_width() / 2 - self.LARGURA
        self.y = janela.get_height() - self.ALTURA
        self.PULO = -15
        self.velocidade_y = 0
        self.caindo = False
        self.cerebro = RedeNeural(2, 6, 1) # entrada -> distância e velocidade do obstáculo; saída -> 1(pular) ou 0(não pular)
    
    def desenha(self):
        return pygame.draw.rect(janela, self.COR, [self.x, self.y, self.LARGURA, self.ALTURA], 0)

    def pular(self):
        # se estiver no chão então pula
        if self.y == janela.get_height() - self.ALTURA:
            self.caindo = True
            self.velocidade_y = self.PULO
    
    def atualiza(self):
        if self.caindo:
            self.velocidade_y += GRAVIDADE
            self.y += self.velocidade_y

        if self.y > janela.get_height() - self.ALTURA:
            self.y = janela.get_height() - self.ALTURA
            self.caindo = False


class Obstaculo:
    def __init__(self):
        self.LARGURA = 20
        self.ALTURA = 40
        self.COR = (0, 0, 255)
        self.x = janela.get_width()
        self.y = janela.get_height() - self.ALTURA
        self.VELOCIDADE_X = -6

    def desenha(self):
        return pygame.draw.rect(janela, self.COR, [self.x, self.y, self.LARGURA, self.ALTURA], 0)

    def atualiza(self):
        self.x += self.VELOCIDADE_X
        if self.x < -self.LARGURA:
            self.x = janela.get_width()
            self.VELOCIDADE_X = randint(-12, -6)


agente = Agente()
obstaculo = Obstaculo()

clock = pygame.time.Clock()

# loop infinito
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # if event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_UP:
        #         agente.pular()
    
    janela.fill(COR_DE_FUNDO)
    
    superficie_obstaculo = obstaculo.desenha()
    superficie_agente = agente.desenha()

    distancia_proximo_obstaculo = obstaculo.x - agente.x

    entrada = np.array([distancia_proximo_obstaculo, obstaculo.VELOCIDADE_X])
    saida = agente.cerebro.feedforward(entrada)
    if saida[0] == 1:
        agente.pular()

    if superficie_agente.colliderect(superficie_obstaculo):
        print('colidiu!')
        agente.cerebro.aprender()
        obstaculo.x = janela.get_width()
        # agente.cerebro.salvar_parametros()

    agente.atualiza()
    obstaculo.atualiza()

    pygame.display.update()
    clock.tick(60)
