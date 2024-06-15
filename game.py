import sys
from random import randint
import numpy as np
import pygame
from rede_neural import RedeNeural

pygame.init()

LARGURA_JANELA = 600
ALTURA_JANELA = 300
COR_DE_FUNDO = (255, 255, 255)
GRAVIDADE = 1

janela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_JANELA))

pygame.display.set_caption('rede neural')

class Agente:
    def __init__(self):
        self.LARGURA = 50
        self.ALTURA = 50
        self.COR = (255, 0, 0)
        self.x = LARGURA_JANELA / 2 - self.LARGURA
        self.y = ALTURA_JANELA - self.ALTURA
        self.PULO = -15
        self.velocidade_y = 0
        self.caindo = False
        self.cerebro = RedeNeural(2, 6, 1) # entrada -> distância e velocidade do obstáculo; saída -> 1(pular) ou 0(não pular)
    
    def desenha(self):
        return pygame.draw.rect(janela, self.COR, [self.x, self.y, self.LARGURA, self.ALTURA], 0)

    def pular(self):
        # se estiver no chão então pula
        if self.y == ALTURA_JANELA - self.ALTURA:
            self.caindo = True
            self.velocidade_y = self.PULO
    
    def atualiza(self):
        if self.caindo:
            self.velocidade_y += GRAVIDADE
            self.y += self.velocidade_y

        if self.y > ALTURA_JANELA - self.ALTURA:
            self.y = ALTURA_JANELA - self.ALTURA
            self.caindo = False


class Obstaculo:
    def __init__(self):
        self.LARGURA = 20
        self.ALTURA = 40
        self.COR = (0, 0, 255)
        self.x = LARGURA_JANELA
        self.y = ALTURA_JANELA - self.ALTURA
        self.VELOCIDADE_X = -6

    def desenha(self):
        return pygame.draw.rect(janela, self.COR, [self.x, self.y, self.LARGURA, self.ALTURA], 0)

    def atualiza(self):
        self.x += self.VELOCIDADE_X
        if self.x < -self.LARGURA:
            self.x = LARGURA_JANELA
            self.VELOCIDADE_X = randint(-12, -6)


class RepresentacaoDaRede:
    def __init__(self, rede):
        self.rede = rede
        self.tam_entrada = self.rede.tam_entrada
        self.tam_oculta = self.rede.tam_oculta
        self.tam_saida = self.rede.tam_saida
        self.COR_NEURONIO_ATIVO = (255, 255, 0)

    def desenha_neuronio(self, x, y, neuronio_ativo=True):
        if neuronio_ativo:
            return pygame.draw.circle(janela, self.COR_NEURONIO_ATIVO, (x, y), 15, 0)
        else:
            return pygame.draw.circle(janela, (0, 0, 0), (x, y), 15, 3)

    def desenha_rede(self, ativacao_oculta, ativacao_saida):
        X_REDE = 50
        entrada = []
        oculta = []
        saida = []
        for i in range(self.tam_entrada):
            entrada.append(self.desenha_neuronio(x=X_REDE, y=80+50*(i+1)*0.8))
        for i, n in enumerate(ativacao_oculta):
            oculta.append(self.desenha_neuronio(x=X_REDE+60, y=50*(i+1)*0.8, neuronio_ativo=True if n > 0 else False))
        for i, n in enumerate(ativacao_saida):
            saida.append(self.desenha_neuronio(x=X_REDE+120, y=100+50*(i+1)*0.8, neuronio_ativo=True if n > 0 else False))
        
        for i in entrada:
            for j in oculta:
                for k in saida:
                    pygame.draw.lines(janela, (0, 0, 0), False, [i.center, j.center, k.center])


agente = Agente()
obstaculo = Obstaculo()
rede = RepresentacaoDaRede(agente.cerebro)

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
    oculta, saida = agente.cerebro.feedforward(entrada)
    if saida[0] == 1:
        agente.pular()

    rede.desenha_rede(oculta, saida)
    # colisão entre o agente e o obstáculo
    if superficie_agente.colliderect(superficie_obstaculo):
        print('colidiu!')
        agente.cerebro.aprender()
        obstaculo.x = LARGURA_JANELA
        # agente.cerebro.salvar_parametros()

    agente.atualiza()
    obstaculo.atualiza()

    pygame.display.update()
    clock.tick(60)
