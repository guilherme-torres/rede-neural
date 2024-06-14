import json
import numpy as np

def relu(x):
    return np.maximum(0, x)


def relu_dx(x):
    return 0 if x < 0 else 1


# Rede Neural com apenas uma camada oculta
class RedeNeural:
    def __init__(self, entrada, oculta, saida):
        # seta valores aleatórios para todos os pesos e bias
        self.tam_entrada = entrada
        self.tam_oculta = oculta
        self.tam_saida = saida
        self.pesos_entrada_oculta = np.random.rand(self.tam_entrada, self.tam_oculta) - 0.5
        self.bias_oculta = np.random.rand(self.tam_oculta) - 0.5
        self.pesos_oculta_saida = np.random.rand(self.tam_oculta, self.tam_saida) - 0.5
        self.bias_saida = np.random.rand(self.tam_saida) - 0.5


    def feedforward(self, X):
        # entrada para oculta
        z_oculta = X.dot(self.pesos_entrada_oculta) + self.bias_oculta
        ativacao_oculta = np.array([relu(z) for z in z_oculta])

        # oculta para saida
        z_saida = ativacao_oculta.dot(self.pesos_oculta_saida) + self.bias_saida
        ativacao_saida = np.array([relu_dx(z) for z in z_saida])

        return ativacao_saida


    # atualizar os pesos e bias aleatoriamente
    def aprender(self):
        self.pesos_entrada_oculta = np.random.rand(self.tam_entrada, self.tam_oculta) - 0.5
        self.bias_oculta = np.random.rand(self.tam_oculta) - 0.5
        self.pesos_oculta_saida = np.random.rand(self.tam_oculta, self.tam_saida) - 0.5
        self.bias_saida = np.random.rand(self.tam_saida) - 0.5


    def obter_parametros(self):
        return {
            "pesos_entrada_oculta": self.pesos_entrada_oculta.tolist(),
            "bias_oculta": self.bias_oculta.tolist(),
            "pesos_oculta_saida": self.pesos_oculta_saida.tolist(),
            "bias_saida": self.bias_saida.tolist()
        }
    
    def salvar_parametros(self):
        parametros = self.obter_parametros()
        with open('parametros_rede_neural.json', 'w') as arquivo:
            json.dump(parametros, arquivo)

    