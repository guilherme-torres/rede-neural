import json
import numpy as np

# Essas são as funcões de ativação, elas definem se um neurônio da rede
# será ativado ou não, ou seja, se for 0, o neurônio não será ativado, se for outro valor,
# ele será ativado.
#
# Sem as funções de ativação, nossa rede neural não seria capaz de formar relações complexas
# entre as entradas e as saídas, ou seja, relações não lineares.
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


    # Aqui é um processo comum em praticamente todos os tipos de rede neural.
    # O feedforward é basicamente passar os dados de entrada até a saída
    # através de uma sequência de operações de matrizes (matemática que aprendemos no ensino médio)
    def feedforward(self, X):
        # entrada para oculta
        z_oculta = X.dot(self.pesos_entrada_oculta) + self.bias_oculta
        ativacao_oculta = np.array([relu(z) for z in z_oculta])

        # oculta para saida
        z_saida = ativacao_oculta.dot(self.pesos_oculta_saida) + self.bias_saida
        ativacao_saida = np.array([relu_dx(z) for z in z_saida])

        return ativacao_oculta, ativacao_saida


    # A maneira de atualizar os pesos da rede neural de forma que ela consiga
    # pegar um conjunto de entradas e mapear para as saídas corretas é a chave para o aprendizado
    # 
    # Aqui eu apenas atualizo os parâmetros da rede de forma aleatória até encontrar uma configuração ideal.
    # Desse modo, essa rede aprende na força bruta
    def aprender(self):
        self.pesos_entrada_oculta = np.random.rand(self.tam_entrada, self.tam_oculta) - 0.5
        self.bias_oculta = np.random.rand(self.tam_oculta) - 0.5
        self.pesos_oculta_saida = np.random.rand(self.tam_oculta, self.tam_saida) - 0.5
        self.bias_saida = np.random.rand(self.tam_saida) - 0.5


    # As funções a seguir não influenciam no aprendizado, são apenas para salvar os parâmetros da rede em um arquivo
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

    