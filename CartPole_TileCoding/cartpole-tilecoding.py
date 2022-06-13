from statistics import median_low
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
random.seed(42)
from tiles3 import tiles, IHT

#Alunos:
# Aldemir Melo
# Darlysson Olimpio
# Sandoval Almeida
# Tayco Murilo

DISCOUNT = 0.99  # gamma
EXPLORE_RATE = 0.1  # epsilon
# LEARNING_RATE = 0.05  # alpha
EPISODES = (50, 10)                     #Tupla definida em (numeros de episodios[0], analise de ambiente[1])
MAX_ITER = 250                          #Numero de vez que o ambiente é analisado
RENDER = False

maxSize = 2048      
iht = IHT(maxSize)                      #Hash de tamanho 2048
numTilings = 32                         #Numero de Tiles
stepSize = EXPLORE_RATE/numTilings      #Processo de atualização cada ladrilho(tile), não sendo necessário calcular ao fim de cada passo

weights = np.zeros(shape=maxSize)       #Peso inicialmente zerado

def mytiles(X, tile_dim=5.0, min_x=-4., max_x=4.):              #Posicionamento e definição dos tiles
    scaleFactor = tile_dim / (max_x - min_x)
    X[0] *= tile_dim/(2*2.4)
    X[1] *= tile_dim/(2*5)
    X[2] *= tile_dim/(2*0.2)
    X[3] *= tile_dim/2
    return tiles(iht, numTilings, X)

def v_hat(X):                                                   #Dado o estado de entrada, após o aprendizado, sabe-se qual o peso para tal estado
    return weights[mytiles(X)].sum()

def qlearning(weights, state_tiles, curr_state, action, v_hat_next, reward):        #acumulo de aprendizado por tile levando em consideração recomenpenssas e descontos atuadores
    weights[state_tiles] += stepSize * (
                        reward + DISCOUNT * v_hat_next - v_hat(
                        np.concatenate([curr_state, [action]])
                    )
                )

def test(MAX_ITER, observation):                                                    #Teste de aprendizado
    for t in range(MAX_ITER):
        cart_pole.render()
        curr_state = observation

        #Baseado no Estado Corrente, em V_hat verifica-se qual o peso para o estado em questão
        push_left = v_hat(np.concatenate([curr_state, [0]]))        
        push_right = v_hat(np.concatenate([curr_state, [1]]))
        action = 1 if push_right > push_left else 0                                 #Tido o retorno nas linhas anteriores, é possível descobrirmos  qual a ação viável para o estado atual

        observation, reward, done, info = cart_pole.step(action)
        cart_pos = observation[0]                                                   #Atualiza a posição do carro 
        if (
            cart_pos < cart_pole.observation_space.low[0]
            or cart_pos > cart_pole.observation_space.high[0]
        ):
            break

def plot(accumulated_return):                                                       #PLOT
    plt.figure(figsize=(8, 4))
    plt.plot(accumulated_return)
    plt.xlabel("epoch")
    plt.title("Mean accumulated return")
    plt.savefig("acc-return-cont.png")
    plt.show()

if __name__ == "__main__":

    cart_pole = gym.make("CartPole-v1")

    accumulated_return = np.zeros(shape=EPISODES[0])

    episodes_loop = trange(EPISODES[0])

    #Treinamento
    for i_episode in episodes_loop:
        acc_ret = np.zeros(shape=EPISODES[1])                                               #Retorno Acumulado
        for i in range(EPISODES[1]):
            observation = cart_pole.reset()                                                 #Estados observados
            for t in range(MAX_ITER):
                cart_pole.render() if RENDER else None
                curr_state = observation[:]                                                 #Estado Atual
                if np.random.uniform() < EXPLORE_RATE:
                    action = cart_pole.action_space.sample()                                #Enquanto não explorou o suficiente
                else:                                                                  
                    push_left = v_hat(np.concatenate([curr_state, [0]]))                    #Após ter exploradoo suficiente, o array de pesos ja nos dá resposta suficientes para diferentes estados
                    push_right = v_hat(np.concatenate([curr_state, [1]]))
                    action = 1 if push_right > push_left else 0
                observation, reward, done, info = cart_pole.step(action)                    #Atualiza o ambiente
                next_state = observation[:]                                                 #Prox passo

                state_tiles = mytiles(np.concatenate([curr_state, [action]]))
                
                v_hat_next = np.max(                                                        #Maximo de Aproximação
                    np.array(
                        [
                            v_hat(np.concatenate(
                                [next_state, [a]])) for a in range(2)
                        ]
                    )
                )

                qlearning(weights, state_tiles, curr_state, action, v_hat_next, reward)     #acumulo de aprendizado por tile levando em consideração recomenpenssas e descontos atuadores

                if done:                                                                    #Finalizado o Episódio
                    acc_ret[i] = t + 1
                    episodes_loop.set_description(
                        desc="Episode {0:0>4} fineshed after {1:0>3} \
                            timesteps".format(
                            i_episode + 1, t + 1
                        )
                    )
                    break
        accumulated_return[i_episode] = acc_ret.mean()                                      #Taxa de aprendizado acumulado do episodio armazenada, e calculada a media

    cart_pole.close() if RENDER else None

    np.save('weights', weights)

    plot(accumulated_return)

    observation = cart_pole.reset()

    test(MAX_ITER, observation)                                                             #Teste após aprendizado

    cart_pole.close()