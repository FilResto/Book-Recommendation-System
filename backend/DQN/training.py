import torch
import pickle
import random
import numpy as np
from collections import deque
import math
from model import DQNAgent
from environment import Environment
from replay_buffer import PrioritizedReplayBuffer
import torch.nn as nn
import matplotlib.pyplot as plt

def train_dqn(agent, environment, num_episodes=10000, batch_size=64, epsilon_start=1.0, epsilon_end=0.2, epsilon_decay=400, target_update=10):
    """
    Addestra un agente DQN nell'ambiente specificato.
    :param agent: L'agente DQN.
    :param environment: L'ambiente di simulazione.
    :param num_episodes: Numero di episodi per l'addestramento.
    :param batch_size: Dimensione del batch per l'ottimizzazione.
    :param epsilon_start: Valore iniziale di epsilon (probabilità di esplorazione).
    :param epsilon_end: Valore finale di epsilon (probabilità di esplorazione minima).
    :param epsilon_decay: Numero di passi in cui epsilon decresce da epsilon_start a epsilon_end.
    :param target_update: Frequenza di aggiornamento della rete target.
    """

    k = math.log(epsilon_start / epsilon_end) / epsilon_decay

    # Replay buffer
    memory = PrioritizedReplayBuffer(capacity=100000)

    # Lista per tracciare la ricompensa media per ogni episodio
    rewards = []
    tot_sfruttamento = 0
    num_sfrut = 0
    tot_try = 0
    for episode in range(num_episodes):
        # Resetta l'ambiente per un nuovo episodio
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-k * episode)
        state = environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.policy_net.network[0].weight.device)  # Usa il device del modello
        steps = 0
        total_reward = 0
        
        done = False
        print(f"\n--- Episode {episode} ---") if episode < 10 else None

        while not done:
            tot_try += 1
            # Seleziona l'azione con epsilon-greedy
            action,sfrutto = agent.select_action(state.clone().detach().to(torch.float32), epsilon)
            
            # Esegui il passo nell'ambiente
            next_state, reward, done = environment.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=state.device)
            if(sfrutto == 1):
              #print("Reward sfruttamento:",reward)
              tot_sfruttamento += reward
              num_sfrut += 1
            if episode < 10:  # Log solo per i primi 10 episodi
                print(f"Step {steps}: State={state}, Action={action}, Reward={reward}, Next State={next_state}, Done={done}\n\n")

            # Memorizza l'esperienza nel buffer

            memory.add_experience(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, reward)

            # Ottimizza l'agente usando il buffer
            agent.optimize(memory, batch_size)

            # Aggiorna lo stato
            state = next_state

            # Accumula la ricompensa
            total_reward += reward
            steps+=1
        print(f"Total Reward: {total_reward} in {steps} steps") if episode < 10 else None


        # Aggiungi la ricompensa media per questo episodio
        rewards.append(total_reward)

        # Aggiorna la rete target
        if episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Salva il modello periodicamente
        if episode % 100 == 0:
            agent.save_model()
            if num_sfrut == 0:
              num_sfrut = 1
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}, Media Sfruttamento: {tot_sfruttamento/num_sfrut}, Sfruttamenti: {num_sfrut}/{tot_try}")
            tot_sfruttamento = 0
            num_sfrut = 0
            tot_try = 0
    return rewards


def load_data():
    with open('../data/PICKLE/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("../data/PICKLE/df_visualization.pkl", "rb") as file:
        df_visualizations = pickle.load(file)
    with open("../data/PICKLE/df_ratings.pkl", "rb") as file:
        df_ratings = pickle.load(file)
    with open("../data/PICKLE/df_user.pkl", "rb") as file:
        df_users = pickle.load(file)
    return df_book, df_ratings, df_visualizations,df_users




# Dopo l'addestramento
def plot_losses(agent):
    plt.figure(figsize=(10, 6))
    plt.plot(agent.losses, label='Loss')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('ricompense_training.png', dpi=300, bbox_inches='tight')
    plt.show()


def initialize_weights(layer):
    """
    Inizializza i pesi dei layer della rete.
    """
    if isinstance(layer, nn.Linear):  # Applica solo ai layer lineari
        # Xavier initialization (ottimale per ReLU e funzioni simili)
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)  # Imposta un valore iniziale per i bias


if __name__ == "__main__":
     # Carica i dati
    df_books,df_ratings,df_visualization,df_users = load_data()
    # Inizializza l'ambiente
    env = Environment(df_users, df_books, df_visualization, df_ratings)

    # Inizializza l'agente
    input_dim = 53  # La dimensione dello stato
    output_dim = len(df_books)  # La dimensione dello spazio delle azioni (numero di libri)
    agent = DQNAgent(input_dim=input_dim, output_dim=output_dim)
    # Inizializza i pesi della policy network e della target network
    agent.policy_net.apply(initialize_weights)
    agent.target_net.apply(initialize_weights)
    # Addestra l'agente
    rewards = train_dqn(agent, env, num_episodes=1000, batch_size=64)
    plot_losses(agent)
    # Salva il modello finale

    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, marker='o', color='b', linestyle='-', label='Total Reward')

    # Aggiungi titoli e etichette
    plt.title('Reward per Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Reward Totale')
    plt.legend()

    # Salva il grafico come immagine
    plt.savefig('reward_per_episode.png')

    # Mostra il grafico
    plt.show()
    agent.save_model()
