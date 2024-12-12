import torch
import pickle
import random
import numpy as np
from collections import deque
from model import DQNAgent
from environment import Environment
from replay_buffer import ReplayBuffer

def train_dqn(agent, environment, num_episodes=1000, batch_size=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500, target_update=10):
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

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    # Replay buffer
    memory = ReplayBuffer(capacity=10000)

    # Lista per tracciare la ricompensa media per ogni episodio
    rewards = []

    for episode in range(num_episodes):
        # Resetta l'ambiente per un nuovo episodio
        state = environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.policy_net.network[0].weight.device)  # Usa il device del modello
        steps = 0
        total_reward = 0
        done = False
        print(f"\n--- Episode {episode} ---") if episode < 10 else None

        while not done:
            # Seleziona l'azione con epsilon-greedy
            action = agent.select_action(torch.tensor(state, dtype=torch.float32), epsilon)

            # Esegui il passo nell'ambiente
            next_state, reward, done = environment.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=state.device)
            if episode < 10:  # Log solo per i primi 10 episodi
                print(f"Step {steps}: State={state}, Action={action}, Reward={reward}, Next State={next_state}, Done={done}")


            # Memorizza l'esperienza nel buffer
            memory.add_experience(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            # Ottimizza l'agente usando il buffer
            agent.optimize(memory, batch_size)

            # Aggiorna lo stato
            state = next_state

            # Accumula la ricompensa
            total_reward += reward
            steps+=1
        print(f"Total Reward: {total_reward}") if episode < 10 else None

        # Decrescita di epsilon
        if epsilon > epsilon_end:
            epsilon -= epsilon_decay_rate

        # Aggiungi la ricompensa media per questo episodio
        rewards.append(total_reward)

        # Aggiorna la rete target
        if episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Salva il modello periodicamente
        if episode % 10 == 0:
            agent.save_model()
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

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

if __name__ == "__main__":

    # Carica i dati
    df_books,df_ratings,df_visualization,df_users = load_data()
    # Inizializza l'ambiente
    env = Environment(df_users, df_books, df_visualization, df_ratings)

    # Inizializza l'agente
    input_dim = 3280  # La dimensione dello stato
    output_dim = len(df_books)  # La dimensione dello spazio delle azioni (numero di libri)
    agent = DQNAgent(input_dim=input_dim, output_dim=output_dim)

    # Addestra l'agente
    rewards = train_dqn(agent, env, num_episodes=1000, batch_size=64)

    # Salva il modello finale
    agent.save_model()
