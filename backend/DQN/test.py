import torch
import numpy as np
from model import DQNAgent
from environment import Environment
import pandas as pd
from training import load_data

def test_agent(agent, environment, num_tests=100):
    """
    Testa l'agente su un determinato numero di episodi.
    :param agent: L'agente DQN addestrato.
    :param environment: L'ambiente per il test.
    :param num_tests: Numero di episodi di test.
    :return: Reward medio e raccomandazioni generate.
    """
    total_reward = 0
    recommendations = []

    for _ in range(num_tests):
        state = environment.reset()
        done = False
        episode_reward = 0

        while not done:
            # Converte lo stato in tensore per il modello
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # L'agente sceglie l'azione migliore (greedy, no esplorazione)
            action = agent.policy_net(state_tensor).argmax(dim=1).item()

            # Esegui un passo nell'ambiente
            next_state, reward, done = environment.step(action)
            episode_reward += reward

            # Aggiorna lo stato corrente
            state = next_state

            # Salva le raccomandazioni per il reporting
            if action not in recommendations:
                recommendations.append(action)

        total_reward += episode_reward

    avg_reward = total_reward / num_tests
    return avg_reward, recommendations

if __name__ == "__main__":
    df_books,df_ratings,df_visualization,df_users = load_data()
    # Inizializza l'ambiente e l'agente
    env = Environment(df_users, df_books, df_visualization, df_ratings)
    input_dim = len(env.reset())  # Dimensione dello stato
    output_dim = len(df_books)  # Numero totale di libri (azioni possibili)

    agent = DQNAgent(input_dim, output_dim)
    agent.load_model()  # Carica il modello addestrato

    # Testa l'agente
    avg_reward, recommendations = test_agent(agent, env, num_tests=100)

    # Stampa i risultati
    print(f"Reward medio: {avg_reward:.2f}")
    print("Raccomandazioni generate:")
    for rec in recommendations:
        book_info = df_books[df_books["bookId"] == rec]
        print(f"Book ID: {rec}, Titolo: {book_info['title'].values[0]}")
