import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Inizializza il replay buffer.
        :param capacity: Capacità massima del buffer.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add_experience(self, state, action, reward, next_state, done):
        """
        Aggiunge una nuova esperienza al buffer.
        :param state: Stato corrente.
        :param action: Azione intrapresa.
        :param reward: Ricompensa ricevuta.
        :param next_state: Stato successivo.
        :param done: Flag che indica se l'episodio è terminato.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        """
        Campiona un mini-batch di esperienze casuali dal buffer.
        :param batch_size: Numero di esperienze da campionare.
        :return: Batch di esperienze.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        """
        Restituisce il numero corrente di esperienze memorizzate nel buffer.
        :return: Lunghezza del buffer.
        """
        return len(self.buffer)
