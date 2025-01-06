import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Inizializza il replay buffer con priorità.
        :param capacity: Capacità massima del buffer.
        :param alpha: Fattore che controlla quanto influiscono le priorità (0 = no priorità, 1 = solo priorità).
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.position = 0

    def add_experience(self, state, action, reward, next_state, done, priority):
        """
        Aggiunge una nuova esperienza al buffer con priorità basata sulla ricompensa.
        :param state: Stato corrente.
        :param action: Azione intrapresa.
        :param reward: Ricompensa ricevuta.
        :param next_state: Stato successivo.
        :param done: Flag che indica se l'episodio è terminato.
        """
        # Calcolo della priorità basata sulla ricompensa
        priority = max(priority, 1e-6)  # Priorità minima per evitare 0
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size, beta=0.4, uniform_weight=0.1):
        """
        Campiona un mini-batch di esperienze bilanciando priorità e diversità.
        :param batch_size: Numero di esperienze da campionare.
        :param beta: Peso per la correzione del bias.
        :param uniform_weight: Peso per il campionamento uniforme.
        :return: Batch di esperienze e pesi d'importanza.
        """
        if len(self.buffer) == 0:
            raise ValueError("Il buffer è vuoto. Non ci sono esperienze da campionare.")

        # Calcolo delle probabilità di campionamento
        priorities = np.array(self.priorities, dtype=np.float32)
        uniform_prob = np.ones_like(priorities) / len(priorities)
        probabilities = priorities ** self.alpha + uniform_weight * uniform_prob
        probabilities /= probabilities.sum()

        # Campionamento con sostituzione
        indices = random.choices(range(len(self.buffer)), k=batch_size, weights=probabilities)

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Calcolo dei pesi d'importanza
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalizzazione

        return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        np.array(indices, dtype=np.int64),
        np.array(weights, dtype=np.float32)
        )


    def update_priorities(self, indices, priorities):
        """
        Aggiorna le priorità delle esperienze campionate.
        :param indices: Indici delle esperienze da aggiornare.
        :param priorities: Nuove priorità calcolate.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-6)  # Evita priorità zero

    def __len__(self):
        return len(self.buffer)
