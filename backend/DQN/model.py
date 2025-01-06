import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Inizializza la rete neurale DQN.
        :param input_dim: Dimensione dell'input (stato codificato).
        :param output_dim: Numero di azioni (dimensione dello spazio delle azioni).
        """
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
            
        )


    def forward(self, x):
        """
        Passa i dati attraverso la rete.
        :param x: Input (batch di stati).
        :return: Valutazione Q per ogni azione.
        """
        return self.network(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-5, gamma=0.99, target_update=10, model_path="dqn_model.pth"):
        """
        Inizializza l'agente DQN.
        :param input_dim: Dimensione dell'input (stato codificato).
        :param output_dim: Numero di azioni.
        :param lr: Tasso di apprendimento per l'ottimizzatore.
        :param gamma: Fattore di sconto per il calcolo del valore futuro.
        :param target_update: Frequenza con cui aggiornare la rete target.
        :param model_path: Percorso per salvare/caricare il modello.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilizzo del dispositivo: {self.device}")
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.target_update = target_update
        self.model_path = model_path
        self.steps_done = 0
        self.losses = []  # Per monitorare la perdita

    def select_action(self, state, epsilon):
        """
        Seleziona un'azione usando epsilon-greedy.
        :param state: Stato corrente (torch.Tensor).
        :param epsilon: Probabilità di esplorare (selezionare un'azione casuale).
        :return: Indice dell'azione selezionata.
        """
        if torch.rand(1).item() < epsilon:
            #print("esplorazione")
            return torch.randint(0, self.policy_net.network[-1].out_features, (1,)).item(),0
        else:
            #print("sfruttamento")
            state = state.to(self.device).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item(),1

    def optimize(self, memory, batch_size):
        """
        Ottimizza la rete basandosi sull'esperienza raccolta.
        :param memory: Replay buffer contenente esperienze (state, action, reward, next_state, done).
        :param batch_size: Dimensione del batch per l'apprendimento.
        """
        if len(memory) < batch_size:
            return

        # Campiona esperienze casuali dal buffer
        batch = memory.sample_batch(batch_size)
        state, action, reward, next_state, done, indices,weights = batch
        # Converti gli stati e gli stati successivi in tensori
        state = torch.stack([torch.tensor(s, dtype=torch.float32) for s in state]).to(self.device)
        next_state = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_state]).to(self.device)

        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(1)  # Azioni devono essere 2D
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Calcola Q(s, a) usando la policy network
        q_values = self.policy_net(state).gather(1, action)

        # Calcola Q_target
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + (1 - done) * self.gamma * next_q_values

        # Calcola la perdita
        loss = self.loss_fn(q_values, q_target)
        self.losses.append(loss.item())
        # Ottimizza la rete
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Aggiorna la rete target se necessario
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """
        Salva i pesi del modello su file.
        """
        torch.save(self.policy_net.state_dict(), self.model_path)

    def load_model(self):
        """
        Carica i pesi del modello da file.
        """
        if os.path.exists(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))  # Assicurati che venga caricato sulla GPU
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            print("Nessun modello salvato trovato.")
