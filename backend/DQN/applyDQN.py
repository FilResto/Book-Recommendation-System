import torch
import pandas as pd
from model import DQNAgent
from environment import Environment
from training import load_data


def recommend_book(agent, environment, user_id, device=torch.device("cpu")):
    """
    Raccomanda un libro dato un utente specifico.
    :param agent: L'agente DQN addestrato.
    :param environment: L'ambiente che gestisce utenti e libri.
    :param user_id: ID dell'utente per cui fare la raccomandazione.
    :param device: Dispositivo (CPU).
    :return: Dettagli del libro raccomandato.
    """
    # Imposta il dispositivo su CPU
    agent.policy_net.to(device)

    # Reset dell'ambiente per l'utente specifico
    state = environment.user_recom(user_id=user_id)

    # Converte lo stato in tensore
    if not isinstance(state, torch.Tensor):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        state_tensor = state.to(torch.float32).to(device).unsqueeze(0)

     # Prevedi i valori Q per tutte le azioni (libri)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).squeeze(0)  # Rimuovi dimensione batch
    
    # Trova gli indici delle `num_books` migliori azioni
    top_actions = torch.topk(q_values, len(df_books)).indices.tolist()

    # Ottieni i dettagli dei libri corrispondenti
    recommended_books = []
    for action in top_actions:
        book_info = environment.books_df.iloc[action]  # Ottieni info libro da df
        recommended_books.append({
            "book_id": book_info["bookId"]
        })
    
    return recommended_books

def dqn(user_id):
    # Carica i dati e inizializza l'ambiente
    df_books, df_ratings, df_visualization, df_users = load_data()
    env = Environment(df_users, df_books, df_visualization, df_ratings)

    # Parametri del modello
    input_dim = 53  # Dimensione dello stato
    output_dim = len(df_books)  # Numero totale di libri (azioni possibili)

    # Inizializza l'agente DQN
    device = torch.device("cpu")
    #print(f"Utilizzo del dispositivo: {device}")
    agent = DQNAgent(input_dim, output_dim)
    agent.load_model()  # Carica il modello addestrato
    recommended_books = recommend_book(agent, env, user_id, device=device)
    return recommended_books

if __name__ == "__main__":
    # Carica i dati e inizializza l'ambiente
    df_books, df_ratings, df_visualization, df_users = load_data()
    env = Environment(df_users, df_books, df_visualization, df_ratings)

    # Parametri del modello
    input_dim = 53  # Dimensione dello stato
    output_dim = len(df_books)  # Numero totale di libri (azioni possibili)

    # Inizializza l'agente DQN
    device = torch.device("cpu")
    print(f"Utilizzo del dispositivo: {device}")
    agent = DQNAgent(input_dim, output_dim)
    agent.load_model()  # Carica il modello addestrato

    # ID dell'utente per cui fare la raccomandazione
    user_id = int(input("Inserisci l'ID utente: "))
    user = df_users[df_users["id"] == user_id].iloc[0]
    # Generi preferiti dell'utente
    preferred_genres = user["generi_preferiti"]

    # Raccomanda un libro
    recommended_books = recommend_book(agent, env, user_id, device=device)
    count = 0
    # Stampa i dettagli dei libri raccomandati
    if len(recommended_books) != 0:
        print(f"Libro raccomandato per l'utente {user_id}:")
        print(f"Generi Preferiti dal user: {preferred_genres}")
        print("Libri raccomandati:")
        
        for rank, book in enumerate(recommended_books, start=1):
            genre_match = any(genre in preferred_genres for genre in book["genre"])  # Controllo dei generi
            match_info = " - MATCH" if genre_match else ""
            #print(f"Posizione: {rank}, ID: {book['book_id']}, Titolo: {book['title']}, Genere: {book['genre']}{match_info}")
            
            # Se è un match, stampa la posizione
            if genre_match:
                count += 1
                if count <300:
                    print(f" -> Libro '{book['title']}' è in posizione {rank} ed è un match con i generi preferiti dell'utente.")
    else:
        print(f"Nessun libro raccomandato per l'utente {user_id}.")

    print(f"Numeri libri buoni: {count}")









