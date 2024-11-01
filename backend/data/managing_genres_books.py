import pickle
import pandas as pd
from collections import Counter
import random

def assign_random_genres():
    # Carica il DataFrame dei libri
    with open('df_book.pkl', 'rb') as file:
        df_books = pickle.load(file)

    # Assicurati che la colonna 'genres' sia una lista e "esplodi" la colonna per ottenere i singoli generi
    df_books_exploded = df_books.explode('genres')

    # Conta la frequenza di ciascun genere e crea una lista dei top 50 generi
    genre_counts = Counter(df_books_exploded['genres'])
    top_50_genres = [genre for genre, _ in genre_counts.most_common(50)]

    # Carica il DataFrame degli utenti
    with open('df_user.pkl', 'rb') as file:
        df_users = pickle.load(file)

    # Funzione per assegnare 2 generi casuali a ciascun utente
    def assign_random_preferred_genres(_):
        return random.sample(top_50_genres, 3)

    # Cancella i generi preferiti esistenti e assegna 3 generi casuali dalla lista top_50
    df_users['generi_preferiti'] = df_users['generi_preferiti'].apply(assign_random_preferred_genres)

    # Salva il DataFrame degli utenti aggiornato
    with open('df_user_randomized.pkl', 'wb') as file:
        pickle.dump(df_users, file)



# Esegui la funzione
assign_random_genres()
