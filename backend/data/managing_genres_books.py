import pickle
import pandas as pd
from collections import Counter

# Carica il DataFrame dei libri
with open('df_book.pkl', 'rb') as file:
    df_books = pickle.load(file)

# Assicurati che la colonna 'genres' sia una lista e "esplodi" la colonna per ottenere i singoli generi
df_books_exploded = df_books.explode('genres')

# Conta la frequenza di ciascun genere e crea una lista dei top 100 generi
genre_counts = Counter(df_books_exploded['genres'])
top_100_genres = [genre for genre, _ in genre_counts.most_common(100)]

# Funzione per filtrare i generi di un libro e un utente mantenendo solo quelli presenti nella top 100
def filter_genres(genres):
    return [genre for genre in genres if genre in top_100_genres]

# Applica la funzione di filtro alla colonna 'genres' del DataFrame dei libri
df_books['genres'] = df_books['genres'].apply(filter_genres)

# Stampa la lista dei generi unici nei libri prima e dopo il filtraggio
unique_genres_books_before = df_books_exploded['genres'].unique()

unique_genres_books_after = df_books.explode('genres')['genres'].unique()

# Salva il DataFrame dei libri aggiornato
with open('df_book_filtered.pkl', 'wb') as file:
    pickle.dump(df_books, file)

# Carica il DataFrame degli utenti
with open('df_user.pkl', 'rb') as file:
    df_users = pickle.load(file)

# Stampa la lista dei generi unici preferiti dagli utenti prima del filtraggio
unique_genres_users_before = df_users.explode('generi_preferiti')['generi_preferiti'].unique()
print("Numero di generi preferiti unici negli utenti prima del filtraggio:", len(unique_genres_users_before))

# Applica la funzione di filtro alla colonna 'generi_preferiti' del DataFrame degli utenti
df_users['generi_preferiti'] = df_users['generi_preferiti'].apply(filter_genres)

# Stampa la lista dei generi unici preferiti dagli utenti dopo il filtraggio
unique_genres_users_after = df_users.explode('generi_preferiti')['generi_preferiti'].unique()
print("Numero di generi preferiti unici negli utenti dopo il filtraggio:", len(unique_genres_users_after))

# Salva il DataFrame degli utenti aggiornato
with open('df_user_filtered.pkl', 'wb') as file:
    pickle.dump(df_users, file)

