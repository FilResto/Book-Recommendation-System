import pickle
import pandas as pd

# Carica il DataFrame dei libri filtrati per ottenere i generi più comuni
with open('df_book_filtered.pkl', 'rb') as file:
    df_books = pickle.load(file)

# Ottieni la lista unica dei generi presenti nel DataFrame dei libri filtrato
top_genres = df_books.explode('genres')['genres'].unique()

# Carica il DataFrame degli utenti
with open('df_user.pkl', 'rb') as file:
    df_users = pickle.load(file)

# Stampa la lista dei generi unici preferiti dagli utenti PRIMA del filtraggio
unique_genres_before = df_users.explode('generi_preferiti')['generi_preferiti'].unique()
print("Generi preferiti unici prima del filtraggio:", unique_genres_before)
print("Numero di generi preferiti unici prima del filtraggio:", len(unique_genres_before))

# Funzione per filtrare i generi preferiti di un utente
def filter_user_genres(preferred_genres):
    return [genre for genre in preferred_genres if genre in top_genres]

# Applica il filtro alla colonna 'generi_preferiti' del DataFrame degli utenti
df_users['generi_preferiti'] = df_users['generi_preferiti'].apply(filter_user_genres)

# Stampa la lista dei generi unici preferiti dagli utenti DOPO il filtraggio
unique_genres_after = df_users.explode('generi_preferiti')['generi_preferiti'].unique()
#print("Generi preferiti unici dopo il filtraggio:", unique_genres_after)
#print("Numero di generi preferiti unici dopo il filtraggio:", len(unique_genres_after))

# Salva il DataFrame degli utenti aggiornato
with open('df_user_filtered.pkl', 'wb') as file:
    pickle.dump(df_users, file)

# Stampa l'head del DataFrame degli utenti aggiornato per verificare
#print("Anteprima del DataFrame utenti aggiornato:")
#print(df_users.head())
