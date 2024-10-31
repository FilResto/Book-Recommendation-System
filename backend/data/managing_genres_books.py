import pickle
import pandas as pd
from collections import Counter

# Carica il DataFrame dal file .pkl
with open('df_book.pkl', 'rb') as file:
    df_books = pickle.load(file)

# Assicurati che la colonna 'genres' sia una lista e "esplodi" la colonna per ottenere i singoli generi
df_books_exploded = df_books.explode('genres')

# Conta la frequenza di ciascun genere e crea una lista dei top 100 generi
genre_counts = Counter(df_books_exploded['genres'])
top_100_genres = [genre for genre, _ in genre_counts.most_common(100)]

# Funzione per filtrare i generi di un libro mantenendo solo quelli presenti nella top 100
def filter_genres(genres):
    return [genre for genre in genres if genre in top_100_genres]

# Applica la funzione di filtro alla colonna 'genres' del DataFrame originale
df_books['genres'] = df_books['genres'].apply(filter_genres)

# Salva il DataFrame aggiornato nel file .pkl
with open('df_book_filtered.pkl', 'wb') as file:
    pickle.dump(df_books, file)

# Explode la colonna 'genres' per ottenere una lista dei generi unici
unique_genres = df_books.explode('genres')['genres'].unique()

# Stampa la lista dei generi unici
#print("Lista dei generi unici nel DataFrame aggiornato:", unique_genres)

# Stampa il numero di generi unici
#print("Numero unico di generi nel DataFrame aggiornato:", len(unique_genres))

# Stampa l'head del DataFrame aggiornato
#print(df_books.head())


