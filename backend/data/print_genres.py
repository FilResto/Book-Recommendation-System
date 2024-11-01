import pickle
import pandas as pd
from collections import Counter

def remove_fiction_and_count_genres():
    # Carica il DataFrame dei libri filtrato
    with open('df_book_filtered.pkl', 'rb') as file:
        df_books = pickle.load(file)

    # Funzione per rimuovere "Fiction" dalla lista dei generi se presente
    def remove_fiction(genres):
        return [genre for genre in genres if genre != 'Fiction']

    # Applica la funzione di rimozione alla colonna 'genres'
    df_books['genres'] = df_books['genres'].apply(remove_fiction)

    # Salva il DataFrame aggiornato senza "Fiction" in un nuovo file
    with open('df_book_filtered.pkl', 'wb') as file:
        pickle.dump(df_books, file)

    # Esplode la colonna 'genres' per ottenere una lista piatta di generi e conta le occorrenze
    df_books_exploded = df_books.explode('genres')
    genre_counts = Counter(df_books_exploded['genres'])

    # Crea un DataFrame per visualizzare i conteggi dei generi aggiornati
    genre_counts_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False)

    # Stampa la lista completa dei generi con il rispettivo conteggio dopo aver rimosso "Fiction"
    print("Conteggio dei generi dopo la rimozione di 'Fiction':")
    print(genre_counts_df)

# Esegui la funzione
remove_fiction_and_count_genres()
