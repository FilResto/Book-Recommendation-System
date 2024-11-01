from transformers import pipeline
from collections import Counter
from google.colab import files

import pickle


def assign_genres(description):
    result = classifier(description, genres, multi_label=True)

    # Ordina e prendi i due generi con la probabilità più alta
    top_genres = sorted(result['labels'], key=lambda x: -result['scores'][result['labels'].index(x)])[:2]
    return top_genres
# Crea una pipeline di classificazione di testo
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", multi_label=True,device = 0)

with open('/content/df_book.pkl', 'rb') as file:
    df_books = pickle.load(file)


# Assicurati che la colonna 'genres' sia una lista e "esplodi" la colonna per ottenere i singoli generi
df_books_exploded = df_books.explode('genres')

# Conta la frequenza di ciascun genere e crea una lista dei top 50 generi
genre_counts = Counter(df_books_exploded['genres'])
genres = [genre for genre, _ in genre_counts.most_common(100)]



"""# Esegui in batch
batch_size = 32  # Adatta batch_size in base alla RAM
for i in range(0, len(df_books), batch_size):
    df_books.loc[i:i+batch_size-1, 'genres'] = assign_genres(df_books['description'][i:i+batch_size].tolist())
"""
df_books['genres'] =df_books['description'].apply(assign_genres)

# Salva il DataFrame aggiornato in un nuovo pickle
df_books.to_pickle("/content/libri_aggiornato.pkl")
files.download("/content/libri_aggiornato.pkl")