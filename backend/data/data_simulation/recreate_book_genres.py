from transformers import pipeline
from collections import Counter
import pickle
import pandas as pd
import json
import time

def assign_genres_batch(descriptions):
    results = classifier(descriptions, genres, multi_label=True)
    return [
        sorted(res['labels'], key=lambda x: -res['scores'][res['labels'].index(x)])[:2] 
        for res in results
    ]

# Crea la pipeline di classificazione di testo
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", multi_label=True, device=0)

# Carica il DataFrame dei libri
with open('/content/df_english_books.pkl', 'rb') as file:
    df_books = pickle.load(file)

# Genera la lista dei generi più frequenti
df_books_exploded = df_books.explode('genres')
genre_counts = Counter(df_books_exploded['genres'])
genres = [genre for genre, _ in genre_counts.most_common(50)]
df_books['description'] = df_books['description'].fillna("")
# Numero di righe per intervallo
step_size = 256
for i in range(0, len(df_books), step_size):
    inizio = time.time()
    end_idx = min(i + step_size, len(df_books))
    
    # Elaborazione 
    descriptions_batch = df_books.loc[i:end_idx-1, 'description'].tolist()
    a = assign_genres_batch(descriptions_batch)
    a = [json.dumps(genres) for genres in a]  # Unisci i generi in una stringa
    df_books.loc[i:end_idx-1, 'new_genres'] = a
    print(time.time()-inizio)
    # Salva il progresso ogni step_size righe
    df_books.to_pickle(f"/content/drive/MyDrive/Colab Notebooks/Pickle Container/libri_aggiornato_{i}-{end_idx}.pkl")
    print(f"Salvati i dati da {i} a {end_idx}")

# Salva il file finale
df_books.to_pickle("/content/drive/MyDrive/Colab Notebooks/libri_aggiornato.pkl")
print("Processo completato e file salvati.")
