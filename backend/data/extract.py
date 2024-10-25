import pandas as pd

# Step 1: Carica il file CSV e seleziona solo le colonne desiderate
df = pd.read_csv('booksAll.csv')

# Filtra le colonne che ti interessano
columns_to_keep = ['bookId', 'title', 'series', 'author', 'description', 'language', 'genres', 'pages', 'awards', 'bbeVotes', 'price']
filtered_df = df[columns_to_keep]

# Step 2: Verifica il contenuto del nuovo dataframe
print(filtered_df.head())  # Stampa le prime righe del dataframe

