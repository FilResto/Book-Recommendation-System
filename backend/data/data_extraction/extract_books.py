import pandas as pd
import pickle
import ast
def extract_userbook():
    df = pd.read_csv('CSV/libri_def.csv',usecols=['bookId', 'title', 'series', 'author', 'description', 'language','publishDate', 'genres', 'pages', 'awards', 'rating', 'price','new_genres'])
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['new_genres'] = df['new_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

  

    columns_to_check = ['bookId', 'title', 'series', 'author', 'description', 
                    'language', 'publishDate', 'genres', 'pages', 
                    'awards', 'rating', 'price','new_genres']

    # Filtra solo i libri in lingua inglese
    df_filtered = df[df['language'] == 'English']

    # Rimuovi le righe con valori NaN nelle colonne specificate
    df_filtered = df_filtered.dropna(subset=columns_to_check)

    # Rimuovi le righe con stringhe vuote nelle colonne specificate
    for column in columns_to_check:
        df_filtered = df_filtered[df_filtered[column] != '']

    with open('PICKLE/df_book.pkl', 'wb') as file:
        pickle.dump(df_filtered, file)



    df = pd.read_csv("CSV/users.csv")
    df['generi_preferiti'] = df['generi_preferiti'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    with open('PICKLE/df_user.pkl', 'wb') as file:
        pickle.dump(df, file)
def extract_visualization():
    df = pd.read_csv("CSV/visualization.csv")
    with open('PICKLE/df_visualization.pkl', 'wb') as file:
        pickle.dump(df, file)
def extract_ratings():
    df = pd.read_csv("CSV/ratings.csv")
    with open('PICKLE/df_ratings.pkl', 'wb') as file:
        pickle.dump(df, file)

