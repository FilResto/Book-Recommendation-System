import pandas as pd
import pickle
import ast
def extract_userbook():
    """
    df = pd.read_csv('CSV/libri_def.csv',usecols=['bookId', 'title', 'series', 'author', 'description', 'language','publishDate', 'genres', 'pages', 'awards', 'rating', 'price','new_genres'])
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['new_genres'] = df['new_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df2 = pd.read_csv('CSV/filtered_data.csv',usecols=['bookId', 'title', 'series', 'author', 'description', 'language','publishDate', 'genres', 'pages', 'awards', 'rating', 'price','new_genres'])
    df2['genres'] = df2['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df2['new_genres'] = df2['new_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    merged_df = pd.concat([df, df2]).drop_duplicates(subset='bookId').reset_index(drop=True)
    """
    df = pd.read_csv('CSV/books_def.csv',usecols=['bookId', 'title', 'series', 'author', 'description', 'language','publishDate', 'genres', 'pages', 'awards', 'rating', 'price','new_genres'])
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['new_genres'] = df['new_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


    print(len(df))
    with open('PICKLE/df_book.pkl', 'wb') as file:
        pickle.dump(df, file)  



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

