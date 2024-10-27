import pandas as pd
import pickle
def extract_userbook():
    df = pd.read_csv('booksAll.csv',usecols=['bookId', 'title', 'series', 'author', 'description', 'language','publishDate', 'genres', 'pages', 'awards', 'bbeVotes', 'price'])
    with open('df_book.pkl', 'wb') as file:
        pickle.dump(df, file)
    df = pd.read_csv("users.csv")
    with open('df_user.pkl', 'wb') as file:
        pickle.dump(df, file)
def extract_visualization():
    df = pd.read_csv("visualization.csv")
    with open('df_visualization.pkl', 'wb') as file:
        pickle.dump(df, file)
def extract_ratings():
    df = pd.read_csv("ratings.csv")
    with open('df_ratings.pkl', 'wb') as file:
        pickle.dump(df, file)
