import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Funzione per caricare i dati
def load_data():
    with open('../data/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("../data/df_visualization.pkl", "rb") as file:
        df_visualizations = pickle.load(file)
    with open("../data/df_ratings.pkl", "rb") as file:
        df_ratings = pickle.load(file)
    with open("../data/df_user.pkl", "rb") as file:
        df_users = pickle.load(file)
    return df_book, df_ratings, df_visualizations,df_users