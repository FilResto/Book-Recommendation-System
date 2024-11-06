import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Funzione per caricare i dati
def load_data():
    with open('../data/PICKLE/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("../data/PICKLE/df_visualization.pkl", "rb") as file:
        df_visualizations = pickle.load(file)
    with open("../data/PICKLE/df_ratings.pkl", "rb") as file:
        df_ratings = pickle.load(file)
    with open("../data/PICKLE/df_user.pkl", "rb") as file:
        df_users = pickle.load(file)
    return df_book, df_ratings, df_visualizations,df_users


def get_reward(given_valuation):
    if(given_valuation == 5):
        reward = 4
    if(given_valuation == 4):
        reward = 3
    if(given_valuation == 3):
        reward = 1
    if(given_valuation==2):
        reward = -3
    if(given_valuation == 1):
        reward = -5
    return reward

def calculate_severity(avg_val):
    if avg_val> 3:
        return 0.3
    else:
        return -0.5

def simualate_valuation(df_users,df_books,df_valuation,user_id, book_id):
    user = df_users.loc[df_users['id'] == user_id]
    book = df_books.loc[df_books['bookId'] == book_id]

    base_rating = book['rating'].iloc[0] -1
    
    user_genres = user['generi_preferiti'].iloc[0]
    book_genres = book['genres'].iloc[0]
    avg_rating_user = df_valuation.loc[df_valuation['userId'] == user_id, 'rating'].mean()
    user_severity = calculate_severity(avg_rating_user)

    if set(user_genres).intersection(book_genres):
        rating = np.random.normal(loc=base_rating + 0.75 + user_severity, scale=0.75)
    else:
        rating = np.random.normal(loc=base_rating - 1.5 + user_severity, scale=1)

    return int(min(max(round(rating), 1), 5))



def main():
    df_books,df_ratings,df_visualization,df_users = load_data()
    simualate_valuation(df_users,df_books,df_ratings,1,"2767052-the-hunger-games")
    return

if __name__ == "__main__":
    main()