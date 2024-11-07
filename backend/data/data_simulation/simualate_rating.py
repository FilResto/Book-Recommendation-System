import pickle
import pandas as pd
import numpy as np
pref = 0
npref = 0
def incrementa_pref():
    global pref
    pref+=1
def stampa_pref():
    print(pref)
def incrementa_npref():
    global npref
    npref+=1
def stampa_npref():
    print(npref)

def create_ratings():
    # Caricamento dei file pickle
    with open('PICKLE/df_visualization.pkl', 'rb') as file:
        df_readings = pickle.load(file)
    with open('PICKLE/df_user.pkl', 'rb') as file:
        df_users = pickle.load(file)
    with open('PICKLE/df_book.pkl', 'rb') as file:
        df_books = pickle.load(file)
    prob_review = 0.05
    simulate_ratings = generate_ratings(df_readings,prob_review,df_users,df_books)

    simulate_ratings = simulate_ratings.drop('reading_date',axis = 1)
    
    simulate_ratings.to_csv('CSV/ratings.csv', index=False)


def generate_ratings(visualizations_df, prob_vote,df_users,df_books):
    
    
    voted_mask = np.random.rand(len(visualizations_df)) < prob_vote
    ratings_df = visualizations_df[voted_mask].copy()


    ratings_df['rating'] = ratings_df.apply(
        lambda row: simualate_valuation(df_users, df_books, row['userId'], row['bookId']),
        axis=1
    )
    stampa_pref()
    stampa_npref()
    return ratings_df

def simualate_valuation(df_users, df_books, user_id, book_id):
    user = df_users.loc[df_users['id'] == user_id]
    book = df_books.loc[df_books['bookId'] == book_id]

    # Riduci la base di rating per rendere più probabili i voti bassi
    base_rating = book['rating'].iloc[0] -1 # Abbassato per aumentare i voti bassi

    user_genres = (user['generi_preferiti'].iloc[0])
    book_genres = (book['new_genres'].iloc[0])
    
    # Aumenta la variabilità per generare più voti estremi
    if len(set(user_genres).intersection(book_genres))==2:
        rating = np.random.normal(loc=base_rating + 1.5 , scale=0.75)
    elif len(set(user_genres).intersection(book_genres))==1:
        rating = np.random.normal(loc=base_rating + 1 , scale=0.75)
    else:
        rating = np.random.normal(loc=base_rating - 1.5 , scale=1)


    # Restituisci un voto assicurando che non scenda sotto 1
    return int(min(max(round(rating), 1), 5))

