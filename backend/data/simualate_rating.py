import pickle
import pandas as pd
import numpy as np

def create_ratings():
    # Caricamento dei file pickle
    with open('df_visualization.pkl', 'rb') as file:
        df_readings = pickle.load(file)

    prob_review = 0.05

    simulate_ratings = generate_ratings(df_readings,prob_review)

    simulate_ratings = simulate_ratings.drop('reading_date',axis = 1)
    
    simulate_ratings.to_csv('ratings.csv', index=False)


def generate_ratings(visualizations_df, prob_vote):
    
    
    voted_mask = np.random.rand(len(visualizations_df)) < prob_vote
    ratings_df = visualizations_df[voted_mask].copy()


    ratings_df['rating'] = np.random.randint(1, 6, size=len(ratings_df))

    return ratings_df


