import pickle
import pandas as pd
import numpy as np


def create_visual():
    # Caricamento dei file pickle
    with open('PICKLE/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("PICKLE/df_user.pkl", "rb") as file:
        df_users = pickle.load(file)

    prob_pref = 0.7
    prob_other = 0.3
    num_views = 200

    simulations_df = simulate_views(df_users, df_book, prob_pref, prob_other, num_views)

    simulations_df = add_random_date(simulations_df)

    simulations_df.to_csv('CSV/visualization.csv', index=False)


def simulate_views(users, books, prob_preferiti, prob_altri, num_views):
    books['new_genres'] = books['new_genres'].apply(lambda g: eval(g) if isinstance(g, str) else g)
    users['generi_preferiti'] = users['generi_preferiti'].apply(lambda g: eval(g) if isinstance(g, str) else g)

    all_simulations = []


    for _, user in users.iterrows():
        user_generi_preferiti = user['generi_preferiti']

        libri_preferiti = books[books['genres'].apply(lambda g: any(genre in user_generi_preferiti for genre in g))].copy()

        if len(libri_preferiti) == 0:
            print(user_generi_preferiti)
            print(f"Nessun libro trovato per i generi preferiti dell'utente {user['id']}.")
            continue

        libri_preferiti['prob'] = prob_preferiti / len(libri_preferiti)
        altri_libri = books[~books['bookId'].isin(libri_preferiti['bookId'])].copy()
        altri_libri['prob'] = prob_altri / len(altri_libri)

        tutti_libri = pd.concat([libri_preferiti, altri_libri])

        tutti_libri['prob'] /= tutti_libri['prob'].sum()

        visualizzati = np.random.choice(tutti_libri['bookId'], size=num_views, p=tutti_libri['prob'])

        user_sim_df = pd.DataFrame({'userId': user['id'], 'bookId': visualizzati})
        all_simulations.append(user_sim_df)

    result_df = pd.concat(all_simulations, ignore_index=True)
    return result_df
from datetime import datetime, timedelta

def add_random_date(simulations_df, years=10, seed=42):

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    simulations_df['reading_date'] = simulations_df.apply(
        lambda _: (start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))).date(),
        axis=1
    )
    return simulations_df



