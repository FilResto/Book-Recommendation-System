import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Funzione per caricare i dati
def load_data():
    with open('../data/PICKLE/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("../data/PICKLE/df_visualization.pkl", "rb") as file:
        df_visualizations = pickle.load(file)
    with open("../data/PICKLE/df_ratings.pkl", "rb") as file:
        df_ratings = pickle.load(file)
    return df_book, df_ratings, df_visualizations


def calculate_user_similarity(user_item_matrix):
    
    # Sostituisci i NaN con 0 solo temporaneamente per il calcolo della distanza
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # Calcola la similarità ignorando i NaN
    similarity_matrix = 1 - pairwise_distances(user_item_matrix_filled, metric="cosine")
    user_similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_similarity_df

# 2. Funzione per creare la matrice utente-libro con tutte le combinazioni
def create_user_item_matrix(df_ratings, df_book):
    
    # Controlla i tipi di dato in `df_ratings` e `df_book`
    print(f"Tipi in df_ratings:\n{df_ratings.dtypes}")
    print(f"Tipi in df_book:\n{df_book.dtypes}")
    
    all_users = df_ratings['userId'].unique()
    all_books = df_book['bookId'].unique()
    print(f"Numero di utenti: {len(all_users)}, Numero di libri: {len(all_books)}")
    
    # Crea un DataFrame con tutte le combinazioni di utenti e libri
    all_combinations = pd.DataFrame([(user, book) for user in all_users for book in all_books], columns=['userId', 'bookId'])
    all_ratings = all_combinations.merge(df_ratings, on=['userId', 'bookId'], how='left')
    
    # Controlla se `rating` contiene NaN e verifica i tipi
    print(f"NaN in colonne dopo il merge:\n{all_ratings.isna().sum()}")
    print(f"Tipi in all_ratings:\n{all_ratings.dtypes}")
    
    # Pivot la tabella per avere gli utenti come righe e i libri come colonne
    user_item_matrix = all_ratings.pivot(index='userId', columns='bookId', values='rating')
    return user_item_matrix

# 3. Funzione per calcolare la similarità tra utenti
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

def create_user_item_matrix(df_ratings, df_book):
    all_users = df_ratings['userId'].unique()
    all_books = df_book['bookId'].unique()
    all_combinations = pd.DataFrame([(user, book) for user in all_users for book in all_books], columns=['userId', 'bookId'])
    all_ratings = all_combinations.merge(df_ratings, on=['userId', 'bookId'], how='left')
    
    # Pivot per creare la matrice utente-libro
    user_item_matrix = all_ratings.pivot(index='userId', columns='bookId', values='rating')
    
    # Normalizza sottraendo la media delle righe (utenti)
    user_item_matrix = user_item_matrix.subtract(user_item_matrix.mean(axis=1), axis=0)
    print(f"Matrice utente-libro normalizzata: {user_item_matrix.shape}")
    return user_item_matrix



def get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, df_visualizations, top_n=5):
    
    # Filtra gli utenti con similarità > 0
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    similar_users = similar_users[similar_users > 0]  # Mantieni solo utenti con similarità > 0
    print(f"Utenti simili trovati (dopo il filtro): {len(similar_users)}")
    
    books_read_by_user = df_visualizations[df_visualizations['userId'] == user_id]['bookId'].unique()
    books_to_recommend = [book for book in user_item_matrix.columns if book not in books_read_by_user]

    recommendations = {}
    for book in books_to_recommend:
        similar_users_ratings = user_item_matrix.loc[similar_users.index, book]
        weighted_ratings = similar_users * similar_users_ratings
        weighted_sum = weighted_ratings.sum()
        similarity_sum = similar_users[similar_users_ratings.notna()].sum()
        
        if similarity_sum > 0:
            recommendations[book] = weighted_sum / similarity_sum
    
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    recommended_books = [book_id for book_id, _ in sorted_recommendations[:top_n]]
    print(f"Raccomandazioni finali: {recommended_books}")
    return recommended_books


# Funzione principale
def cfu(user_id=None):
    # Carica i dati
    df_book, df_ratings, df_visualizations = load_data()
    
    # Crea la matrice utente-libro
    user_item_matrix = create_user_item_matrix(df_ratings, df_book)
    
    # Calcola la similarità tra utenti
    user_similarity_df = calculate_user_similarity(user_item_matrix)
    
    if user_id is not None:
        # Genera le raccomandazioni per l'utente specificato
        recommendations = get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, df_visualizations, top_n=3)
        print(f"Recommended books for user {user_id}: {recommendations}")
    else:
        print("Please provide a valid user_id.")

import numpy as np

def calculate_all_recommendations(user_item_matrix, user_similarity_df, top_n=5):
    
    # Riempie i NaN con 0 nella matrice utente-libro
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # Calcola il punteggio predetto per ogni utente-libro
    predicted_scores = np.dot(user_similarity_df.values, user_item_matrix_filled.values)
    
    # Crea una matrice per mascherare i libri già letti
    books_read_mask = user_item_matrix_filled > 0
    
    # Imposta i punteggi dei libri già letti a -inf per escluderli dalle raccomandazioni
    predicted_scores[books_read_mask.values] = -np.inf
    
    # Trova i migliori `top_n` libri per ogni utente
    top_recommendations = {
        user_id: user_item_matrix.columns[np.argsort(-predicted_scores[i, :])[:top_n]].tolist()
        for i, user_id in enumerate(user_item_matrix.index)
    }
    
    return top_recommendations

def cfu_all_users_optimized(top_n=3):
    # Carica i dati
    df_book, df_ratings, df_visualizations = load_data()
    
    # Crea la matrice utente-libro
    user_item_matrix = create_user_item_matrix(df_ratings, df_book)
    
    # Calcola la similarità tra utenti
    user_similarity_df = calculate_user_similarity(user_item_matrix)
    
    # Calcola le raccomandazioni per tutti gli utenti in parallelo
    all_recommendations = calculate_all_recommendations(user_item_matrix, user_similarity_df, top_n=top_n)
    
    # Stampa tutte le raccomandazioni
    print("\nRaccomandazioni per tutti gli utenti:")
    for user_id, rec_books in all_recommendations.items():
        print(f"Utente {user_id}: {rec_books}")
    
    return all_recommendations

def calculate_recommendations_for_user(user_id, user_item_matrix, user_similarity_df, top_n=5):

    # Controlla se l'utente esiste nella matrice
    if user_id not in user_item_matrix.index:
        print(f"L'utente {user_id} non esiste nei dati.")
        return []
    
    # Riempie i NaN con 0 nella matrice utente-libro
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # Calcola il punteggio predetto per ogni libro
    user_similarities = user_similarity_df.loc[user_id].values  # Similarità dell'utente con tutti gli altri
    predicted_scores = np.dot(user_similarities, user_item_matrix_filled.values)
    
    # Maschera i libri già letti
    books_read_mask = user_item_matrix_filled.loc[user_id] > 0
    predicted_scores[books_read_mask.values] = -np.inf  # Escludi libri già letti
    
    # Seleziona i migliori `top_n` libri
    recommended_books = user_item_matrix.columns[np.argsort(-predicted_scores)[:top_n]].tolist()
    return recommended_books

def cfu_single_user(user_id, top_n=5):
    
    # Carica i dati
    df_book, df_ratings, df_visualizations = load_data()
    
    # Crea la matrice utente-libro
    user_item_matrix = create_user_item_matrix(df_ratings, df_book)
    
    # Calcola la similarità tra utenti
    user_similarity_df = calculate_user_similarity(user_item_matrix)
    
    # Genera le raccomandazioni per l'utente specificato
    recommendations = calculate_recommendations_for_user(user_id, user_item_matrix, user_similarity_df, top_n=top_n)
    return recommendations

# Funzione principale
if __name__ == "__main__":
    user_id_to_recommend = 3  # Sostituisci con l'ID utente desiderato
    recommended_books = cfu_single_user(user_id_to_recommend, top_n=3)
    print(f"\nLibri raccomandati per l'utente {user_id_to_recommend}: {recommended_books}")


