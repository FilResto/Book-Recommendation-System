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
    return df_book, df_ratings, df_visualizations

# 2. Funzione per creare la matrice utente-libro
def create_user_item_matrix(df_ratings):
    # Pivot la tabella per avere gli utenti come righe e i libri come colonne
    user_item_matrix = df_ratings.pivot(index='userId', columns='bookId', values='rating')
    return user_item_matrix

# 3. Funzione per calcolare la similarità tra utenti
def calculate_user_similarity(user_item_matrix):
    # Sostituisci i NaN con 0 per calcolare la similarità
    user_item_matrix_filled = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_matrix_filled)
    # Crea un DataFrame per la similarità tra utenti
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_similarity_df

# 4. Funzione per generare raccomandazioni per un singolo utente
def get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, top_n=5):
    # Trova gli utenti simili
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    
    # Prendi i libri non valutati dall'utente
    user_ratings = user_item_matrix.loc[user_id]
    books_not_rated = user_ratings[user_ratings.isna()].index
    
    # Calcola un punteggio ponderato per ciascun libro non valutato dall'utente
    recommendations = {}
    for book in books_not_rated:
        # Filtra le valutazioni per il libro corrente e per gli utenti simili
        similar_users_ratings = user_item_matrix.loc[similar_users.index, book]
        
        # Pondera le valutazioni degli utenti simili con i loro coefficienti di similarità
        weighted_ratings = similar_users * similar_users_ratings
        weighted_sum = weighted_ratings.sum()
        similarity_sum = similar_users[similar_users_ratings.notna()].sum()
        
        # Evita divisione per zero
        if similarity_sum > 0:
            recommendations[book] = weighted_sum / similarity_sum
    
    # Ordina i libri per punteggio e prendi i primi `top_n`
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    recommended_books = [book_id for book_id, _ in sorted_recommendations[:top_n]]
    
    return recommended_books

# Funzione principale
def main(user_id=None):
    # Carica i dati
    df_book, df_ratings, df_visualizations = load_data()
    
    # Crea la matrice utente-libro
    user_item_matrix = create_user_item_matrix(df_ratings)
    
    # Calcola la similarità tra utenti
    user_similarity_df = calculate_user_similarity(user_item_matrix)
    
    if user_id is not None:
        # Genera le raccomandazioni per l'utente specificato
        recommendations = get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, top_n=3)
        print(f"Recommended books for user {user_id}: {recommendations}")
    else:
        print("Please provide a valid user_id.")
# Esegui il programma
if __name__ == "__main__":
    main(user_id=12)