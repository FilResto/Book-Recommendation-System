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

# 2. Funzione per creare la matrice utente-libro con tutte le combinazioni
def create_user_item_matrix(df_ratings, df_book):
    # Estrai tutti gli utenti e tutti i libri
    all_users = df_ratings['userId'].unique()
    all_books = df_book['bookId'].unique()
    
    # Crea un DataFrame con tutte le combinazioni di utenti e libri
    all_combinations = pd.DataFrame([(user, book) for user in all_users for book in all_books], columns=['userId', 'bookId'])
    
    # Unisci il dataframe delle valutazioni con tutte le combinazioni
    all_ratings = all_combinations.merge(df_ratings, on=['userId', 'bookId'], how='left')
    
    # Pivot la tabella per avere gli utenti come righe e i libri come colonne
    item_user_matrix = all_ratings.pivot(index='bookId', columns='userId', values='rating')
    #print(user_item_matrix)
    return item_user_matrix

# 3. Funzione per calcolare la similarità tra utenti
def calculate_user_similarity(item_user_matrix):
    # Sostituisci i NaN con 0 per calcolare la similarità
    item_user_matrix_filled = item_user_matrix.fillna(0)
    item_similarity = cosine_similarity(item_user_matrix)
    # Crea un DataFrame per la similarità tra utenti
    user_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)
    return user_similarity_df

def get_item_based_recommendations(item_id, item_user_matrix, item_similarity_df, df_visualizations, top_n=5):
    # Trova gli utenti simili
    similar_items = item_similarity_df[item_id].sort_values(ascending=False).drop(item_id)
    
    # Trova i libri già letti dall'utente utilizzando df_visualizations
    books_read_by_user = df_visualizations[df_visualizations['item_id'] == item_id]['userId'].unique()
    
    # Prendi i libri che non sono stati letti dall'utente
    books_to_recommend = [book for book in item_user_matrix.columns if book not in books_read_by_user]
    
    # Calcola un punteggio ponderato per ciascun libro non letto dall'utente
    recommendations = {}
    for book in books_to_recommend:
        # Filtra le valutazioni per il libro corrente e per gli utenti simili
        similar_users_ratings = item_user_matrix.loc[similar_items.index, book]
        
        # Pondera le valutazioni degli utenti simili con i loro coefficienti di similarità
        weighted_ratings = similar_items * similar_users_ratings
        weighted_sum = weighted_ratings.sum()
        similarity_sum = similar_items[similar_users_ratings.notna()].sum()
        
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
    user_item_matrix = create_user_item_matrix(df_ratings, df_book)
    
    # Calcola la similarità tra utenti
    user_similarity_df = calculate_user_similarity(user_item_matrix)
    
    if user_id is not None:
        # Genera le raccomandazioni per l'utente specificato
        recommendations = get_item_based_recommendations(user_id, user_item_matrix, user_similarity_df, df_visualizations, top_n=3)
        print(f"Recommended books for user {user_id}: {recommendations}")
    else:
        print("Please provide a valid user_id.")
# Esegui il programma
if __name__ == "__main__":
    main(user_id=12)
