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
def create_item_user_matrix(df_ratings, df_book):
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
def calculate_item_similarity(item_user_matrix):
    # Sostituisci i NaN con 0 per calcolare la similarità
    item_user_matrix_filled = item_user_matrix.fillna(0)
    item_similarity = cosine_similarity(item_user_matrix_filled)
    # Crea un DataFrame per la similarità tra utenti
    item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)
    return item_similarity_df

def get_item_based_recommendations(user_id, item_user_matrix, item_similarity_df, df_visualizations, top_n=5):
    # Trova i libri già letti dall'utente
    books_read_by_user = df_visualizations[df_visualizations['userId'] == user_id]['bookId'].unique()
    
    # Raccolta delle raccomandazioni con punteggi di similarità
    recommendations = {}
    
    for book_id in books_read_by_user:
        # Trova i libri simili a quelli letti dall'utente, ordinati per similarità
        similar_books = item_similarity_df[book_id].sort_values(ascending=False).drop(book_id).index
        
        for similar_book in similar_books:
            # Se il libro non è stato ancora letto dall'utente, accumula il punteggio di similarità
            if similar_book not in books_read_by_user:
                if similar_book not in recommendations:
                    recommendations[similar_book] = item_similarity_df[book_id][similar_book]
                else:
                    recommendations[similar_book] += item_similarity_df[book_id][similar_book]
    
    # Ordina i libri raccomandati per punteggio e prendi i top_n
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    recommended_books = [book_id for book_id, _ in sorted_recommendations[:top_n]]
    
    return recommended_books
# Funzione principale
def main(user_id=None):
    # Carica i dati
    df_book, df_ratings, df_visualizations = load_data()
    
    # Crea la matrice utente-libro
    user_item_matrix = create_item_user_matrix(df_ratings, df_book)
    
    # Calcola la similarità tra utenti
    user_similarity_df = calculate_item_similarity(user_item_matrix)
    
    if user_id is not None:
        # Genera le raccomandazioni per l'utente specificato
        recommendations = get_item_based_recommendations(user_id, user_item_matrix, user_similarity_df, df_visualizations, top_n=3)
        print(f"Recommended books for user {user_id}: {recommendations}")
    else:
        print("Please provide a valid user_id.")
# Esegui il programma
if __name__ == "__main__":
    main(user_id=12)
