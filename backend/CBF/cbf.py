import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Funzione per caricare i dati
def load_data():
    with open('../data/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("../data/df_user.pkl", "rb") as file:
        df_users = pickle.load(file)
    with open("../data/df_visualization.pkl", "rb") as file:
        df_visualizations = pickle.load(file)
    with open("../data/df_ratings.pkl","rb") as file:
        df_ratings = pickle.load(file)
    return df_book, df_users, df_visualizations

# 2. Funzione per creare il profilo dei libri basato sui generi
def create_book_profiles(df_book):
    # Convert genres column to list format
    df_book['genres'] = df_book['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    books_genres_encoded = mlb.fit_transform(df_book['genres'])
    books_genres_df = pd.DataFrame(books_genres_encoded, columns=mlb.classes_)
    books_with_genres_profile = pd.concat([df_book[['bookId', 'title']], books_genres_df], axis=1)
    return books_with_genres_profile, mlb

# 3. Funzione per creare il profilo degli utenti basato sui generi preferiti
def create_user_profiles(df_users, mlb):
    df_users['generi_preferiti'] = df_users['generi_preferiti'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    users_genres_encoded = mlb.transform(df_users['generi_preferiti'])
    users_genres_df = pd.DataFrame(users_genres_encoded, columns=mlb.classes_)
    users_with_genres_profile = pd.concat([df_users[['id', 'age']], users_genres_df], axis=1)
    return users_with_genres_profile

# 4. Funzione per calcolare la similarità e generare raccomandazioni
def calculate_similarity(users_with_genres_profile, books_with_genres_profile):
    user_genres_matrix = users_with_genres_profile.drop(columns=['id', 'age']).values
    book_genres_matrix = books_with_genres_profile.drop(columns=['bookId', 'title']).values
    similarity_matrix = cosine_similarity(user_genres_matrix, book_genres_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=users_with_genres_profile['id'], columns=books_with_genres_profile['title'])
    return similarity_df

def get_top_recommendations(similarity_df, top_n=2):
    recommendations = {}
    for user_id in similarity_df.index:
        top_books = similarity_df.loc[user_id].sort_values(ascending=False).head(top_n)
        recommendations[user_id] = list(top_books.index)  # Get book titles
    return recommendations

# Funzione principale per eseguire tutto
def main():
    # Carica i dati
    df_book, df_users, df_visualizations = load_data()
    
    # Crea i profili dei libri e degli utenti
    books_with_genres_profile, mlb = create_book_profiles(df_book)
    users_with_genres_profile = create_user_profiles(df_users, mlb)
    
    # Calcola la similarità e ottieni le raccomandazioni
    similarity_df = calculate_similarity(users_with_genres_profile, books_with_genres_profile)
    recommendations = get_top_recommendations(similarity_df, top_n=3)
    
    # Stampa le raccomandazioni
    for user_id, rec_books in recommendations.items():
        print(f"User {user_id} recommendations: {rec_books}")

# Esegui il programma
if __name__ == "__main__":
    main()