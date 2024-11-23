import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Funzione per caricare i dati
def load_data():
    with open('../data/PICKLE/df_book.pkl', 'rb') as file:
        df_book = pickle.load(file)
    with open("../data/PICKLE/df_user.pkl", "rb") as file:
        df_users = pickle.load(file)
    with open("../data/PICKLE/df_visualization.pkl", "rb") as file:
        df_visualizations = pickle.load(file)
    with open("../data/PICKLE/df_ratings.pkl","rb") as file:
        df_ratings = pickle.load(file)
    return df_book, df_users, df_visualizations

# 2. Funzione per creare il profilo dei libri basato sui generi
def create_book_profiles(df_book):
    # Convert genres column to list format
    mlb = MultiLabelBinarizer()
    books_genres_encoded = mlb.fit_transform(df_book['new_genres'])
    books_genres_df = pd.DataFrame(books_genres_encoded, columns=mlb.classes_)
    books_with_genres_profile = pd.concat([df_book[['bookId', 'title']], books_genres_df], axis=1)
    #print(books_with_genres_profile)
    return books_with_genres_profile, mlb

# 3. Funzione per creare il profilo degli utenti basato sui generi preferiti
def create_user_profile(df_users, mlb, user_id):
    # Filtra i dati per l'utente specifico
    user_data = df_users[df_users['id'] == user_id].copy()  # Usa .copy() per evitare il warning
    
    # Verifica se l'utente esiste nei dati
    if user_data.empty:
        print(f"User {user_id} not found.")
        return None
    
    # Trasforma i generi preferiti in formato lista utilizzando .loc per evitare il warning
    #user_data.loc[:, 'generi_preferiti'] = user_data['generi_preferiti'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    #print(user_data["generi_preferiti"])
    # Codifica i generi preferiti dell'utente
    user_genres_encoded = mlb.transform(user_data['generi_preferiti'])
    user_profile = pd.DataFrame(user_genres_encoded, columns=mlb.classes_)
    
    # Combina le informazioni dell'utente con i generi codificati
    user_profile = pd.concat([user_data[['id', 'age']].reset_index(drop=True), user_profile], axis=1)
    
    return user_profile



# 4. Funzione per calcolare la similarità escludendo i libri già letti
def calculate_similarity_for_user(user_profile, books_with_genres_profile, df_visualizations, user_id):
    # Filtra i libri già letti dall'utente
    books_read_by_user = df_visualizations[df_visualizations['userId'] == user_id]['bookId'].tolist()
    books_not_read = books_with_genres_profile[~books_with_genres_profile['bookId'].isin(books_read_by_user)]
    
    # Calcola la similarità solo con i libri non letti
    user_genres_vector = user_profile.drop(['id', 'age']).values.reshape(1, -1)
    book_genres_matrix = books_not_read.drop(columns=['bookId', 'title']).values
    similarity_scores = cosine_similarity(user_genres_vector, book_genres_matrix).flatten()
    similarity_series = pd.Series(similarity_scores, index=books_not_read['title'])
    
    return similarity_series

def get_top_recommendations_for_user(similarity_series, top_n=2):
    top_books = similarity_series.sort_values(ascending=False).head(top_n)
    return list(top_books.index)

# Funzione principale per eseguire tutto
def cbf(user_id=None):
    # Carica i dati
    df_book, df_users, df_visualizations = load_data()
    
    # Crea i profili dei libri e degli utenti
    books_with_genres_profile, mlb = create_book_profiles(df_book) 
    #users_with_genres_profile = create_user_profile(df_users, mlb)
    users_with_genres_profile = create_user_profile(df_users, mlb,user_id)
    

    if user_id is not None:
        # Filtra il profilo dell'utente specifico
        user_profile = users_with_genres_profile[users_with_genres_profile['id'] == user_id]
        if user_profile.empty:
            print(f"User {user_id} not found.")
            return
        
        # Calcola la similarità per l'utente specifico e ottieni le raccomandazioni
        similarity_series = calculate_similarity_for_user(user_profile.iloc[0], books_with_genres_profile, df_visualizations, user_id)
        recommendations = get_top_recommendations_for_user(similarity_series, top_n=3)
        
        # Stampa le raccomandazioni per l'utente specifico
        print(f"User {user_id} recommendations: {recommendations}")
    else:
        print("Please provide a user_id.")
    # Stampa le raccomandazioni
    #for user_id, rec_books in recommendations.items():
    #   print(f"User {user_id} recommendations: {rec_books}")

# Esegui il programma
if __name__ == "__main__":
    cbf(user_id = 56)