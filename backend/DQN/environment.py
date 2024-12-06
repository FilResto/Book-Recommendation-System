import numpy as np
import random
from collections import Counter
import pandas as pd
import datetime

class Environment:
    def __init__(self, users_df, books_df,visualization_df,ratings_df):
        """
        Inizializza l'ambiente.
        :param users_df: DataFrame degli utenti.
        :param books_df: DataFrame dei libri.
        """
        self.users_df = users_df
        self.books_df = books_df
        self.visualization_df = visualization_df
        self.ratings_df = ratings_df
        self.state = None
        self.current_user = None

    def get_reward(self,given_valuation):
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
    def caluclate_state(self):
    
        #AGE CALCULATION
        age = self.current_user['age']
        if age<25:
            age = "young"
        elif age>= 25 and age <=55:
            age = "adult"
        else:
            age = "old"

        #RECENT GENRE CALCULATION
        recent_visualizzazioni = (
        self.visualization_df[self.visualization_df['userId'] == self.current_user["userId"]]
        .sort_values(by='reading_date', ascending=False)
        .head(5)
    )
        recent_books = recent_visualizzazioni['bookId'].tolist()
        generi_count = Counter()
        for book_id in recent_books:
        # Trova il libro nel DataFrame dei libri
            book_info = self.books_df[self.books_df['bookId'] == book_id].iloc[0]
            # Supponiamo che i generi siano in una lista nella colonna 'new_genres'
            for genere in book_info['new_genres']:
                generi_count[genere] += 1

        if generi_count:
        # Ottieni il conteggio massimo
            max_count = max(generi_count.values())
        # Ottieni il primo genere con il conteggio massimo
            recent_genre = next(genere for genere, count in generi_count.items() if count == max_count)
        else:
            recent_genre = None  # Nessun libro visualizzato di recente

        avg_rating_user = self.ratings_df.loc[self.ratings_df['userId'] == self.current_user["userId"], 'rating'].mean()
        if avg_rating_user<=2:
            severity = "high"
        elif avg_rating_user>2 and avg_rating_user<= 3.5:
            severity = "medium"
        else:
            severity = "low"

        # Definisci lo stato iniziale basato sull'utente selezionato.
        state = {
            "age": age,  # young,adult,old
            "severity": severity,  # low,medium,high
            "preferred_genres": self.current_user["generi_preferiti"],  # Lista di generi separati da virgole
            "recent_genre": recent_genre
        }
        return state

    def reset(self):
        """
        Resetta l'ambiente per un nuovo episodio.
        :return: Stato iniziale.
        """
        # Seleziona un utente casuale dal DataFrame.
        
        self.current_user = self.users_df.sample().iloc[0]
        self.state = self.caluclate_state()
        return self.encode_state()

    def aggiorna_dati(self, book_id, rating):
    # Aggiungi una nuova visualizzazione per l'utente e il libro con la data corrente
        self.visualization_df = pd.concat([
        self.visualization_df,
        pd.DataFrame({"userId": [self.current_user["id"]], "bookId": [book_id], "reading_date": [datetime.now()]})
    ], ignore_index=True)

        # Aggiungi o aggiorna la valutazione dell'utente per il libro
        if ((self.ratings_df["userId"] == self.current_user["id"]) & (self.ratings_df["bookId"] == book_id)).any():
            # Aggiorna la valutazione esistente
            self.ratings_df.loc[(self.ratings_df["userId"] == self.current_user["id"]) & (self.ratings_df["bookId"] == book_id), "rating"] = rating
        else:
            # Aggiungi una nuova valutazione
            self.ratings_df = pd.concat([
                self.ratings_df,
                pd.DataFrame({"userId": [self.current_user["id"]], "bookId": [book_id], "rating": [rating]})
            ], ignore_index=True)

        


    def step(self, action):
        """
        Simula il passo successivo nel sistema dato un'azione.
        :param action: Azione intrapresa (indice del libro nel DataFrame).
        :return: Nuovo stato, ricompensa, flag done.
        """
        # Recupera il libro consigliato dall'azione (indice del DataFrame).
        recommended_book = self.books_df[self.books_df["bookId"] == action].iloc[0]

        # Simula il feedback dell'utente.
        if not self.visualization_df[self.visualization_df["bookId"] == action].empty:
            valuation = self.simulate_user_feedback(recommended_book)
            reward = self.get_reward(valuation)
            self.aggiorna_dati()
        else:
            reward = -10
        # Aggiorna lo stato (es. il genere recente diventa il genere del libro consigliato).
        self.state = self.caluclate_state()

        # Determina se l'episodio è terminato (può essere basato su un criterio come il numero di raccomandazioni).
        done = False  # Può essere cambiato se hai un limite di iterazioni.

        return self.encode_state(), reward, done
    
    def calculate_severity_deficit(severity,avg_val):
        if avg_val> 3:
            return 0.3
        else:
            return -0.5
    def simulate_user_feedback(self, bookid):
        """
        Simula il voto che un utente darebbe al libro consigliato.
        :param book: Riga del DataFrame (serie con informazioni sul libro).
        :return: Ricompensa calcolata in base al voto simulato.
        """
        user = self.current_user
        book = self.books_df.loc[self.books_df['bookId'] == bookid]

        base_rating = book['rating'].iloc[0] -1
        user_genres = user['generi_preferiti']
        book_genres = book['new_genres'].iloc[0]
        avg_rating_user = self.visualization_df.loc[self.visualization_df['userId'] == self.current_user["id"], 'rating'].mean()
        user_severity = self.calculate_severity_deficit(avg_rating_user)

        if len(set(user_genres).intersection(book_genres))==2:
            rating = np.random.normal(loc=base_rating + 1.5 + user_severity, scale=0.75)
        elif len(set(user_genres).intersection(book_genres))==1:
            rating = np.random.normal(loc=base_rating + 1 + user_severity, scale=0.75)
        else:
            rating = np.random.normal(loc=base_rating - 1.5 + user_severity, scale=1)

        return int(min(max(round(rating), 1), 5))

    def encode_state(self):
        """
        Converte lo stato in una rappresentazione numerica (es. one-hot encoding).
        :return: Stato codificato.
        """
        age_mapping = {"young": 0, "adult": 1, "old": 2}
        severity_mapping = {"low": 0, "medium": 1, "high": 2}

        # One-hot encoding per età e severità.
        age_encoded = np.eye(3)[age_mapping[self.state["age"]]]
        severity_encoded = np.eye(3)[severity_mapping[self.state["severity"]]]

        # Codifica i generi preferiti come una media delle categorie disponibili.
        all_genres = self.books_df["new_genres"].unique()
        genre_encoded = np.zeros(len(all_genres))
        for genre in self.state["preferred_genres"]:
            if genre in all_genres:
                genre_index = np.where(all_genres == genre)[0][0]
                genre_encoded[genre_index] = 1

        # Codifica il genere recente.
        recent_genre_encoded = np.zeros(len(all_genres))
        if self.state["recent_genre"] in all_genres:
            genre_index = np.where(all_genres == self.state["recent_genre"])[0][0]
            recent_genre_encoded[genre_index] = 1

        # Concatenazione finale dello stato codificato.
        return np.concatenate([age_encoded, severity_encoded, genre_encoded, recent_genre_encoded])
