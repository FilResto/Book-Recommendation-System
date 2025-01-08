import numpy as np
import random
from collections import Counter
import pandas as pd
from datetime import datetime

class Environment:
    def __init__(self, users_df, books_df,visualization_df,ratings_df,max_steps=10, patience=3):
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
        self.max_steps = max_steps
        self.patience = patience
        self.current_step = 0
        self.no_feedback_steps = 0
        self.recommendation_counts = {book_id: 0 for book_id in self.books_df['bookId']}

    def user_recom(self,user_id):
        u = {}
        user = self.users_df[self.users_df["id"] == user_id].iloc[0]
        u["age"] = user["age"]
        u["id"] = user["id"]
        u["generi_preferiti"] = user["generi_preferiti"]
        self.current_user = u
        self.state = self.caluclate_state()
        
        
        return self.encode_state()

    def get_reward(self,given_valuation,book_id,genre_recomendation):
        reward = 0
        if genre_recomendation == 2:
            reward += 80
        if genre_recomendation == 1:
            reward += 40
        if genre_recomendation == 0:
            reward += -10
        if(given_valuation == 5):
            reward += 5
        if(given_valuation == 4):
            reward += 3
        if(given_valuation == 3):
            reward += 1
        if(given_valuation==2):
            reward += -3
        if(given_valuation == 1):
            reward += -5
         # Exploration bonus
        count = self.recommendation_counts.get(book_id, 0)
        exploration_bonus = 1 / (np.sqrt(count + 1))  # Bonus decrescente con il numero di raccomandazioni
        reward += exploration_bonus  # Somma il bonus alla ricompensa
        
        return reward
    def caluclate_state(self):
    
        #AGE CALCULATION
        
        age = int(self.current_user['age'])
        #age = self.current_user['age']
        if age<25:
            age = "young"
        elif age>= 25 and age <=55:
            age = "adult"
        else:
            age = "old"
        #RECENT GENRE CALCULATION
        recent_visualizzazioni = (
        self.visualization_df[self.visualization_df["userId"] == self.current_user["id"]]
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

        avg_rating_user = self.ratings_df.loc[self.ratings_df['userId'] == self.current_user["id"], 'rating'].mean()
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
            #"recent_genre": recent_genre
        }
        return state

    def reset(self):
        """
        Resetta l'ambiente per un nuovo episodio.
        :return: Stato iniziale.
        """
        # Seleziona un utente casuale dal DataFrame.
        self.current_step = 0
        self.no_feedback_steps = 0
        self.current_user = self.users_df.sample().iloc[0]
        self.state = self.caluclate_state()
        #print(self.state)
        return self.encode_state()

    def aggiorna_dati(self, book_id, rating):
    # Aggiungi una nuova visualizzazione per l'utente e il libro con la data corrente
        self.visualization_df = pd.concat([
        self.visualization_df,
        pd.DataFrame({"userId": [self.current_user["id"]], "bookId": [book_id], "reading_date": [datetime.now().strftime("%Y-%m-%d")]})
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
        recommended_book = self.books_df.iloc[action]
        book_id = recommended_book["bookId"]
        self.recommendation_counts[book_id] += 1
        # Simula il feedback dell'utente.
        #if not self.visualization_df[self.visualization_df["bookId"] == book_id].empty:
        self.current_step += 1
        valuation,genre_recomandation = self.simulate_user_feedback(book_id)
        reward = self.get_reward(valuation,book_id,genre_recomandation)
        self.aggiorna_dati(book_id,valuation)
        #else:
            #print("entrato")
            #reward = -5

        if reward < 1:  # Soglia per feedback negativo
            self.no_feedback_steps += 1
        else:
            self.no_feedback_steps = 0
        # Aggiorna lo stato (es. il genere recente diventa il genere del libro consigliato).
        self.state = self.caluclate_state()

        # Determina se l'episodio è terminato (può essere basato su un criterio come il numero di raccomandazioni).
        done = (
            self.current_step >= self.max_steps or 
            self.no_feedback_steps >= self.patience
        )
        #print(self.state)
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
        #print(bookid)
        base_rating = book['rating'].iloc[0]
        user_genres = user['generi_preferiti']
        book_genres = book['new_genres'].iloc[0]
        #print(book_genres,base_rating)
        avg_rating_user = self.ratings_df.loc[self.ratings_df['userId'] == self.current_user["id"], 'rating'].mean()
        user_severity = self.calculate_severity_deficit(avg_rating_user)
        genre_recomendation = 0
        if len(set(user_genres).intersection(book_genres))==2:
            #print("\n\n\n sono entrato qui2 \n\n\n")
            rating = np.random.normal(loc=base_rating + 0.75 + user_severity, scale=0.1)
            genre_recomendation = 2
        elif len(set(user_genres).intersection(book_genres))==1:
            rating = np.random.normal(loc=base_rating  + user_severity+ 0.5, scale=0.3)
            genre_recomendation = 1
            #print("\n\n\n sono entrato qui\n\n\n")
        else:
            rating = np.random.normal(loc=base_rating - 0.3 + user_severity, scale=0.5)
        #print(rating)
        return int(min(max(round(rating), 1), 5)),genre_recomendation

    def encode_state(self):
        """
        Converte lo stato in una rappresentazione numerica (es. one-hot encoding).
        :return: Stato codificato.
        """
        #age_mapping = {"young": 0, "adult": 1, "old": 2}
        severity_mapping = {"low": 0, "medium": 1, "high": 2}

        # One-hot encoding per età e severità.
        #age_encoded = np.eye(3)[age_mapping[self.state["age"]]]
        severity_encoded = np.eye(3)[severity_mapping[self.state["severity"]]]

        all_genres = set(genre for genres in self.books_df["new_genres"] for genre in genres)
        all_genres = sorted(all_genres)  # Ordine deterministico
        genre_to_index = {genre: idx for idx, genre in enumerate(all_genres)}
        # Codifica i generi come un array binario
        fav_genre_encoded = np.zeros(len(all_genres))
        for genre in self.state["preferred_genres"]:
            genre_index = genre_to_index[genre]
            fav_genre_encoded[genre_index] = 1

        # Codifica il genere recente.
        #recent_genre_encoded = np.zeros(len(all_genres))
        #if self.state["recent_genre"] in all_genres:
        #    genre_index = np.where(all_genres == self.state["recent_genre"])[0][0]
        #    recent_genre_encoded[genre_index] = 1

        # Concatenazione finale dello stato codificato.
        #return np.concatenate([age_encoded, severity_encoded, genre_encoded, recent_genre_encoded])
        #return np.concatenate([age_encoded, severity_encoded, fav_genre_encoded])
        return np.concatenate([severity_encoded, fav_genre_encoded])
