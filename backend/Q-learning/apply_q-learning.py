import numpy as np
import pickle
from qlearing import get_user_state,load_data
def recommend_book(user_id, Q_table, df_utenti, df_visual, df_ratings, available_books):
    # Ottieni lo stato dell'utente
    state = str(get_user_state(user_id, df_utenti, df_visual, df_ratings, available_books))
    
    # Se lo stato è nella Q-table, trova l'azione con il valore Q più alto
    if state in Q_table:
        # Ottieni il libro con il massimo valore Q per lo stato corrente
        best_action = max(Q_table[state], key=Q_table[state].get)
    else:
        # Se lo stato non è nella Q-table, scegli un libro a caso o secondo una politica di fallback
        best_action = np.random.choice(available_books)
    
    return best_action
def main():
    df_books,df_ratings,df_visualization,df_users = load_data()
    with open("../model/Q_learning.pkl", "rb") as f:
        Q_table = pickle.load(f)
    print("ci sono")

if __name__ == "__main__":
    main()