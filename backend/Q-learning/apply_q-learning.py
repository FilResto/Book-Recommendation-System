import numpy as np
import pickle
from qlearing import get_user_state,load_data
def recommend_book(user_id, Q_table, df_utenti, df_visual, df_ratings, available_books):
    # Ottieni lo stato dell'utente
    state = str(get_user_state(user_id, df_utenti, df_visual, df_ratings, available_books))
    print(state)
    # Se lo stato è nella Q-table, trova l'azione con il valore Q più alto
    if state in Q_table:
        # Ottieni il libro con il massimo valore Q per lo stato corrente
        print("presente")
        best_action = max(Q_table[state], key=Q_table[state].get)
        print(best_action)
    else:
        # Se lo stato non è nella Q-table, scegli un libro a caso o secondo una politica di fallback
        print("no presente")
        best_action = np.random.choice(available_books)
    
    return best_action
def main():
    df_books,df_ratings,df_visualization,df_users = load_data()
    with open("../model/qlearning.pkl", "rb") as f:
        Q_table = pickle.load(f) 
    print(len(Q_table))
    recommend_book(87,Q_table,df_users,df_visualization,df_ratings,df_books)

if __name__ == "__main__":
    main()