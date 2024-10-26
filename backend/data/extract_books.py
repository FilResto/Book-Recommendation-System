import pandas as pd
import pickle
df = pd.read_csv('booksAll.csv',usecols=['bookId', 'title', 'series', 'author', 'description', 'language', 'genres', 'pages', 'awards', 'bbeVotes', 'price'])
with open('df_book.pkl', 'wb') as file:
    pickle.dump(df, file)
df = pd.read_csv("users.csv")
with open('df_user.pkl', 'wb') as file:
    pickle.dump(df, file)
