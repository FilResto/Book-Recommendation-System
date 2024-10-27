import pickle

with open('df_book.pkl', 'rb') as file:
    df_book = pickle.load(file)
print(df_book.head())


with open("df_user.pkl","rb") as file:
    df_users = pickle.load(file)
print(df_users.head())

with open("df_ratings.pkl","rb") as file:
    df_ratings = pickle.load(file)
print(df_ratings.head())

with open("df_visualization.pkl","rb") as file:
    df_visualizations = pickle.load(file)
print(df_visualizations.head())