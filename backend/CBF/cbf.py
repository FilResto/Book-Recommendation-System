import pickle

# Carica il DataFrame dal file df.pkl
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

print(df.head())
