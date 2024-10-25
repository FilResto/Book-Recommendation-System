import pandas as pd
df = pd.read_csv('/data/booksAll.csv', usecols=['bookId','title','series','author','description','language','genres','pages','awards','bbevote','price'])
prova = df.head()
print(df)