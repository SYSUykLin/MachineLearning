import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_Data(moves_name, ratings_name):
    print('loading data ......')
    movies = pd.read_csv('../Data/' + moves_name)
    ratings = pd.read_csv('../Data/' + ratings_name)
    n_movies = len(movies)
    n_ratings = len(ratings)
    last_movies = int(movies.iloc[-1].movieId)
    last_users = int(ratings.iloc[-1].userId)
    dataMat = np.zeros((last_users, last_movies))
    for i in range(len(ratings)):
        rating = ratings.loc[i]
        dataMat[int(rating.userId) - 1, int(rating.movieId) - 1] = rating['rating']
        pass
    return dataMat

if __name__ == '__main__':
    load_Data('movies.csv', 'ratings.csv')