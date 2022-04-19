import numpy as np
import os
import collections

from irt.algorithms.conditional_mle import INVALID_RESPONSE

def lsat():
    A = []
    path = "datasets/LSAT_dataset.txt"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    f.readline()
    lines = f.readlines()
    for line in lines:
        data = line.rstrip().split('\t')
        student_id = data[0]
        response = [int(x) for x in data[1:]]
        A.append(response)
    A = np.array(A)
    A = A.T
    return A


def ml_100k():
    path = "datasets/ml-100k-ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    
    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split(',')
        user_id = int(data[0])-1
        movie_id = int(data[1])-1
        movie_id_set.add(movie_id)
        rating = int(float(data[2]))
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = [
        (user_id, replace_movie_id[movie_id], 1) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], 0) for user_id, movie_id, rating in ratings
    ]
    
    n = len(data_by_user)
    m = len(movie_id_set)
    
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    
    for user_id, movie_id, res in binary_response:
        A[movie_id, user_id] = res
    
    return A


def uci_student_performance():
    
    
    return