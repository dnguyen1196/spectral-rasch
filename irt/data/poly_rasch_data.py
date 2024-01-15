import numpy as np
import os
import collections
import pandas as pd
import ast

INVALID_RESPONSE = -99999

############### RECSYS DATASETS

def get_movies_ratings_data(path, min_n_ratings_movies=100, min_n_ratings_users=100, 
                            separator="::", user_idx=0, movie_idx=1, rating_idx=2, seed=None):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    if filename.endswith(".json"):
        f = open(filename, 'r')
    elif filename.endswith(".xls"):
        return
    
    elif "BX-Book" in filename:
        f = open(filename, encoding='latin-1')
        f.readline()
    else:
        f = open(filename, 'r')
        f.readline() # First line contains the header, throw away
    
    data_by_user = collections.defaultdict(list)
    ratings = []
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))
    movie_id_set = set()
    all_ratings = set()
    
    for line in f:        
        if filename.endswith(".json"):
            data = ast.literal_eval(line.rstrip())
            user_id = int(data["user_id"])
            movie_id = int(data["item_id"])
            rating = int(float(data["rating"]))
        elif filename.endswith(".xls"):
            return
        
        else:
            data = line.lstrip().rstrip().split(separator)
            user_id = data[user_idx].strip('"')
            movie_id = data[movie_idx].strip('"')
            rating = int(float(data[rating_idx].strip('"')))

        movie_id_set.add(movie_id)
        # Update the movies ratings and total number of observed ratings
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings+1, sum_ratings + rating)
        
        # For each user
        data_by_user[user_id].append((movie_id,rating))
        ratings.append((user_id, movie_id, rating))
        all_ratings.add(rating)
    
    # print(len(ratings_by_movies), len(movie_id_set))
    # print(len())
    # for movie_idx in movie_id_set.keys():
    #     if ratings_by_movie[movie_idx][0] < 10:
    #         print(movie_idx)
        
    # Only consider users and movies that meet the minimum number of ratings
    relevant_movie_ids = set(    
        [idx for idx in movie_id_set if ratings_by_movies[idx][0] > min_n_ratings_movies]
    )
    relevant_user_ids = set(
        [idx for idx in data_by_user.keys() if len(data_by_user[idx]) > min_n_ratings_users]
    )
    relevant_ratings = [
        (user_id, movie_id, rating) for (user_id, movie_id, rating) in ratings \
            if (user_id in relevant_user_ids and movie_id in relevant_movie_ids )
    ]
    
    # print(f"Found {len(relevant_movie_ids)} movie ids and {len(relevant_user_ids)} user ids")
    
    dataset = {}
    
    relevant_movie_ids = list(relevant_movie_ids)
    relevant_user_ids = list(relevant_user_ids)
    
    n_movies = len(relevant_movie_ids)
    n_users = len(relevant_user_ids)
    
    movie_idx_map = dict([(movie_idx, i) for i, movie_idx in enumerate(relevant_movie_ids)])
    user_idx_map = dict([(user_idx, i) for i, user_idx in enumerate(relevant_user_ids)])
    
    sorted_ratings = sorted(list(all_ratings), reverse=True) # NOTE: assuming that the larger the rating score, the 'better' the item
    # To translate into the context of education testing, movies that have high ratings -> have low 'difficulty'
    # To capture this, flip the ordering of the ratings highest_ratings -> 0, lowest_ratings -> number of unique ratings
    rating_map = dict([(rating, new_score) for new_score, rating in enumerate(sorted_ratings)])
    # This start from 0, 1, ..., number of unique ratings -1
    
    # Initially, filled with INVALID_RESPONSE
    X = np.ones((n_users, n_movies), dtype=int) * INVALID_RESPONSE
    
    # hetrec has .5 rating scheme
    for user, movie, rating in relevant_ratings:
        X[user_idx_map[user]][movie_idx_map[movie]] = rating_map[rating] # NOTE: this changes from dataset to dataset
        
    at_least_two = (X.shape[1] - np.sum(X != INVALID_RESPONSE, 1)) >= 2
    X = X[at_least_two, :]
            
    dataset['movie_idx_mapping'] = movie_idx_map
    dataset['user_idx_mapping'] = user_idx_map
    dataset['X'] = X
    dataset['rating_map'] = rating_map
    return dataset



def ml_1m(min_n_ratings_movies=50, min_n_ratings_users=50, seed=None):
    path = "datasets/ml-1m-ratings.dat"
    separator = "::"
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, separator=separator, seed=seed)
    return dataset


def ml_10m(min_n_ratings_movies=50, min_n_ratings_users=50, seed=None):
    path = "datasets/ml-10m-ratings.dat"
    separator = "::"
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, separator=separator, seed=seed)
    return dataset


def ml_20m(min_n_ratings_movies=100, min_n_ratings_users=100, seed=None):
    path = "datasets/ml-20m-ratings.csv"
    separator = ","
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, separator=separator, seed=seed)
    return dataset


def each_movie(min_n_ratings_movies=100, min_n_ratings_users=100, seed=None):
    path = "datasets/eachmovie_triple"
    separator = "   "
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, separator=separator, seed=seed)
    return dataset


def bx_book(min_n_ratings_movies=100, min_n_ratings_users=100, seed=None):
    path = "datasets/BX-Book-Ratings.csv"
    separator = ";"
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, separator=separator, seed=seed)
    return dataset


def hetrec_2k(min_n_ratings_movies=10, min_n_ratings_users=100, seed=None):
    path = "datasets/hetrec_2klens_data.dat"
    separator = '\t'
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, separator=separator, seed=seed)
    return dataset
    

def book_genome(min_n_ratings_movies=100, min_n_ratings_users=100, seed=None):
    path = "datasets/book-genome-ratings.json"
    dataset = get_movies_ratings_data(path=path, min_n_ratings_movies=min_n_ratings_movies, 
                                      min_n_ratings_users=min_n_ratings_users, seed=seed)
    return dataset



def get_heldout_ratings(X, seed=None):
    X_loo = np.copy(X)
    np.random.seed(seed)
    heldout_ratings = []
    n_users = len(X)
    
    for user in range(n_users):
        idx_responses = np.where(X[user, :] != INVALID_RESPONSE)[0]
        heldout_movie_idx = np.random.choice(idx_responses)
        held_out_rating = X[user, heldout_movie_idx]
        X_loo[user, heldout_movie_idx] = INVALID_RESPONSE
        heldout_ratings.append((user, heldout_movie_idx, held_out_rating))
        
    return X_loo, heldout_ratings



# Education dataset
def algebra_05_06(min_n_ratings_movies=10, min_n_ratings_users=10, seed=None):
    path = "datasets/algebra_2005_2006_master.txt"
    separator = "\t"
    dataset = get_movies_ratings_data(path, min_n_ratings_movies, min_n_ratings_users, separator, 
                                      user_idx=1, movie_idx=3, rating_idx=-3, seed=seed)
    return dataset

def lsat(min_n_ratings_movies=0, min_n_ratings_users=0, seed=None):
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
    return {"X": A}

def uci_student(min_n_ratings_movies=0, min_n_ratings_users=0, seed=None):
    A = []

    def grade(s):
        if s == "Best" or s == "Vg":
            return 1
        else:
            return 0

    path = "datasets/UCI-student-academic-performance.arff"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    for line in f:
        data = line.rstrip().split(",")
        relevant_grades = data[2:6]
        response = [grade(x) for x in relevant_grades]
        A.append(response)

    A = np.array(A)
    return {"X": A}

    
def grades_three(min_n_ratings_movies=0, min_n_ratings_users=0, seed=None):
    A = []
    path = "datasets/grades_three.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    for line in f:
        data = line.rstrip().split(";")
        relevant_grades = data[-3:]
        response = [int(x.strip('"')) for x in relevant_grades]
        A.append(response)

    A = np.array(A)
    median_grades = np.median(A, 0) # Compute the median grades for all students 
    # Mark students below median grades as 0 and above as 1
    _, m = A.shape 
    for i in range(m):
        A[:, i] = np.where(A[:, i] <= median_grades[i], 0, 1)

    return {"X": A}