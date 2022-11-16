from typing import Mapping
import numpy as np
import os
import collections
import pandas as pd
import ast
import nimfa
import torch as th
import scipy as sp

MOVIES_RATINGS_CUTOFF_LOW = 10
MOVIES_RATINGS_CUTOFF = 25
MOVIES_RATINGS_CUTOFF_HIGH = 50

MOVIES_RATINGS_CUTOFF_ML20M = 50
USERS_RATINGS_CUTOFF_ML20M = 40

BOOK_GENOME_RATINGS_CUTOFF = 50
USERS_RATINGS_CUTOFF_GENOME = 25

TOP_K_CUTOFF = 100
TOP_K_CUTOFF_BOOK = 250


APPROVAL = 0 # This is so that the items with the highest difficulty parameters also corresponds to the items of highest quality
# If a user can't 'solve' a movie, then she likes it.
DISAPPROVAL = 1
INVALID_RESPONSE = -99999

############### EDUCATION DATASETS

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

def uci_student():
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
    A = A.T
    return A
    
def grades_three():
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

    A = A.T
    return A
    
def algebra_05_06():
    A = []
    path = "datasets/algebra_2005_2006_master.txt"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    f.readline() # Skip first line

    all_problems = set()
    all_students = set()
    all_responses = []

    for line in f:
        data = line.rstrip().split("\t")
        problem_id = data[3]
        student_id = data[1]
        all_problems.add(problem_id)
        all_students.add(student_id)

        response = int(data[-3])
        all_responses.append((problem_id, student_id, response))

    m = len(all_problems)
    n = len(all_students)
    
    problems_idx = dict([problem, idx] for idx, problem in enumerate(all_problems))
    students_idx = dict([student, idx] for idx, student in enumerate(all_students))

    A = np.ones((m, n)) * INVALID_RESPONSE

    for (problem, student, response) in all_responses:
        A[problems_idx[problem], students_idx[student]] = response
    
    return A, all_problems

def riiid():
    path = "datasets/records_qmin=1000_smin=10000.txt"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    all_students = set()
    all_problems = set()
    f.readline()
    binary_responses = []

    for line in f:
        data = line.rstrip().split(',') # Note the seperator
        student = int(data[0])
        question = int(data[1])
        response = int(data[2])
        all_students.add(student)
        all_problems.add(question)
        binary_responses.append([student, question, response])
            
    replace_question_id = dict([idx, i] for i, idx in enumerate(all_problems))
    replace_student_id =  dict([idx, i] for i, idx in enumerate(all_students))
    
    n = len(all_students)
    m = len(all_problems)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE

    for (s, q, res) in binary_responses:
        A[replace_question_id[q], replace_student_id[s]] = res
    
    return A

def riiid_small():
    path = "datasets/records_qmin=5000_smin=5000.txt"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    all_students = set()
    all_problems = set()
    f.readline()
    binary_responses = []

    for line in f:
        data = line.rstrip().split(',') # Note the seperator
        student = int(data[0])
        question = int(data[1])
        response = int(data[2])
        all_students.add(student)
        all_problems.add(question)
        binary_responses.append([student, question, response])
    
    replace_question_id = dict([idx, i] for i, idx in enumerate(all_problems))
    replace_student_id =  dict([idx, i] for i, idx in enumerate(all_students))
    
    n = len(all_students)
    m = len(all_problems)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE

    for (s, q, res) in binary_responses:
        A[replace_question_id[q], replace_student_id[s]] = res
    
    return A

############### RECSYS DATASETS


def factorize(V, rank=30):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model. 
    
    :param V: The MovieLens data matrix. 
    :type V: `numpy.matrix`
    """
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=rank, max_iter=30, version='r', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    fit = snmf()
    fit.fit.sparseness()
    return fit.basis(), fit.coef()


def lrmc(A, rank=30):
    W, H = factorize(A, rank)
    return W, H


def construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users=None, num_movies=None, return_mapping=False):
    user_id_list = [(user_id, ratings_count_user[user_id]) for user_id in user_id_set]
    movie_id_list = [(movie_id, ratings_count_movie[movie_id]) for movie_id in movie_id_set]
    user_id_list = sorted(user_id_list, key=lambda x: x[1])[::-1]
    movie_id_list = sorted(movie_id_list, key=lambda x: x[1])[::-1]

    if num_users is None:
        num_users = len(user_id_list)
    if num_movies is None: 
        num_movies = len(movie_id_list)

    user_id_list = user_id_list[:min(num_users, len(user_id_list))]
    movie_id_list = movie_id_list[:min(num_movies, len(movie_id_list))]

    user_mapping = dict([(user_id, i) for i, (user_id, _) in enumerate(user_id_list)])
    movie_mapping = dict([(movie_id, i) for i, (movie_id, _) in enumerate(movie_id_list)])
    
    m, n = len(user_mapping), len(movie_mapping)
    A = np.zeros((m, n))

    for user, movie, rating in ratings:
        if user in user_mapping and movie in movie_mapping:
            uid = user_mapping[user]
            mid = movie_mapping[movie]
            A[uid, mid] = rating

    # Remove row with all zeros
    all_zeros_row = np.all(A == 0, 1)
    non_zeros_row = np.logical_not(all_zeros_row)
    A = A[non_zeros_row, :]

    A = sp.sparse.csr_matrix(A)
    if return_mapping:
        return A
    return A, user_mapping, movie_mapping


def remove_low_ratings_top_items(ratings, cutoff):
    ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in ratings if num_ratings > cutoff
        ]
    return ratings


def sparsify(full_response, p, seed=None):
    m, n = full_response.shape
    np.random.seed(seed)
    mask = np.random.rand(m, n)
    mask = np.where(mask < p, np.ones_like(mask), np.zeros_like(mask))
    sparse_response = np.where(mask == 1, full_response, INVALID_RESPONSE*np.ones_like(full_response))
    return sparse_response


def convert_completed_ratings_to_binary(W, H, by_user=True, rating_data=True):
    A_lr = W @ H
    print(A_lr.shape)
    n, m = A_lr.shape
    average = np.mean(A_lr, 1)
    
    approval = 0 if rating_data else 1
    disapproval = 1 if rating_data else 0
    
    for l in range(n):
        A_l = A_lr[l, :]
        A_lr[l, :] = np.where(A_l < average[l], disapproval * np.ones_like(A_l), approval * np.ones_like(A_l))
    
    return A_lr if by_user else A_lr.T


def ml_100k(return_ratings=True, rank=30):
    preload_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets/preload/ml_100K_r=30_WH.pkl")
    if not os.path.isfile(preload_path): # If the preload file doesn't exist do matrix ocmpletion
        path = "datasets/ml-100k-ratings.csv"
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
        f = open(filename, 'r')
        
        f.readline()
        ratings = []
        movie_id_set = set()
        user_id_set = set()

        ratings_count_user = collections.defaultdict(int)
        ratings_count_movie = collections.defaultdict(int)
        ratings_for_movies = collections.defaultdict(float)

        for line in f:
            data = line.rstrip().split(',') # Note the seperator
            user_id = int(data[0])
            movie_id = int(data[1])
            rating = float(data[2])

            movie_id_set.add(movie_id)
            user_id_set.add(user_id)
            ratings.append((user_id, movie_id, rating))
            ratings_count_user[user_id] += 1
            ratings_count_movie[movie_id] += 1
            ratings_for_movies[movie_id] += rating

        # Remove movies with very few ratings
        num_users = len(user_id_set)
        num_movies = len(movie_id_set)
        # Learn a low rank approximation of the rating matrix
        A, _, movie_id_map = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)
        W, H = lrmc(A, rank)
        
    else:
        return th.load(preload_path)
    
    return W, H


def ml_1m(return_ratings=True, rank=30):
    preload_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets/preload/ml_1M_r=30_WH.pkl")
    
    if not os.path.isfile(preload_path):
        path = "datasets/ml-1m-ratings.dat"
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
        f = open(filename, 'r')
        
        f.readline()
        ratings = []
        movie_id_set = set()
        user_id_set = set()

        ratings_count_user = collections.defaultdict(int)
        ratings_count_movie = collections.defaultdict(int)
        ratings_for_movies = collections.defaultdict(float)

        for line in f:
            data = line.rstrip().split('::') # Note the separator
            user_id = int(data[0])
            movie_id = int(data[1])
            rating = float(data[2])

            movie_id_set.add(movie_id)
            user_id_set.add(user_id)
            ratings.append((user_id, movie_id, rating))
            ratings_count_user[user_id] += 1
            ratings_count_movie[movie_id] += 1
            ratings_for_movies[movie_id] += rating

        # Remove movies with very few ratings
        num_users = len(user_id_set)
        num_movies = len(movie_id_set)
        # Learn a low rank approximation of the rating matrix
        A, _, movie_id_map = construct_rating_matrix(ratings, user_id_set, movie_id_set, ratings_count_user, ratings_count_movie, num_users, num_movies)
        W, H = lrmc(A, rank)
    else:
        W, H = th.load(preload_path)
    
    return W, H


