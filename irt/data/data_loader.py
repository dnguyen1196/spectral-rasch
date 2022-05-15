import numpy as np
import os
import collections
import pandas as pd
import ast


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

def remove_low_ratings_top_items(ratings, cutoff):
    ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in ratings if num_ratings > cutoff
        ]
    return ratings


def hetrec_2k(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_LOW, top_k_cutoff=TOP_K_CUTOFF):
    path = "datasets/hetrec_2klens_data.dat"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    
    f = open(filename, 'r')        
    f.readline()

    data_by_user = collections.defaultdict(list)    
    ratings = []
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))
    
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split('\t')
        user_id = int(data[0])
        movie_id = int(data[1])
        movie_id_set.add(movie_id)
        rating = float(data[2])
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
    
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL)
            )
    
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(data_by_user.keys())])
    
    n = len(new_user_id)
    m = len(movie_id_set)
    
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
        
    return A
    

def ml_1m(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_LOW, top_k_cutoff=TOP_K_CUTOFF):
    path = "datasets/ml-1m-ratings.dat"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split('::') # Note the separator
        user_id = int(data[0])-1
        movie_id = int(data[1])-1
        movie_id_set.add(movie_id)
        rating = float(data[2])
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
        
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])

    # Convert ratings to binary responses
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL)
            )
    
    # Construct the incident A matrix
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(data_by_user.keys())])
    n = len(new_user_id)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
        
    return A
    

def ml_100k(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_LOW, top_k_cutoff=TOP_K_CUTOFF):
    # TODO: load ranking data as well (ranked data determined by ratings)
    path = "datasets/ml-100k-ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split(',') # Note the seperator
        user_id = int(data[0])-1
        movie_id = int(data[1])-1
        movie_id_set.add(movie_id)
        rating = float(data[2])
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
    
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL) # Note the flipping of 0 and 1 for the response
            )
    
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(data_by_user.keys())])
    n = len(new_user_id)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
    
    return A


def ml_10m(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF, top_k_cutoff=TOP_K_CUTOFF):
    path = "datasets/ml-10m-ratings.dat"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split('::') # Note the separator
        user_id = int(data[0])-1
        movie_id = int(data[1])-1
        movie_id_set.add(movie_id)
        rating = float(data[2])
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
        
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL)
            )
    
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(data_by_user.keys())])
    n = len(new_user_id)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
        
    return A


def ml_20m(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_ML20M, top_k_cutoff=TOP_K_CUTOFF, user_rating_cutoff=USERS_RATINGS_CUTOFF_ML20M):
    path = "datasets/ml-20m-ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.rstrip().split(',') # Note the seperator
        user_id = int(data[0])-1
        movie_id = int(data[1])-1
        movie_id_set.add(movie_id)
        rating = float(data[2])
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
    
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)

    # Remove users with a small number of ratings (noisy)
    relevant_users = set()
    for user_id in data_by_user.keys():
        if len(data_by_user[user_id]) >= user_rating_cutoff:
            relevant_users.add(user_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set and user_id in relevant_users:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL) # Note the flipping of 0 and 1 for the response
            )

    updated_user = set([user for user, _, _ in binary_response])
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(updated_user)])
    n = len(updated_user)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
    
    return A


def jester(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_LOW):
    path = "datasets/jester-data-1.xls"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    df = pd.read_excel(filename)
    A = df.to_numpy()[:, 1:] # The first column is the number of jokes rated by that user
    _, m = A.shape
    A = np.ma.masked_where(A == 99, A) # Mask the missing data value
    avg_ratings = np.ma.mean(A, 1) # Compute the average rating for each user
    
    all_ratings = [(int(joke_id), float(avg_rating), int(num_ratings)) 
                   for (joke_id, avg_rating, num_ratings) in list(zip(np.arange(m).astype(int), np.ma.mean(A, 0), A.count(0).astype(int)))]
    
    # All the jokes with higher than average ratings
    avg_A = np.outer(avg_ratings, np.ones((m,))) # This should have shape (n, m)
    B = np.where(A > avg_A, APPROVAL, DISAPPROVAL) # Replace the ratings > than the average with APPROVAL and DISAPPROVAL otherwise
    B = np.where(A == 99, INVALID_RESPONSE, B)
    include_idx = np.where(A.count(0) > cutoff)[0] # Remove the number of jokes with too few ratings
    
    if return_ratings:
        return B[:, include_idx].T, [all_ratings[i] for i in include_idx]
    
    return B[:, include_idx].T


def book_genome(return_ratings=True, cutoff=BOOK_GENOME_RATINGS_CUTOFF, top_k_cutoff=TOP_K_CUTOFF_BOOK, user_rating_cutoff=USERS_RATINGS_CUTOFF_GENOME):
    # {"item_id": 41335427, "user_id": 0, "rating": 5}
    path = "datasets/book-genome-ratings.json"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = ast.literal_eval(line.rstrip())
        user_id = data["user_id"]
        movie_id = data["item_id"]
        movie_id_set.add(movie_id)
        rating = float(data["rating"])
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
        
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    # Remove users with a small number of ratings (noisy)
    relevant_users = set()
    for user_id in data_by_user.keys():
        if len(data_by_user[user_id]) >= user_rating_cutoff:
            relevant_users.add(user_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set and user_id in relevant_users:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL) # Note the flipping of 0 and 1 for the response
            )

    updated_user = set([user for user, _, _ in binary_response])
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(updated_user)])
    n = len(updated_user)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
        
    return A
    

def bx_book(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_HIGH, top_k_cutoff=TOP_K_CUTOFF):
    # "276725";"034545104X";"0"
    path = "datasets/BX-Book-Ratings.csv"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, encoding="latin-1")
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        # line = str.unicode(line, errors='ignore')
        data = line.rstrip().split(';') # Note the separator
        user_id = int(data[0].strip('"'))
        movie_id = data[1].strip('"') # Note the movie id type
        movie_id_set.add(movie_id)
        rating = float(data[2].strip('"'))
        
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
        
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL)
            )
    
    updated_user = set([user for user, _, _ in binary_response])
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(updated_user)])
    n = len(updated_user)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
        
    return A
    

def each_movie(return_ratings=True, cutoff=MOVIES_RATINGS_CUTOFF_HIGH, top_k_cutoff=TOP_K_CUTOFF):
    #    1.5840000e+03   7.4424000e+04   5.0000000e+00
    path = "datasets/eachmovie_triple"
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    f = open(filename, 'r')
    
    data_by_user = collections.defaultdict(list)
    f.readline()
    ratings_by_movies = collections.defaultdict(lambda : (0, 0.))

    ratings = []
    movie_id_set = set()
    
    for line in f:
        data = line.lstrip().rstrip().split("   ") # Note the separator
        # data = line.rstrip().split('\t') # Note the separator
        user_id = int(float(data[1])) # Note the user id is on the second
        movie_id = int(float(data[0]))
        movie_id_set.add(movie_id)
        rating = float(data[2])
        num_ratings, sum_ratings = ratings_by_movies[movie_id]
        ratings_by_movies[movie_id] = (num_ratings + 1, sum_ratings + rating)
        
        data_by_user[user_id].append(rating)    
        ratings.append((user_id, movie_id, rating))
        
    # Remove movies with a small number of ratings (noisy)
    for movie_id in list(ratings_by_movies.keys()):
        (num_ratings, sum_ratings) = ratings_by_movies[movie_id]
        if num_ratings < cutoff:
            ratings_by_movies.pop(movie_id, None)
            movie_id_set.remove(movie_id)
    
    replace_movie_id = dict([idx, i] for i, idx in enumerate(movie_id_set))
    avg_ratings = dict([(user_id, np.mean(data_by_user[user_id])) for user_id in data_by_user.keys()])
    binary_response = []
    for  user_id, movie_id, rating in ratings:
        if movie_id in movie_id_set:
            binary_response.append(
                (user_id, replace_movie_id[movie_id], APPROVAL) if rating > avg_ratings[user_id] else (user_id, replace_movie_id[movie_id], DISAPPROVAL)
            )
    
    new_user_id = dict([(user_id, i) for i, user_id in enumerate(data_by_user.keys())])
    n = len(new_user_id)
    m = len(movie_id_set)
    A = np.ones((m, n), dtype=np.int) * INVALID_RESPONSE
    for user_id, movie_id, res in binary_response:
        A[movie_id, new_user_id[user_id]] = res
        
    if return_ratings:
        # Return normalized ratings
        avg_ratings = [
            (replace_movie_id[old_movie_id], sum_ratings/num_ratings, num_ratings) for old_movie_id, (num_ratings, sum_ratings) in ratings_by_movies.items()
        ]
        avg_ratings = [
            (movie_id, avg_rating, num_ratings) for movie_id, avg_rating, num_ratings in avg_ratings
        ]        
        return A, remove_low_ratings_top_items(avg_ratings, top_k_cutoff)
        
    return A