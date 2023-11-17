import projectLib as lib
import rbm
import numpy as np

def get_R(data):
    Stats = lib.getUsefulStats(data)
    R = np.zeros((np.max(Stats["users"])+1, np.max(Stats["n_movies"])+1))
    for x in data:
        R[[x[1],x[0]]] = x[2]
    return R

def predict_N(Stats, W, allUsersRatings, predictType="exp"):
    R_hat = np.zeros((np.max(Stats["users"])+1, np.max(Stats["n_movies"])+1))
    for i in Stats["u_users"]:
        for j in Stats["u_movies"]:
            R_hat[i, j] = rbm.predictMovieForUser(j, i, W, allUsersRatings, predictType=predictType)
    return R_hat


def neighbourhood(dataset,R_hat):
    Stats = lib.getUsefulStats(dataset)
    num_items = Stats["n_movies"]
    num_users = Stats["n_users"]
    R = get_R(dataset)
    R_error = R - R_hat
    for (movie, user) in zip(Stats["movies"], Stats["users"]):
        R_error[user, movie] = None

    # Calculate the similarity matrix between movies using cosine similarity
    n1 = 0
    n2 = 0
    dot_product = 0
    similarity_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            if i == j:
                similarity_matrix[i, j] = 1
                continue
            for k in range(num_users):
                if R_error[k,i] != None and R_error[k,j] != None:
                    dot_product += R_error[k, i] * R_error[k, j]
                    n1 += R_error[k,i]**2
                    n2 += R_error[k,j]**2
            similarity = dot_product / (np.sqrt(n1) * np.sqrt(n2))
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
            n1 = 0
            n2 = 0
            dot_product = 0

    # Find the L nearest neighbors for each movie (excluding itself)
    L = 10  # Number of neighbors to consider
    neighbors = []

    for i in range(num_items):
        sorted_indices = np.argsort(abs(similarity_matrix[i]))[::-1]  # Sort in descending order
        nearest_neighbors = [idx for idx in sorted_indices if idx != i][:L]
        neighbors.append(nearest_neighbors)

    # Compute R^N for each user and each movie
    R_N = np.zeros(R.shape)

    for user in range(num_users):
        for movie in range(num_items):
            if R_error[user, movie] == None:  # Only compute R^N for rated movies
                numerator = 0
                denominator = 0
                for neighbor in neighbors[movie]:
                    if R_error[user, neighbor] != None:
                        numerator += similarity_matrix[movie, neighbor] * R_error[user, neighbor]
                        denominator += np.abs(similarity_matrix[movie, neighbor])
                if denominator != 0:
                    R_N[user, movie] = R_hat[user, movie] + (numerator / denominator)
            else:
                R_N[user, movie] = R[user, movie]
    
    return [R_N[user, movie] for (movie, user) in zip(Stats["movies"], Stats["users"])]