import numpy as np
import matplotlib.pyplot as plt

def getTrainingData():
    return np.genfromtxt("training.csv", delimiter=",",dtype=int)

def getValidationData():
    return np.genfromtxt("validation.csv", delimiter=",",dtype=int)


def getUsefulStats(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()

    users = [x[1] for x in training]
    u_users = np.unique(users).tolist()

    return {
        "movies": movies, # movie IDs
        "u_movies": u_movies, # unique movie IDs
        "n_movies": len(u_movies), # number of unique movies

        "users": users, # user IDs
        "u_users": u_users, # unique user IDs
        "n_users": len(u_users), # number of unique users

        "ratings": [x[2] for x in training], # ratings
        "n_ratings": len(training) # number of ratings
    }


def getAllUsersRatings(users, training):
    return [getRatingsForUser(user, training) for user in np.sort(users)]


def getRatingsForUser(user, training):
    # user is a user ID
    # training is the training set
    # ret is a matrix, each row is [m, r] where
    #   m is the movie ID
    #   r is the rating, 1, 2, 3, 4 or 5
    return np.array([[x[0], x[2]] for x in training if x[1] == user])

# RMSE function to tune your algorithm
def rmse(r, r_hat):
    r = np.array(r)
    r_hat = np.array(r_hat)
    return np.linalg.norm(r - r_hat) / np.sqrt(len(r))


def adaptive_lr(total_list, size, threshold, rate):
    try:
        tested_avg = np.mean(total_list[-size:])
        last_avg = np.mean(total_list[-2*size : -size])
        if last_avg - tested_avg < threshold:   
            return rate
        else:   
            return 1
    except:
        return 1

def plot_loss(train_list, validation_list):
    # x = np.linspace(len(train_list))
    plt.plot(train_list, label = 'train')
    plt.plot(validation_list, label = 'validation')
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.title('RMSE Lists')
    plt.legend()
    plt.show()