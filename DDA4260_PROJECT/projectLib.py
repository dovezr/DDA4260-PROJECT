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


def adaptive_lr(total_list, test_size, improve_threshold, rate):
    if 2*test_size <= len(total_list):
        tested_avg = np.mean(total_list[-test_size:])
        last_avg = np.mean(total_list[-2*test_size : -test_size])
        if last_avg - tested_avg < improve_threshold:   
            return rate
    return 1

def early_stop(total_list, stop_size, stop_threshold):
    if 2*stop_size <= len(total_list):
        tested_avg = np.mean(total_list[-stop_size:])
        last_avg = np.mean(total_list[-2*stop_size : -stop_size])
        if last_avg - tested_avg < stop_threshold:  
            print('early stop') 
            return True
    return False

def kernighan_lin(graph):

    # Initialize the matching to be empty.
    matching = []

    # While there are still unmatched vertices, find a pair of vertices that are not
    # matched and swap them if doing so reduces the total weight of the matching.
    while len(matching) < len(graph):
        # Find a pair of vertices that are not matched.
        unmatched_vertices = set(graph.keys()) - set(matching)
        best_vertex_1, best_vertex_2 = None, None
        best_weight_change = float("inf")
        for vertex_1 in unmatched_vertices:
            for vertex_2 in graph[vertex_1]:
                if vertex_2 not in matching:
                    weight_change = graph[vertex_1][vertex_2] - sum(
                        graph[vertex_1][v] for v in matching if v != vertex_2)
                if weight_change < best_weight_change:
                    best_vertex_1, best_vertex_2, best_weight_change = vertex_1, vertex_2, weight_change

    # Swap the vertices if doing so reduces the total weight of the matching.
    if best_weight_change < 0:
        matching.append((best_vertex_1, best_vertex_2))
        del graph[best_vertex_1][best_vertex_2]
        for vertex in graph[best_vertex_2]:
            graph[vertex_2][vertex] -= best_weight_change

    # Return the matching.
    return matching


def createBipartiteGraph(training, W):
    graph = np.zeros(training['n_users']+training['n_movies'])
    for user, movie in zip(training['users'],training['movies']):
        graph[user][movie] = graph[movie][user] = W[movie][user]
    return graph

def improveRBMWithKL(training, W, b_h, b_v):
    # 创建RBM模型的二部图
    graph = createBipartiteGraph(training, W)

    # 使用Kernighan-Lin算法找到最小权重的完美匹配
    matching = kernighan_lin(graph)

    # 根据匹配更新RBM模型的权重和偏置
    for visible, hidden in matching:
        # 更新可见单元和隐藏单元之间的权重
        W[visible, hidden] -= 1

        # 更新可见单元的偏置
        b_h[visible] -= 1

        # 更新隐藏单元的偏置
        b_v[hidden] -= 1

    return W, b_h, b_v

def plot_loss(train_list, validation_list):
    # x = np.linspace(len(train_list))
    plt.plot(train_list, label = 'train')
    plt.plot(validation_list, label = 'validation')
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.title('RMSE Lists')
    plt.legend()
    plt.show()