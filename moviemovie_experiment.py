import numpy as np
from src import utils


def movie_dot_mean(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    user_matrix = utils.indot_sim(name,matrix)

    data = utils.input_data(path)
    mean_result = []

    for i in data:
        movieID = i[0]
        # movieID = 3
        userID = i[1]
        user = user_matrix[movieID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if movieID in knn:
            i = np.where(knn == movieID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        score = (np.sum(np.take(matrix[:, userID], knn.tolist())) / float(k)) + 3
        mean_result.append(score)

    rating = 'movie_dot_mean'
    # print(knn)

    return mean_result, rating


def movie_dot_weigthed_mean(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    user_matrix = utils.indot_sim(name,matrix)

    data = utils.input_data(path)
    weighted_mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        user = user_matrix[movieID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]



        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        knn_sim = user[knn]
        if np.sum(knn_sim) != 0:
            weight = knn_sim / np.sum(knn_sim)
            score = np.sum(np.multiply(np.take(matrix[:,userID], knn.tolist()), weight)) + 3
        else:
            score = np.sum(matrix[:,userID]) / np.size(np.nonzero(matrix[:,userID])) + 3
        weighted_mean_result.append(score)

    rating = 'movie_dot_weigthed_mean'

    return weighted_mean_result, rating

def movie_cos_mean(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    normalized_matrix = np.linalg.norm(matrix, axis=0) * matrix
    user_matrix = utils.cosine_sim(name, normalized_matrix)

    data = utils.input_data(path)
    mean_result = []
    for i in data:
        movieID = i[0]
        # movieID = 3
        userID = i[1]
        user = user_matrix[movieID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        score = (np.sum(np.take(matrix[movieID, :], knn.tolist())) / float(k)) + 3
        mean_result.append(score)

    rating = 'movie_cos_mean'
    # print(knn)

    return mean_result, rating

def movie_cos_weigthedmean(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    normalized_matrix = np.linalg.norm(matrix, axis=0) * matrix
    user_matrix = utils.cosine_sim(name, normalized_matrix)

    data = utils.input_data(path)
    weighted_mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        user = user_matrix[movieID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        knn_sim = user[knn]
        if np.sum(knn_sim) != 0:
            weight = knn_sim / np.sum(knn_sim)
            score = np.sum(np.multiply(np.take(matrix[:,userID], knn.tolist()), weight)) + 3
        else:
            score = np.sum(matrix[:,userID]) / np.size(np.nonzero(matrix[:,userID])) + 3
        weighted_mean_result.append(score)

    rating = 'movie_cos_weigthedmean'

    return weighted_mean_result, rating










