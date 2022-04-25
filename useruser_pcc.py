import numpy as np
from src import utils


def user_dot_mean_pcc(path, k):
    name = 'user'
    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (matrix - np.sum(matrix, axis=0)) / len(matrix)

    pcc = pcc/np.linalg.norm(matrix, axis=0)

    user_matrix = utils.indot_sim(name,pcc)

    data = utils.input_data(path)
    mean_result = []

    for i in data:
        movieID = i[0]
        userID = i[1]
        user = user_matrix[userID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        score = (np.sum(np.take(matrix[movieID, :], knn.tolist())) / float(k)) + 3
        mean_result.append(score)

    rating = 'user_dot_mean_pcc'

    return mean_result, rating


def user_dot_weigthed_mean_pcc(path, k):
    name = 'user'
    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (matrix - np.sum(matrix, axis=0)) / len(matrix)

    pcc = pcc/np.linalg.norm(matrix, axis=0)

    user_matrix = utils.indot_sim(name,pcc)

    data = utils.input_data(path)
    weighted_mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        user = user_matrix[userID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        knn_sim = user[knn]
        if np.sum(knn_sim) != 0:
            weight = knn_sim / np.sum(knn_sim)
            score = np.sum(np.multiply(np.take(matrix[movieID, :], knn.tolist()), weight)) + 3
        else:
            score = np.sum(matrix[movieID, :]) / np.size(np.nonzero(matrix[movieID, :])) + 3
        weighted_mean_result.append(score)

    rating = 'user_dot_weigthed_mean_pcc'

    return weighted_mean_result, rating

def user_cos_mean_pcc(path, k):
    name = 'user'
    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (matrix - np.sum(matrix, axis=0)) / len(matrix)

    pcc = pcc/np.linalg.norm(matrix, axis=0)

    user_matrix = utils.cosine_sim(name, pcc)

    data = utils.input_data(path)
    mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        user = user_matrix[userID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        score = (np.sum(np.take(matrix[movieID, :], knn.tolist())) / float(k)) + 3
        mean_result.append(score)

    rating = 'user_cos_mean_pcc'

    return mean_result, rating

def user_cos_weigthedmean_pcc(path, k):
    name = 'user'
    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (matrix - np.sum(matrix, axis=0)) / len(matrix)

    pcc = pcc/np.linalg.norm(matrix, axis=0)

    user_matrix = utils.cosine_sim(name, pcc)

    data = utils.input_data(path)
    weighted_mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        user = user_matrix[userID]

        knn = np.argsort(user, kind='heapsort')[::-1][0: k + 1]

        if userID in knn:
            i = np.where(knn == userID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        knn_sim = user[knn]
        if np.sum(knn_sim) != 0:
            weight = knn_sim / np.sum(knn_sim)
            score = np.sum(np.multiply(np.take(matrix[movieID, :], knn.tolist()), weight)) + 3
        else:
            score = np.sum(matrix[movieID, :]) / np.size(np.nonzero(matrix[movieID, :])) + 3
        weighted_mean_result.append(score)

    rating = 'user_cos_weigthedmean_pcc'

    return weighted_mean_result, rating


