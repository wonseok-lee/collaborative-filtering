import numpy as np
from src import utils



def movie_dot_mean_pcc(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (np.transpose(matrix) - np.sum(matrix, axis=1)) / len(matrix)

    pcc = pcc / np.linalg.norm(matrix, axis=1)
    pcc = np.transpose(pcc)

    item_matrix = utils.indot_sim(name,pcc)

    data = utils.input_data(path)
    mean_result = []

    for i in data:
        movieID = i[0]
        userID = i[1]
        item = item_matrix[movieID]

        knn = np.argsort(item, kind='heapsort')[::-1][0: k + 1]

        if movieID in knn:
            i = np.where(knn == movieID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        score = (np.sum(np.take(matrix[:, userID], knn.tolist())) / float(k)) + 3
        mean_result.append(score)

    rating = 'movie_dot_mean_pcc'

    return mean_result, rating


def movie_dot_weigthed_mean_pcc(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (np.transpose(matrix) - np.sum(matrix, axis=1)) / len(matrix)

    pcc = pcc / np.linalg.norm(matrix, axis=1)
    pcc = np.transpose(pcc)

    item_matrix = utils.indot_sim(name,pcc)

    data = utils.input_data(path)
    weighted_mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        item = item_matrix[movieID]

        knn = np.argsort(item, kind='heapsort')[::-1][0: k + 1]

        if movieID in knn:
            i = np.where(knn == movieID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        knn_sim = item[knn]
        if np.sum(knn_sim) != 0:
            weight = knn_sim / np.sum(knn_sim)
            score = np.sum(np.multiply(np.take(matrix[:, userID], knn.tolist()), weight)) + 3
        else:
            score = np.sum(matrix[:, userID]) / np.size(np.nonzero(matrix[:, userID])) + 3
        weighted_mean_result.append(score)

    rating = 'movie_dot_weigthed_mean_pcc'

    return weighted_mean_result, rating

def movie_cos_mean_pcc(path, k):
    name = 'item'
    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (np.transpose(matrix) - np.sum(matrix, axis=1)) / len(matrix)

    pcc = pcc / np.linalg.norm(matrix, axis=1)
    pcc = np.transpose(pcc)

    item_matrix = utils.cosine_sim(name, pcc)

    data = utils.input_data(path)
    mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        item = item_matrix[movieID]

        knn = np.argsort(item, kind='heapsort')[::-1][0: k + 1]

        if movieID in knn:
            i = np.where(knn == movieID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        score = (np.sum(np.take(matrix[:, userID], knn.tolist())) / float(k)) + 3
        mean_result.append(score)

    rating = 'movie_cos_mean_pcc'

    return mean_result, rating

def movie_cos_weigthedmean_pcc(path, k):
    name = 'item'

    matrix = utils.train_matrix(3)
    matrix = matrix.toarray()
    matrix[matrix == 0] = 0.0005

    pcc = (np.transpose(matrix) - np.sum(matrix, axis=1)) / len(matrix)

    pcc = pcc/np.linalg.norm(matrix, axis=1)
    pcc = np.transpose(pcc)

    item_matrix = utils.cosine_sim(name, pcc)

    data = utils.input_data(path)
    weighted_mean_result = []
    for i in data:
        movieID = i[0]
        userID = i[1]
        item = item_matrix[movieID]

        knn = np.argsort(item, kind='heapsort')[::-1][0: k + 1]

        if movieID in knn:
            i = np.where(knn == movieID)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)

        knn_sim = item[knn]
        if np.sum(knn_sim) != 0:
            weight = knn_sim / np.sum(knn_sim)
            score = np.sum(np.multiply(np.take(matrix[:, userID], knn.tolist()), weight)) + 3
        else:
            score = np.sum(matrix[:, userID]) / np.size(np.nonzero(matrix[:, userID])) + 3
        weighted_mean_result.append(score)

    rating = 'movie_cos_weigthedmean_pcc'

    return weighted_mean_result, rating


