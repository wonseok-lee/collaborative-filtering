import numpy as np
import csv
import scipy.sparse as sp


def input_data(path):
    with open(path, "r") as f:
        rough_data = csv.reader(f)
        movie = []
        user = []
        for i in rough_data:
            movie.append(int(i[0]))
            user.append(int(i[1]))
    matrix = np.column_stack((movie,user))
    return matrix

# def output_helper():
#     path = 'data/train.csv'
#     with open(path, "r") as f:
#         rough_data = csv.reader(f)
#         movie = []
#         user = []
#         for i in rough_data:
#             movie.append(int(i[0]))
#             user.append(int(i[1]))
#     return movie, user

def indot_sim(option, matrix):
    if option == 'user':
        user_score = np.dot(np.transpose(matrix),matrix)
        return user_score

    elif option == 'item':
        item_score = np.dot(matrix,np.transpose(matrix))
        return item_score

def cosine_sim(option, matrix):
    if option=='user':
        user_cos = np.dot(np.transpose(matrix),matrix)
        return user_cos

    elif option=='item':
        item_cos = np.dot(matrix,np.transpose(matrix))
        return item_cos

def train_matrix(normalize):
    path = "data/train.csv"
    data = []
    item = []
    user = []
    with open(path,'r') as f:
        input_data = csv.reader(f)
        for i in input_data:
            item.append(int(i[0]))
            user.append(int(i[1]))
            data.append(float(i[2]) - normalize)
    matrix = sp.csr_matrix((data, (item, user)))
    return matrix

def result(path):
    with open(path, 'r') as f:
        data = csv.reader(f)
        golden_data = []
        for i in data:
            golden_data += i
        golden_data = np.asarray(golden_data,dtype=float)
    return golden_data



def PMFrecord(path, inputs):
    # movies, users = output_helper()
    # prediction_score_for_UserID13_to_MovieID1
    with open(path, 'w') as f:
        for i in range(0, len(inputs)):
            # user = str(users[i])
            # movie = str(movies[i])
            # user = users[i]
            # movie = movies[i]
            if inputs[i] < 1:
                # f.write("%s_for_%s_to_%s\n" % (1.00,user,movie))
                f.write("%s\n" % 1.00)
            elif inputs[i] > 5:
                f.write("%s\n" % 5.00)
            else:
                f.write("%s\n" % inputs[i])







