from src import utils
import numpy as np
import torch
from copy import deepcopy
import time


# def pmf_record(path, data):
#     with open(path, 'w') as f:
#         for i in range(0, len(data)):
#             if data[i] < 1:
#                 f.write("%s\n" % 1.00)
#             elif data[i] > 5:
#                 f.write("%s\n" % 5.00)
#             else:
#                 f.write("%s\n" % data[i])

def score(U, V, data):
    data_0 = data.take(0, axis=1)
    data_1 = data.take(1, axis=1)
    a = V.numpy().take(data_0, axis=0)
    b = U.numpy().take(data_1, axis=0)
    score = np.sum(a*b, 1)
    return score

def pmf(latent, iteration):
    global U_last
    global V_last

    data = utils.input_data('data/dev.csv')
    result = utils.result('eval/dev.golden')
    matrix = utils.train_matrix(0).toarray()

    matrix = np.transpose(matrix)
    item_dim = len(matrix[0])
    user_dim = len(matrix)
    lambda_U = 0.1
    lambda_V = 0.1

    I = deepcopy(matrix)
    I[I != 0] = 1
    I = torch.from_numpy(I).double()
    matrix = torch.from_numpy(matrix).double()

    moment = 0.8
    lr = 0.0001

    U = 0.01 * torch.rand(user_dim, latent).double()
    V = 0.01 * torch.rand(item_dim, latent).double()
    m_U = torch.zeros(U.shape).double()
    m_V = torch.zeros(V.shape).double()

    initial_loss = 3.5

    start = time.time()
    for i in range(iteration):
        U_tV = torch.matmul(U, torch.transpose(V, 1, 0))
        grad_u = torch.matmul(I * (matrix - U_tV), -V) + lambda_U * U
        grad_v = torch.matmul(torch.transpose(I * (matrix - U_tV), 1, 0), -U) + lambda_V * V
        m_U = moment * m_U + lr * grad_u
        m_V = moment * m_V + lr * grad_v
        U = U - m_U
        V = V - m_V
        initial_score = score(U, V, data)
        rmse = np.sqrt(np.mean(np.square(initial_score - result)))

        print('RMSE : ', rmse)

        if 0 < initial_loss - rmse < 10e-6:
            print('done')
            # U_last = U
            # V_last = V
            break

        if rmse < 0.92:
            # U_last = U
            # V_last = V
            break
        initial_loss = rmse

    end = time.time()
    tooktime = end - start
    print('time :', tooktime)

    return U, V

    # test_data = utils.input_data('data/test.csv')
    # dev_data = utils.input_data('data/dev.csv')
    # score_test = score(U_last, V_last, test_data)
    # score_dev = score(U_last, V_last, dev_data)
    #
    # return score_test, score_dev



# pmf_record('test-predictions.txt', score_test)
# pmf_record('dev-predictions.txt', score_dev)