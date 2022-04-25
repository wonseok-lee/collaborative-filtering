from src import utils
import time
import numpy as np
from src import useruser_experiment
from src import moviemovie_experiment
from src import useruser_pcc
from src import moviemovie_pcc
from src import PMF


def evaluation():
    path = 'data/dev.csv'

    k_list = [10, 100, 500]
    result = utils.result("eval/dev.golden")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        user_cos_mean,rating = useruser_experiment.user_cos_mean(path, k)
        end.append(time.time())
        print("RMSE :",np.sqrt(np.mean(np.square(user_cos_mean-result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        user_cos_weigthedmean,rating  = useruser_experiment.user_cos_weigthedmean(path, k)
        end.append(time.time())
        print("RMSE :",np.sqrt(np.mean(np.square(user_cos_weigthedmean-result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        user_dot_mean,rating  = useruser_experiment.user_dot_mean(path, k)
        end.append(time.time())
        print("RMSE :",np.sqrt(np.mean(np.square(user_dot_mean-result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    print("movie start")

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        movie_cos_mean,rating  = moviemovie_experiment.movie_cos_mean(path, k)
        end.append(time.time())
        print("RMSE :",np.sqrt(np.mean(np.square(movie_cos_mean-result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        movie_cos_weigthedmean,rating = moviemovie_experiment.movie_cos_weigthedmean(path, k)
        end.append(time.time())
        print("RMSE :",np.sqrt(np.mean(np.square(movie_cos_weigthedmean-result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        movie_dot_mean,rating  = moviemovie_experiment.movie_dot_mean(path, k)
        end.append(time.time())
        print("RMSE :",np.sqrt(np.mean(np.square(movie_dot_mean-result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    print("pcc user start")

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        user_cos_mean_pcc, rating = useruser_pcc.user_cos_mean_pcc(path, k)
        end.append(time.time())
        print("RMSE :", np.sqrt(np.mean(np.square(user_cos_mean_pcc - result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        user_cos_weigthedmean_pcc, rating = useruser_pcc.user_cos_weigthedmean_pcc(path, k)
        end.append(time.time())
        print("RMSE :", np.sqrt(np.mean(np.square(user_cos_weigthedmean_pcc - result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        user_dot_mean_pcc, rating = useruser_pcc.user_dot_mean_pcc(path, k)
        end.append(time.time())
        print("RMSE :", np.sqrt(np.mean(np.square(user_dot_mean_pcc - result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    print("pcc movie start")

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        movie_cos_mean_pcc, rating = moviemovie_pcc.movie_cos_mean_pcc(path, k)
        end.append(time.time())
        print("RMSE :", np.sqrt(np.mean(np.square(movie_cos_mean_pcc - result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        movie_cos_weigthedmean_pcc, rating = moviemovie_pcc.movie_cos_weigthedmean_pcc(path, k)
        end.append(time.time())
        print("RMSE :", np.sqrt(np.mean(np.square(movie_cos_weigthedmean_pcc - result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    for k in k_list:
        start = []
        end = []
        start.append(time.time())
        movie_dot_mean_pcc, rating = moviemovie_pcc.movie_dot_mean_pcc(path, k)
        end.append(time.time())
        print("RMSE :", np.sqrt(np.mean(np.square(movie_dot_mean_pcc - result))))
        print(end[0] - start[0])

    print("\n")
    print("------------------------------------------")
    print("\n")

    print("PMF begin")

    print("\n")
    print("------------------------------------------")
    print("\n")

    latent_list = [2, 5, 10, 20]

    for k in latent_list:
        start = []
        end = []
        start.append(time.time())
        iteration = 2000
        start.append(time.time())
        pmf = PMF.pmf(k, iteration)
        end.append(time.time())
        print(end[0] - start[0])

    d = 50
    U, V = PMF.pmf(d, 2000)

    test_data = utils.input_data('data/test.csv')
    dev_data = utils.input_data('data/dev.csv')
    score_test = PMF.score(U, V, test_data)
    score_dev = PMF.score(U, V, dev_data)

    utils.PMFrecord('test-predictions.txt', score_test)
    utils.PMFrecord('dev-predictions.txt', score_dev)

#
#
# path = 'data/dev.csv'
#
# k_list = [10, 100, 500]
# result = utils.result("eval/dev.golden")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     user_cos_mean, rating = useruser_experiment.user_cos_mean(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(user_cos_mean - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     user_cos_weigthedmean, rating = useruser_experiment.user_cos_weigthedmean(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(user_cos_weigthedmean - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     user_dot_mean, rating = useruser_experiment.user_dot_mean(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(user_dot_mean - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# print("movie start")
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     movie_cos_mean, rating = moviemovie_experiment.movie_cos_mean(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(movie_cos_mean - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     movie_cos_weigthedmean, rating = moviemovie_experiment.movie_cos_weigthedmean(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(movie_cos_weigthedmean - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     movie_dot_mean, rating = moviemovie_experiment.movie_dot_mean(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(movie_dot_mean - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# print("pcc user start")
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     user_cos_mean_pcc, rating = useruser_pcc.user_cos_mean_pcc(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(user_cos_mean_pcc - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     user_cos_weigthedmean_pcc, rating = useruser_pcc.user_cos_weigthedmean_pcc(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(user_cos_weigthedmean_pcc - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     user_dot_mean_pcc, rating = useruser_pcc.user_dot_mean_pcc(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(user_dot_mean_pcc - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# print("pcc movie start")
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     movie_cos_mean_pcc, rating = moviemovie_pcc.movie_cos_mean_pcc(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(movie_cos_mean_pcc - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     movie_cos_weigthedmean_pcc, rating = moviemovie_pcc.movie_cos_weigthedmean_pcc(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(movie_cos_weigthedmean_pcc - result))))
#     print(end[0] - start[0])
#
# print("\n")
# print("------------------------------------------")
# print("\n")
#
# for k in k_list:
#     start = []
#     end = []
#     start.append(time.time())
#     movie_dot_mean_pcc, rating = moviemovie_pcc.movie_dot_mean_pcc(path, k)
#     end.append(time.time())
#     print("RMSE :", np.sqrt(np.mean(np.square(movie_dot_mean_pcc - result))))
#     print(end[0] - start[0])
#
# latent_list = [2, 5, 10, 20]
#
# for k in latent_list:
#     start = []
#     end = []
#     start.append(time.time())
#     iteration = 2000
#     pmf = PMF.pmf(k, iteration)




