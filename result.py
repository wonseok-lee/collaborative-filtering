import numpy as np
from src import utils
from src import Evaluation


trainmatrix = utils.train_matrix(0).toarray()

train = utils.input_data('data/train.csv')

movie = []
user = []
for i in train:
    movieID = i[0]
    userID = i[1]
    movie.append(movieID)
    user.append(userID)

a = len(set(movie))
b = len(set(user))

print('the the number of movies :', a, '\n')
print('the the number of users :', b, '\n')
one = np.where(trainmatrix == 1)
print('# of  1: ', one[0].size, '\n')
two = np.where(trainmatrix == 3)
print( '# of  3: ', two[0].size, '\n')
three = np.where(trainmatrix == 5)
print( '# of 5: ', three[0].size, '\n')
four = np.sum(trainmatrix) / np.count_nonzero(trainmatrix)
print( 'average: ', four, '\n')

user1 = trainmatrix[:, 4321]
movie1 = np.count_nonzero(user1)
print( 'number of movie rated: ', movie1, '\n')
five = np.where(user1 == 1)
print( '# of 1: ', five[0].size, '\n')
six = np.where(user1 == 3)
print( '# of 3: ', six[0].size, '\n')
seven = np.where(user1 == 5)
print( '# of 5: ', seven[0].size, '\n')
eight = np.sum(user1) / np.count_nonzero(user1)
print( 'average: ', eight, '\n')

movie2 = trainmatrix[3, :]
user_num = np.count_nonzero(movie2)
print( 'number of user rated: ', user_num, '\n')
oo = np.where(movie2 == 1)
print( '# of 1: ', oo[0].size, '\n')
pp = np.where(movie2 == 3)
print( '# of 3: ', pp[0].size, '\n')
qq = np.where(movie2 == 5)
print( '# of 5: ', qq[0].size, '\n')
ww = np.sum(movie2) / np.count_nonzero(movie2)
print( 'average: ', ww, '\n')

eval = Evaluation.evaluation()

