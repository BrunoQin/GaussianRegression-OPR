import GPy
from sklearn import preprocessing
import multiprocessing as mp
import numpy as np
import gzip
import pickle
from matplotlib import pyplot as plt

# read data
with gzip.open('./data/predict.pkl.gz') as fp:
    predict = np.array(pickle.load(fp))

with gzip.open('./data/start.pkl.gz') as fp:
    start = np.array(pickle.load(fp))

# clean data
start[start == -1.0E20] = 'nan'
start = start[:, ~np.all(np.isnan(start), axis=0)]

predict[predict == -1.0E20] = 'nan'
predict = predict[:, ~np.all(np.isnan(predict), axis=0)]

# preprocess
start = preprocessing.scale(start)
predict = preprocessing.scale(predict)


def train(s, p):
    input = [s[:, i].reshape(s.shape[0], 1) for i in range(len(s[0]))]
    output = [(p[:, i]-s[:, i]).reshape(s.shape[0], 1) for i in range(len(s[0]))]

    k = GPy.kern.Matern32(1)
    icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=start.shape[1], kernel=k)
    m = GPy.models.GPCoregionalizedRegression(input, output, kernel=icm)
    m['.*Mat32.var'].constrain_fixed(1.)
    m.optimize(messages=True, max_f_eval=1000)

    np.save('GP_Regression.npy', m.param_array)


# pool = mp.Pool()
# res = pool.starmap(train, [(start[:, 0:2], predict[:, 0:2])])
# pool.close()
# pool.join()
train(start[:, 0:2], predict[:, 0:2])

plt.show()
