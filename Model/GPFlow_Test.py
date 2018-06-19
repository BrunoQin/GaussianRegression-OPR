import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

import gpflow
import GPy
from sklearn import preprocessing
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

k1 = gpflow.kernels.Matern32(1, active_dims=[0])
coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[1])
kern = k1 * coreg

lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.StudentT(), gpflow.likelihoods.StudentT()])

X_augmented = np.vstack((np.hstack((start[:, 0:1], np.zeros_like(start[:, 0:1]))), np.hstack((start[:, 1:2], np.ones_like(start[:, 1:2])))))
Y_augmented = np.vstack((np.hstack((predict[:, 0:1]-start[:, 0:1], np.zeros_like(start[:, 0:1]))), np.hstack((predict[:, 1:2]-start[:, 1:2], np.ones_like(start[:, 1:2])))))

m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)
gpflow.train.ScipyOptimizer().minimize(m)


def plot_gp(x, mu, var, color='k'):
    plt.plot(x, mu, color=color, lw=2)
    plt.plot(x, mu + 2*np.sqrt(var), '--', color=color)
    plt.plot(x, mu - 2*np.sqrt(var), '--', color=color)


def plot(m):
    xtest = np.linspace(0, 1, 100)[:, None]
    line, = plt.plot(start[:, 0:1], predict[:, 0:1]-start[:, 0:1], 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())

    line, = plt.plot(start[:, 1:2], predict[:, 1:2]-start[:, 1:2], 'x', mew=2)
    mu, var = m.predict_f(np.hstack((xtest, np.ones_like(xtest))))
    plot_gp(xtest, mu, var, line.get_color())


plot(m)
plt.show()
