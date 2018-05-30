import GPy
import numpy as np
import gzip
import pickle
import matplotlib; matplotlib.rcParams['figure.figsize'] = (8, 6)
from matplotlib import pyplot as plt

with gzip.open('./data/predict.pkl.gz') as fp:
    predict = pickle.load(fp)

predict = np.array(predict)
print(predict.shape)

with gzip.open('./data/start.pkl.gz') as fp:
    start = pickle.load(fp)

start = np.array(start)
print(start.shape)

# X = np.random.uniform(0, 6., (20, 1))
# Y = np.log(X) + np.random.randn(20, 1)*0.05
#
# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# m = GPy.models.GPRegression(X, Y, kernel)
#
# fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
#
# m.optimize(messages=True)
#
# fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')
#
# plt.show()
