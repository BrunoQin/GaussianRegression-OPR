import GPy
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
start_ave = np.mean(start, axis=0)
start = start - start_ave

predict[predict == -1.0E20] = 'nan'
predict_ave = np.mean(predict, axis=0)
predict = predict - predict_ave

start = start[:, 0:1]
predict = predict[:, 0:1]

# poor `Sparse' GP Fit
noise_var = 0.05

# Z = np.hstack((np.linspace(2.5, 4., 3), np.linspace(7, 8.5, 3)))[:, None]
Z = np.random.rand(100, 1)*100
m = GPy.models.SparseGPRegression(start, predict - start, Z=Z)
m.likelihood.variance = noise_var

m.inducing_inputs.fix()
m.optimize('bfgs')

m.plot()

plt.show()
