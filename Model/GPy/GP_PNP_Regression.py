import GPy
import numpy as np
import gzip
import pickle
from GPy.kern import LinearSlopeBasisFuncKernel, DomainKernel, ChangePointBasisFuncKernel
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

k = GPy.kern.Matern32(1, .3)
Kf = k.K(start)

starts, stops = np.arange(np.min(start), np.max(start), 100), np.arange(np.min(start) + 1, np.max(start) + 1, 100)

k_per = GPy.kern.PeriodicMatern32(1, variance=100, period=1)
k_per.period.fix()
k_dom = DomainKernel(1, 1., 5.)
k_perdom = k_per * k_dom
Kpd = k_perdom.K(start)

k = (GPy.kern.Bias(1)
     + GPy.kern.Matern52(1)
     + LinearSlopeBasisFuncKernel(1, ARD=1, start=starts, stop=stops, variance=.1, name='linear_slopes')
     + k_perdom.copy()
     )

k.randomize()
m = GPy.models.GPRegression(start, predict - start, k)
m.checkgrad()
m.optimize()
m.plot()

plt.show()
