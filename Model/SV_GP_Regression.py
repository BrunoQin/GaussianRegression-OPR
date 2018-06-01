import GPy
import numpy as np
import gzip
import pickle
import climin
import sys
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

Z = np.random.rand(20, 1)

batchsize = 10
m = GPy.core.SVGP(start, predict - start, Z, GPy.kern.RBF(1) + GPy.kern.White(1), GPy.likelihoods.Gaussian(), batchsize=batchsize)
m.kern.white.variance = 1e-5
m.kern.white.fix()

opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)


def callback(i):
    t = str(m.log_likelihood())
    # Stop after 5000 iterations
    if i['n_iter'] > 5000:
        return True
    return False


info = opt.minimize_until(callback)
print(info)

# m.plot()
#
# plt.show()

ax = plt.gca()
ax.plot(start, predict - start, 'kx', alpha=0.1)
ax.set_xlabel('start')
ax.set_ylabel('predict - start')
ax.set_title('SVI Y1 prediction with data')
_ = m.plot(which_data_ycols=[0], plot_limits=(start.min(), start.max()), ax=ax)
ax.set_xlim((start.min(), start.max()))

plt.show()
