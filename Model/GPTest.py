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
start[start == -1.0E20] = 0
start_ave = np.mean(start, axis=0)
start = start - start_ave

predict[predict == -1.0E20] = 0
predict_ave = np.mean(predict, axis=0)
predict = predict - predict_ave


def plot_2outputs(xlim, ylim):
    plt.scatter(xlim, ylim, alpha=0.5)   # s为size，按每个点的坐标绘制，alpha为透明度
    plt.xlim(0, 300)
    plt.ylim(-4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_line(xlim, ylim):
    plt.plot(xlim, ylim)
    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.show()


# plot_2outputs(start[:, 0:1].T, predict[:, 0:1].T)
# plot_2outputs(start[:, 0:1].T, (predict[:, 0:1] - start[:, 0:1]).T)
plot_line(np.linspace(0, 300, 300).reshape(1, 300), start[:, 0:1].T)


# # build model
# K = GPy.kern.Matern32(1)
# icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=1, kernel=K)
#
# kernel = GPy.kern.Matern52(1, ARD=True) + GPy.kern.White(1)
#
# kernel = GPy.kern.MLP(1) + \
#          GPy.kern.White(1) + \
#          GPy.kern.Bias(1)\
#            # * GPy.kern.Coregionalize(1, output_dim=1, rank=1, active_dims=1, name='gender')
#            # * GPy.kern.Coregionalize(1, output_dim=3, rank=1, active_dims=2, name='event')
#
# print(np.arange(300).reshape(300, 1).shape)
# m = GPy.models.GPRegression(np.arange(300).reshape(300, 1), start[:, 0:1], kernel)
#
# print(m)
#
# m.optimize(messages=True, max_f_eval=1000)
#
# fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
#
# plt.show()
#
# newX = start[:, 0:1]
# newY = m.predict(newX)
# newY = np.array(newY)
# print(newY.shape)
# print(np.mean(newY, axis=1))
