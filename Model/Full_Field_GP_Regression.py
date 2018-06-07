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

start = start[:, 0:2]
predict = predict[:, 0:2]

input = [start[:, i].reshape(start.shape[0], 1) for i in range(len(start[0]))]
output = [(predict[:, i]-start[:, i]).reshape(start.shape[0], 1) for i in range(len(start[0]))]


def plot_2outputs(m, xlim, ylim):
    fig = plt.figure(figsize=(12, 8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim, fixed_inputs=[(1, 0)], which_data_rows=slice(0, 100), ax=ax1)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim, fixed_inputs=[(1, 1)], which_data_rows=slice(100, 200), ax=ax2)


K = GPy.kern.Matern32(1)
icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=2, kernel=K)

m = GPy.models.GPCoregionalizedRegression(input, output, kernel=icm)
m['.*Mat32.var'].constrain_fixed(1.)
m.optimize()

plot_2outputs(m, xlim=(-5, 5), ylim=(-5, 5))
plt.show()
