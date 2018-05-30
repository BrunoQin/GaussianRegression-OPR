import GPy
import numpy as np
import gzip
import pickle
# import matplotlib; matplotlib.rcParams['figure.figsize'] = (8, 6)
# from matplotlib import pyplot as plt


with gzip.open('./data/predict.pkl.gz') as fp:
    predict = np.array(pickle.load(fp))

with gzip.open('./data/start.pkl.gz') as fp:
    start = np.array(pickle.load(fp))


K = GPy.kern.Matern32(1)
icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=predict.shape[1], kernel=K)

m = GPy.models.GPCoregionalizedRegression(start, predict, kernel=icm)
m['.*Mat32.var'].constrain_fixed(1.)
m.optimize()
print(m)

m.optimize(messages=True)
