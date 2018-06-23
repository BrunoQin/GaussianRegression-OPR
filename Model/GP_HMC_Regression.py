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

m = GPy.models.GPRegression(start, predict - start)

m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))

hmc = GPy.inference.mcmc.HMC(m, stepsize=5e-2)

# hmc = GPy.inference.optimization.HMC(m, stepsize=5e-1)

m.plot()

# s = hmc.sample(num_samples=300)
#
# samples = s[300:]
# m.kern.variance[:] = samples[:, 0].mean()
# m.kern.lengthscale[:] = samples[:, 1].mean()
# m.likelihood.variance[:] = samples[:, 2].mean()
# m.plot()

plt.show()
