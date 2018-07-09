import os
from sklearn import preprocessing
import numpy as np
import gzip
import pickle
import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

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
predict = predict-start

# start = start[:, 0:2]
# predict = predict[:, 0:2]

D = start.shape[1]  # number of input dimensions
M = 20  # number of inducing points
L = 1  # number of latent GPs
P = predict.shape[1]  # number of observations = output dimensions
MAXITER = gpflow.test_util.notebook_niter(int(1e100))

q_mu = np.zeros((M, L))
q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

kern_list = [gpflow.kernels.RBF(D) + gpflow.kernels.Linear(D) for _ in range(L)]
kernel = mk.SeparateMixedMok(kern_list, W=np.random.randn(P, L))
feature = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(start[:M, ...].copy()))
m = gpflow.models.SVGP(start, predict, kernel, gpflow.likelihoods.Gaussian(), feat=feature, q_mu=q_mu, q_sqrt=q_sqrt)
opt = gpflow.train.ScipyOptimizer()
opt.minimize(m, disp=True, maxiter=MAXITER)

saver = gpflow.Saver()
saver.save('./model/multioutput.mdl', m)
