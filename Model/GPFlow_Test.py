import os
import gpflow
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import gzip
import pickle
from matplotlib import pyplot as plt
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


with gpflow.defer_build():
    M = 2
    k1 = gpflow.kernels.Matern32(1, active_dims=[0])
    coreg = gpflow.kernels.Coregion(1, output_dim=M, rank=M, active_dims=[1])
    kern = k1 * coreg
    # construct kernel
    like_array = [gpflow.likelihoods.StudentT(), gpflow.likelihoods.StudentT()]
    for i in range(M-2):
        like_array.append(gpflow.likelihoods.StudentT())
    lik = gpflow.likelihoods.SwitchedLikelihood(like_array)
    # construct x
    X_augmented = np.vstack(
        (np.hstack((start[:, 0:1], np.zeros_like(start[:, 0:1]))), np.hstack((start[:, 1:2], np.ones_like(start[:, 1:2]))))
    )
    for i in range(M-2):
        X_augmented = np.vstack(
            (X_augmented, np.hstack((start[:, i+2:i+3], (i+2)*np.ones_like(start[:, i+2:i+3]))))
        )
    # construct y
    Y_augmented = np.vstack(
        (np.hstack((predict[:, 0:1], np.zeros_like(start[:, 0:1]))), np.hstack((predict[:, 1:2], np.ones_like(start[:, 1:2]))))
    )
    for i in range(M-2):
        Y_augmented = np.vstack(
            (Y_augmented, np.hstack((predict[:, i+2:i+3], (i+2)*np.ones_like(start[:, i+2:i+3]))))
        )
    # construct model
    m = gpflow.models.VGP(X_augmented, Y_augmented, kern=kern, likelihood=lik, num_latent=1)

tf.local_variables_initializer()
tf.global_variables_initializer()
tf_session = m.enquire_session()
m.compile(tf_session)
gpflow.train.ScipyOptimizer().minimize(m)
saver = tf.train.Saver()
save_path = saver.save(tf_session, "./model/model.ckpt")
print("Model saved in path: %s" % save_path)
