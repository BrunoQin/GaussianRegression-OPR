import os
import gpflow
from sklearn import preprocessing
import numpy as np
import gzip
import pickle
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


M = 3
start = start[0:10, 0:M]
predict = predict[0:10, 0:M]

m = gpflow.saver.Saver().load('./model/gp.mdl')

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


print(X_augmented)
pY, pYv = m.predict_f(X_augmented)
print(pY)
# print(pYv)
print(Y_augmented)
# print(pY - Y_augmented[:, 0])
