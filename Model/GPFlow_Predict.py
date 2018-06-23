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

start = start[:, 0:2]
predict = predict[:, 0:2]

m = gpflow.saver.Saver().load('./model/multioutput.mdl')

pX = [start[1]]
print(pX)
pY, pYv = m.predict_f(pX)
print(pY)
print(pYv)
print(predict[1])
