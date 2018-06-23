import os
import gpflow
from sklearn import preprocessing
import numpy as np
import gzip
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error

m = gpflow.saver.Saver().load('./model/gp')

xtest = np.linspace(0, 1, 10)[:, None]
print(np.hstack((xtest, np.zeros_like(xtest))))
mu1, var1 = m.predict_f(np.hstack((xtest, np.zeros_like(xtest))))
print(mu1)
print(var1)
