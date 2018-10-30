import GPy
from sklearn import preprocessing
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
start = start[:, ~np.all(np.isnan(start), axis=0)]
# start_ave = np.mean(start, axis=0)
# start = start - start_ave

predict[predict == -1.0E20] = 'nan'
predict = predict[:, ~np.all(np.isnan(predict), axis=0)]
# predict_ave = np.mean(predict, axis=0)
# predict = predict - predict_ave

start = preprocessing.scale(start)
predict = preprocessing.scale(predict)

start = start[:, 0:2]
predict = predict[:, 0:2]

input = [start[:, i].reshape(start.shape[0], 1) for i in range(len(start[0]))]
output = [(predict[:, i]-start[:, i]).reshape(start.shape[0], 1) for i in range(len(start[0]))]

K = GPy.kern.Matern32(1)
icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=2, kernel=K)

m_load = GPy.models.GPCoregionalizedRegression(input, output, kernel=icm)
m_load.update_model(False)                          # do not call the underlying expensive algebra on load
m_load.initialize_parameter()                       # Initialize the parameters (connect the parameters up)
m_load[:] = np.load('GP_Regression.npy')            # Load the parameters
m_load.update_model(True)                           # Call the algebra only once


print(start[0])
print(predict[0])
newX = np.array([[4.496155, 0], [4.4616776, 1]], dtype=float)
# print(newX)
# newX = np.hstack([newX, np.ones_like(newX)])
# print(newX)
noise_dict = {'output_index': newX[:, 1:].astype(int)}
Y = m_load.predict(newX, Y_metadata=noise_dict)

print(Y)
