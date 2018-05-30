import pandas as pd
import numpy as np
import gzip
import pickle

start_month = []
for i in range(300):
    filename = "./meta/start/" + str(i + 51) + ".start.csv"
    df = pd.read_csv(filename, header=None, dtype=np.float64).values
    data = df.flatten()
    start_month.append(data)

start_month = np.array(start_month)
with gzip.open('./meta/start.pkl.gz', 'wb') as f:
    f.write(pickle.dumps(start_month))

print(start_month.shape)

predict_month = []
for i in range(300):
    filename = "./meta/predict/" + str(i + 51) + ".predict.csv"
    df = pd.read_csv(filename, header=None, dtype=np.float64).values
    data = df.flatten()
    predict_month.append(data)

predict_month = np.array(predict_month)
with gzip.open('./meta/predict.pkl.gz', 'wb') as f:
    f.write(pickle.dumps(predict_month))

print(predict_month.shape)
