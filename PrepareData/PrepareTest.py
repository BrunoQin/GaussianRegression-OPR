import pandas as pd
import numpy as np
import gzip
import pickle

data = []
for i in range(400):
    for j in range(12):
        try:
            filename = "/home/dl/Desktop/QinBo/data/" + str(i + 51) + "." + str(j + 1) + ".csv"
            df = pd.read_csv(filename, header=None, dtype=np.float64).values
            df = df.flatten()
            data.append(df)
        except FileNotFoundError:
            print(i, j)

data = np.array(data)
print(data.shape)
with gzip.open('/home/dl/Desktop/QinBo/data.pkl.gz', 'wb') as f:
    f.write(pickle.dumps(data))
