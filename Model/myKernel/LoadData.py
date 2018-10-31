import gzip
import pickle
import numpy as np


class LoadData():

    @staticmethod
    def load_ocean():
        # read data
        with gzip.open('OCEAN_data/redata.pkl.gz') as fp:
            redata = np.array(pickle.load(fp)).astype(float)

        with gzip.open('OCEAN_data/nino.pkl.gz') as fp:
            nino = np.array(pickle.load(fp)).astype(float)

        # ALL = None
        # for i in redata:
        #     i.reshape(150, 160)
        #     i = scipy.ndimage.zoom(i.reshape(150, 160), 0.2)
        #     i = i.reshape(1, 960)
        #     if ALL is None:
        #         ALL = i
        #     else:
        #         ALL = np.vstack((ALL, i))
        X = redata[0:4600]
        Y = nino[11:4611]
        Xt = redata[4601:4789]
        Yt = nino[4612:4800]
        return X, Y, Xt, Yt
