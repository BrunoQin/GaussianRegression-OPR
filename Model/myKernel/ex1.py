import numpy as np
import tensorflow as tf
import gpflow
from scipy.cluster.vq import kmeans2

from myKernel.KernelWithNN import KernelWithNN
from myKernel.KernelWithNN import NNComposedKernel
from myKernel.KernelWithNN import NN_SVGP

from myKernel.NN import NN

from myKernel.LoadData import LoadData

float_type = gpflow.settings.float_type


def ex1():
    fX_dim = 5
    M = 100
    X, Y, Xt, Yt = LoadData.load_ocean()

    # annoyingly only float32 and lower is supported by the conv layers
    f = lambda x: tf.cast(NN.cnn_fn(tf.cast(x, tf.float32), fX_dim), float_type)
    kern = NNComposedKernel(gpflow.kernels.Matern32(fX_dim), f)

    # build the model
    lik = gpflow.likelihoods.Gaussian()

    Z = kmeans2(X, M, minit='points')[0]

    model = NN_SVGP(X, Y, kern, lik, Z=Z, minibatch_size=200)

    # use gpflow wrappers to train. NB all session handling is done for us
    gpflow.training.AdamOptimizer(0.001).minimize(model, maxiter=30000)

    # predictions
    pY, pYv = model.predict_y(Xt)
    rmse = np.mean((pY - Yt) ** 2.0) ** 0.5
    nlpp = -np.mean(-0.5 * np.log(2 * np.pi * pYv) - 0.5 * (Yt - pY) ** 2.0 / pYv)

    print('rmse is {:.4f}%, nlpp is {:.f}%'.format(rmse, nlpp))


if __name__ == "__main__":
    ex1()
    gpflow.reset_default_graph_and_session()
