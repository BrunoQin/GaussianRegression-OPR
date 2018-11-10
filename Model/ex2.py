import numpy as np
import tensorflow as tf
import gpflow

import myKernel.convkernels as ckern
from myKernel.NN import NN
from myKernel.LoadData import LoadData

float_type = gpflow.settings.dtypes.float_type
ITERATIONS = 1000

class Ocean:
    input_dim = 150 * 160
    out_dim = 1
    X, Y, Xt, Yt = LoadData.load_ocean()


def ex2():
    minibatch_size = 200
    gp_dim = 15*16
    M = 32

    ## placeholders
    X = tf.placeholder(tf.float32, [minibatch_size, Ocean.input_dim])  # fixed shape so num_data works in SVGP
    Y = tf.placeholder(tf.float32, [minibatch_size, 1])
    Xtest = tf.placeholder(tf.float32, [None, Ocean.input_dim])

    ## build graph
    with tf.variable_scope('cnn'):
        f_X = tf.cast(NN.cnn_fn(X, gp_dim), dtype=float_type)

    with tf.variable_scope('cnn', reuse=True):
        f_Xtest = tf.cast(NN.cnn_fn(Xtest, gp_dim), dtype=float_type)

    k = ckern.WeightedConv(gpflow.kernels.RBF(9), [15, 16], [3, 3]) + gpflow.kernels.White(1, 1e-3)

    # Z = None
    # if Z is None:
    #     Z = (k.kernels[0].init_inducing(f_X, minibatch_size, M, method="patches-unique")
    #          if type(k) is gpflow.kernels.Sum else
    #          k.init_inducing(f_X, minibatch_size, M, method="patches-unique"))

    gp_model = gpflow.models.SVGP(f_X, tf.cast(Y, dtype=float_type),
                                k, gpflow.likelihoods.Gaussian(),
                                Z=np.zeros((M, gp_dim)), # we'll set this later
                                num_latent=1)

    loss = -gp_model.likelihood_tensor

    m, v = gp_model._build_predict(f_Xtest)
    my, yv = gp_model.likelihood.predict_mean_and_var(m, v)

    with tf.variable_scope('adam'):
        opt_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adam')
    tf_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn')

    ## initialize
    sess = tf.Session()
    sess.run(tf.variables_initializer(var_list=tf_vars))
    gp_model.initialize(session=sess)

    ## reset inducing (they live in a different space as X, so need to be careful with this)
    ind = np.random.choice(Ocean.X.shape[0], minibatch_size, replace=False)

    fZ = sess.run(f_X, feed_dict={X:Ocean.X[ind]})
    # Z_0 = kmeans2(fZ, M)[0] might fail
    Z_0 = fZ[np.random.choice(len(fZ), M, replace=True)]

    def set_gp_param(param, value):
        sess.run(tf.assign(param.unconstrained_tensor, param.transform.backward(value)))

    set_gp_param(gp_model.feature.Z, Z_0)

    ## train
    for i in range(ITERATIONS):
        ind = np.random.choice(Ocean.X.shape[0], minibatch_size, replace=True)
        sess.run(opt_step, feed_dict={X:Ocean.X[ind], Y:Ocean.Y[ind]})
        print('step {:.4f}'.format(i))
        if i % 10 == 0:
            rmse = np.mean((sess.run(my, feed_dict={Xtest:Ocean.Xt}) - Ocean.Yt.shape) ** 2) * 0.5
            rmse = np.average(rmse.astype(float))
            print('rmse is {:.4f}'.format(rmse))

    ## predict
    rmse = np.mean((sess.run(my, feed_dict={Xtest:Ocean.Xt}) - Ocean.Yt.shape) ** 2) * 0.5
    rmse = np.average(rmse.astype(float)) * 100.
    print('rmse is {:.4f}'.format(rmse))


ex2()