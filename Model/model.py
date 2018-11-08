import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, kernels, features

from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import SVGP_Layer
from myKernel.kernels import ConvKernel, PatchInducingFeatures, AdditivePatchKernel
from myKernel.layers import ConvLayer
from myKernel.views import FullView, RandomPartialView
from myKernel.mean_functions import Conv2dMean, IdentityConv2dMean
from sklearn import cluster

float_type = gpflow.settings.float_type

def image_HW(patch_count):
    image_height = int(np.sqrt(patch_count))
    return [image_height, image_height]

def select_initial_inducing_points(X, M):
    kmeans = cluster.KMeans(n_clusters=M, init='k-means++', n_jobs=-1)
    kmeans.fit(X)
    return kmeans.cluster_centers_

def identity_conv(NHWC_X, filter_size, feature_maps_in, feature_maps_out, stride):
    conv = IdentityConv2dMean(filter_size, feature_maps_in, feature_maps_out, stride)
    sess = conv.enquire_session()
    random_images = np.random.choice(np.arange(NHWC_X.shape[0]), size=1000)
    return sess.run(conv(NHWC_X[random_images]))


class ModuleBuilder(object):
    def __init__(self, flags, NHWC_X_train, Y_train, NHWC_X_test, Y_test, model_path=None):
        self.flags = flags
        self.X_train = NHWC_X_train
        self.Y_train = Y_train
        self.X_test = NHWC_X_test
        self.Y_test = Y_test
        self.model_path = model_path
        self.global_step = None

    def build(self):
        Ms = [] # the number of inducing point
        simple_feature_maps = []
        simple_conv_sizes = []
        simple_conv_strides = []
        simple_pool_sizes = []
        simple_pool_strides = []

        gp_feature_maps = []
        gp_strides = []
        gp_filter_sizes = []
        layers = []

        X = tf.placeholder(tf.float32, [32, self.X_train.shape[1]])  # fixed shape so num_data works in SVGP
        Y = tf.placeholder(tf.float32, [32, 1])
        Xtest = tf.placeholder(tf.float32, [None, self.X_train.shape[1]])

        with tf.variable_scope('cnn'):
            f_X = tf.cast(self._simple_conv_layers(self.X_train, simple_feature_maps, simple_conv_sizes, simple_conv_strides, simple_pool_sizes, simple_pool_strides),
                          dtype=float_type)

        with tf.variable_scope('cnn', reuse=True):
            f_Xtest = tf.cast(self._simple_conv_layers(self.X_test, simple_feature_maps, simple_conv_sizes, simple_conv_strides, simple_pool_sizes, simple_pool_strides),
                              dtype=float_type)

        H_X = self._simple_conv_layers(simple_feature_maps, simple_conv_sizes, simple_conv_strides, simple_pool_sizes, simple_pool_strides)
        gp_conv_layers, HX = self._gp_conv_layers(H_X, Ms[0:-1], gp_feature_maps, gp_strides, gp_filter_sizes)
        last_layer = self._last_layer(H_X, Ms[-1], gp_filter_sizes[-1], gp_strides[-1])

        layers = gp_conv_layers + [last_layer]
        gp_model = DGP_Base(self.X_train, self.Y_train,
                            likelihood=gpflow.likelihoods.Gaussian(),
                            num_samples=self.flags.num_samples,
                            layers=layers,
                            minibatch_size=self.flags.batch_size, name='DGP')
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
        ind = np.random.choice(self.X_train.shape[0], 32, replace=False)

        fZ = sess.run(f_X, feed_dict={X:self.X_test[ind]})
        # Z_0 = kmeans2(fZ, M)[0] might fail
        Z_0 = fZ[np.random.choice(len(fZ), 32, replace=False)]

        def set_gp_param(param, value):
            sess.run(tf.assign(param.unconstrained_tensor, param.transform.backward(value)))

        set_gp_param(gp_model.feature.Z, Z_0)

        ## train
        for i in range(30000):
            ind = np.random.choice(self.X_train.shape[0], 32, replace=False)
            sess.run(opt_step, feed_dict={X:self.X_train[ind], Y:self.Y_train[ind]})

        ## predict
        preds = np.argmax(sess.run(my, feed_dict={Xtest:self.X_test}), 1).reshape(self.Y_test.shape)
        # correct = preds == Mnist.Ytest.astype(int)
        # acc = np.average(correct.astype(float)) * 100.
        # print('acc is {:.4f}'.format(acc))


    def _simple_conv_layers(self, H_X, feature_maps, conv_sizes, conv_strides, pool_sizes, pool_strides):
        H_X = H_X
        for i in range(len(feature_maps)):
            feature_map = feature_maps[i]
            conv_size = conv_sizes[i]
            conv_stride = conv_strides[i]
            pool_size = pool_sizes[i]
            pool_stride = pool_strides[i]

            H_X = self._simple_conv_layer(H_X, feature_map, conv_size, conv_stride, pool_size, pool_stride)
        return H_X

    def _simple_conv_layer(self, NHWC_X, feature_map, conv_size, conv_stride, pool_size, pool_stride):
        conv = tf.layers.conv2d(
            inputs=tf.reshape(NHWC_X, [-1, 150, 160, 1]),
            filters=feature_map,
            kernel_size=conv_size,
            strides=conv_stride,
            padding="same",
            activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_size, strides=pool_stride)
        return pool

    def _gp_conv_layers(self, H_X, Ms, feature_maps, filter_sizes, strides):
        layers = []
        for i in range(len(feature_maps)):
            M = Ms[i]
            feature_map = feature_maps[i]
            filter_size = filter_sizes[i]
            stride = strides[i]

            conv_layer, H_X = self._gp_conv_layer(H_X, M, feature_map, filter_size, stride)
            layers.append(conv_layer)
        return layers, H_X

    def _gp_conv_layer(self, NHWC_X, M, feature_map, filter_size, stride, layer_params=None):
        if layer_params is None:
            layer_params = {}
        NHWC = NHWC_X.shape
        view = FullView(input_size=NHWC[1:3],
                        filter_size=filter_size,
                        feature_maps=NHWC[3],
                        stride=stride)

        if self.flags.identity_mean:
            conv_mean = Conv2dMean(filter_size, NHWC[3], feature_map,
                                   stride=stride)
        else:
            conv_mean = gpflow.mean_functions.Zero()
        conv_mean.set_trainable(False)

        output_shape = image_HW(view.patch_count) + [feature_map]

        H_X = identity_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride)
        if len(layer_params) == 0:
            conv_features = PatchInducingFeatures.from_images(
                NHWC_X,
                M,
                filter_size)
        else:
            conv_features = PatchInducingFeatures(layer_params.get('Z'))

        patch_length = filter_size ** 2 * NHWC[3]
        if self.flags.base_kernel == 'rbf':
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            base_kernel = kernels.RBF(patch_length, variance=variance, lengthscales=lengthscales)
        elif self.flags.base_kernel == 'acos':
            base_kernel = kernels.ArcCosine(patch_length, order=0)
        else:
            raise ValueError("Not a valid base-kernel value")

        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        conv_layer = ConvLayer(
            base_kernel=base_kernel,
            mean_function=conv_mean,
            feature=conv_features,
            view=view,
            white=self.flags.white,
            gp_count=feature_map,
            q_mu=q_mu,
            q_sqrt=q_sqrt)

        if q_sqrt is None:
            # Start with low variance.
            conv_layer.q_sqrt = conv_layer.q_sqrt.value * 1e-5

        return conv_layer, H_X

    def _last_layer(self, H_X, M, filter_size, stride, layer_params=None):
        if layer_params is None:
            layer_params = {}

        NHWC = H_X.shape
        conv_output_count = np.prod(NHWC[1:])
        Z = layer_params.get('Z')
        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        if Z is not None:
            saved_filter_size = int(np.sqrt(Z.shape[1] / NHWC[3]))
            if filter_size != saved_filter_size:
                print("filter_size {} != {} for last layer. Resetting parameters.".format(filter_size, saved_filter_size))
                Z = None
                q_mu = None
                q_sqrt = None

        if self.flags.last_kernel == 'rbf':
            H_X = H_X.reshape(H_X.shape[0], -1)
            lengthscales = layer_params.get('lengthscales', 5.0)
            variance = layer_params.get('variance', 5.0)
            kernel = gpflow.kernels.RBF(conv_output_count, lengthscales=lengthscales, variance=variance,
                                        ARD=True)
            if Z is None:
                Z = select_initial_inducing_points(H_X, M)
            inducing = features.InducingPoints(Z)
        else:
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            input_dim = filter_size**2 * NHWC[3]
            view = FullView(input_size=NHWC[1:],
                            filter_size=filter_size,
                            feature_maps=NHWC[3],
                            stride=stride)
            if Z is None:
                inducing = PatchInducingFeatures.from_images(H_X, M, filter_size)
            else:
                inducing = PatchInducingFeatures(Z)
            patch_weights = layer_params.get('patch_weights')
            if self.flags.last_kernel == 'conv':
                kernel = ConvKernel(
                    base_kernel=gpflow.kernels.RBF(input_dim, variance=variance, lengthscales=lengthscales),
                    view=view, patch_weights=patch_weights)
            elif self.flags.last_kernel == 'add':
                kernel = AdditivePatchKernel(
                    base_kernel=gpflow.kernels.RBF(input_dim, variance=variance, lengthscales=lengthscales),
                    view=view, patch_weights=patch_weights)
            else:
                raise ValueError("Invalid last layer kernel")
        return SVGP_Layer(kern=kernel,
                          num_outputs=1,
                          feature=inducing,
                          mean_function=gpflow.mean_functions.Zero(output_dim=1),
                          white=self.flags.white,
                          q_mu=q_mu,
                          q_sqrt=q_sqrt)
