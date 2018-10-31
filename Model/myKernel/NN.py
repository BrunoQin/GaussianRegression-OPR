import tensorflow as tf


class NN():
    # a vanilla conv net
    # this gets 97.3% accuracy on MNIST when used on its own (+ final linear layer) after 20K iterations
    def cnn_fn(x, output_dim):
        """
        Adapted from https://www.tensorflow.org/tutorials/layers
        """
        conv1 = tf.layers.conv2d(
            inputs=tf.reshape(x, [-1, 28, 28, 1]),
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        return tf.layers.dense(inputs=pool2_flat, units=output_dim, activation=tf.nn.relu)