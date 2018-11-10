import os
import numpy as np
import GPflow
import GPflow.training.monitor as mon
import tensorflow as tf


class CustomTensorBoardTask(mon.BaseTensorBoardTask):
    def __init__(self, file_writer, model, Xt, Yt):
        super().__init__(file_writer, model)
        self.Xt = Xt
        self.Yt = Yt
        self._full_test_err = tf.placeholder(gpflow.settings.tf_float, shape=())
        self._full_test_nlpp = tf.placeholder(gpflow.settings.tf_float, shape=())
        self._summary = tf.summary.merge([tf.summary.scalar("test_rmse", self._full_test_err),
                                          tf.summary.scalar("test_nlpp", self._full_test_nlpp)])

    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
        minibatch_size = 100
        preds = np.vstack([self.model.predict_y(self.Xt[mb * minibatch_size:(mb + 1) * minibatch_size, :])[0]
                           for mb in range(-(-len(self.Xt) // minibatch_size))])
        test_err = np.mean((self.Yt - preds) ** 2.0)**0.5
        self._eval_summary(context, {self._full_test_err: test_err, self._full_test_nlpp: 0.0})
