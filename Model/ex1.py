import os
import numpy as np
import tensorflow as tf
import gpflow
from scipy.cluster.vq import kmeans2
import gpflow.training.monitor as mon

from myKernel.KernelWithNN import KernelWithNN
from myKernel.KernelWithNN import NNComposedKernel
from myKernel.KernelWithNN import NN_SVGP
from myKernel.CustomTensorBoardTask import CustomTensorBoardTask
from myKernel.NN import NN
from myKernel.LoadData import LoadData

float_type = gpflow.settings.float_type


def ex1():
    fX_dim = 1
    M = 100
    X, Y, Xt, Yt = LoadData.load_ocean()

    # annoyingly only float32 and lower is supported by the conv layers
    f = lambda x: tf.cast(NN.cnn_fn(tf.cast(x, tf.float32), fX_dim), float_type)
    kern = NNComposedKernel(gpflow.kernels.Matern32(fX_dim), f)

    # build the model
    lik = gpflow.likelihoods.Gaussian()

    Z = kmeans2(X, M, minit='points')[0]

    model = NN_SVGP(X, Y, kern, lik, Z=Z, minibatch_size=200)

    session = model.enquire_session()
    global_step = mon.create_global_step(session)

    # print
    print_task = mon.PrintTimingsTask().with_name('print') \
        .with_condition(mon.PeriodicIterationCondition(10)) \
        .with_exit_condition(True)

    sleep_task = mon.SleepTask(0.01).with_name('sleep').with_name('sleep')

    saver_task = mon.CheckpointTask('./monitor-saves').with_name('saver') \
        .with_condition(mon.PeriodicIterationCondition(10)) \
        .with_exit_condition(True)

    file_writer = mon.LogdirWriter('./model-tensorboard')

    model_tboard_task = mon.ModelToTensorBoardTask(file_writer, model).with_name('model_tboard') \
        .with_condition(mon.PeriodicIterationCondition(10)) \
        .with_exit_condition(True)

    lml_tboard_task = mon.LmlToTensorBoardTask(file_writer, model).with_name('lml_tboard') \
        .with_condition(mon.PeriodicIterationCondition(100)) \
        .with_exit_condition(True)

    custom_tboard_task = CustomTensorBoardTask(file_writer, model, Xt, Yt).with_name('custom_tboard') \
        .with_condition(mon.PeriodicIterationCondition(100)) \
        .with_exit_condition(True)

    monitor_tasks = [print_task, model_tboard_task, lml_tboard_task, custom_tboard_task, saver_task, sleep_task]
    monitor = mon.Monitor(monitor_tasks, session, global_step)

    if os.path.isdir('./monitor-saves'):
        mon.restore_session(session, './monitor-saves')

    # use gpflow wrappers to train. NB all session handling is done for us
    optimiser = gpflow.training.AdamOptimizer(0.001)

    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, maxiter=30000, global_step=global_step)

    file_writer.close()

    print('LML after the optimisation: %f' % m.compute_log_likelihood())
    # # predictions
    pY, pYv = model.predict_y(Xt)
    rmse = np.mean((pY - Yt) ** 2.0) ** 0.5
    nlpp = -np.mean(-0.5 * np.log(2 * np.pi * pYv) - 0.5 * (Yt - pY) ** 2.0 / pYv)

    print('rmse is {:.4f}%, nlpp is {:.f}%'.format(rmse, nlpp))


if __name__ == "__main__":
    ex1()
    gpflow.reset_default_graph_and_session()


