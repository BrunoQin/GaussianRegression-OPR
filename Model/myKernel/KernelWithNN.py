import tensorflow as tf
import gpflow


class KernelWithNN(gpflow.kernels.Kernel):
    """
    This kernel class allows for easily adding a NN (or other function) to a GP model.
    The kernel does not actually do anything with the NN.
    """

    def __init__(self, kern, f):
        """
        kern.input_dim needs to be consistent with the output dimension of f
        """
        super().__init__(kern.input_dim)
        self.kern = kern
        self._f = f

    def f(self, X):
        if X is not None:
            with tf.variable_scope('forward', reuse=tf.AUTO_REUSE):
                return self._f(X)

    def _get_f_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward')

    @gpflow.autoflow([gpflow.settings.float_type, [None,None]])
    def compute_f(self, X):
        return self.f(X)

    def K(self, X, X2=None):
        return self.kern.K(X, X2)

    def Kdiag(self, X):
        return self.kern.Kdiag(X)


class KernelSpaceInducingPoints(gpflow.features.InducingPointsBase):
    pass

# same Kuu as regular inducing points
gpflow.features.Kuu.register(KernelSpaceInducingPoints, KernelWithNN)(
    gpflow.features.Kuu.dispatch(gpflow.features.InducingPoints, gpflow.kernels.Kernel)
)


# Kuf is in NN output space
@gpflow.features.dispatch(KernelSpaceInducingPoints, KernelWithNN, object)
def Kuf(feat, kern, Xnew):
    with gpflow.params_as_tensors_for(feat):
        return kern.K(feat.Z, kern.f(Xnew))


class NNComposedKernel(KernelWithNN):
    """
    This kernel class applies f() to X before calculating K
    """

    def K(self, X, X2=None):
        return super().K(self.f(X), self.f(X2))

    def Kdiag(self, X):
        return super().Kdiag(self.f(X))


# we need to add these extra functions to the model so the tensorflow variables get picked up
class NN_SVGP(gpflow.models.SVGP):
    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.kern._get_f_vars()

    @property
    def initializables(self):
        return super().initializables + self.kern._get_f_vars()
