

class Experiment(object):
    def __init__(self, flags):
        self.flags = flags
        self._load_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logger()

    def _load_data(self):
        raise NotImplementedError()

    def _setup_model(self):
        raise NotImplementedError()

    def _setup_optimizer(self):
        raise NotImplementedError()

    def _setup_logger(self):
        raise NotImplementedError()
