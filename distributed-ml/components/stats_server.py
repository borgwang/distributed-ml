import numpy as np
import ray

from utils.metric import accuracy


@ray.remote
class StatsServer(object):

    def __init__(self, model):
        self.model = model
        self.model.set_phase("TEST")

    def set_params(self, params):
        self.model.net.params = params

    def evaluate(self, test_set):
        test_x, test_y = test_set
        test_pred = self.model.forward(test_x)

        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.argmax(test_y, axis=1)
        return accuracy(test_pred_idx, test_y_idx)
