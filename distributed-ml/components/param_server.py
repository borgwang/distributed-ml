import time

import ray


@ray.remote
class ParamServer(object):

    def __init__(self, model):
        self.model = model

        self.start_time = time.time()

    def get_params(self):
        return self.model.net.params

    def set_params(self, params):
        self.model.net.params = params

    def apply_update(self, values, type_):
        assert type_ in ("grads", "params")
        if type_ == "grads":
            self.apply_grads(values)
        else:
            self.set_params(values)

    def apply_grads(self, grads):
        self.model.apply_grads(grads)
