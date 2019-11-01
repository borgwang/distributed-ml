class Worker(object):

    def __init__(self, model):
        self.model = model

    def get_params(self):
        return self.model.net.params

    def set_params(self, params):
        self.model.net.params = params

    def compute_grads(self, batch):
        preds = self.model.forward(batch.inputs)
        _, grads = self.model.backward(preds, batch.targets)
        return grads

    def apply_grads(self, grads):
        self.model.apply_grads(grads)
