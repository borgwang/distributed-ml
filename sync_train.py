"""Synchronous SGD"""

import runtime_path  # isort:skip
import argparse
import copy
import os
import sys
import time

import numpy as np
import ray

sys.path.insert(0, os.path.abspath("./tinynn"))

from core.layer import Dense
from core.layer import ReLU
from core.loss import SoftmaxCrossEntropy
from core.model import Model
from core.net import Net
from core.optimizer import Adam
from utils.data_iterator import BatchIterator
from utils.dataset import mnist
from utils.metric import accuracy
from utils.seeder import random_seed

# TODO: Something wrong when trying to import tinynn with Ray.
# see: https://github.com/ray-project/ray/issues/5125

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def get_model():
    net = Net([Dense(200), ReLU(), Dense(50), ReLU(), Dense(10)])
    model = Model(net=net, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=args.lr))
    return model


@ray.remote
class ParamServer(object):

    def __init__(self, model, test_set):
        self.test_set = test_set
        self.model = model
        self.model.net.init_params(input_shape=(784,))

        self.cnt = 0
        self.start_time = time.time()

    def get_params(self):
        return self.model.net.params

    def apply_grads(self, *grads):
        self.model.apply_grad(sum(grads))

        self.cnt += len(grads)
        if self.cnt % 100 == 0:
            self.evaluate()  
    
    def evaluate(self):
        self.model.set_phase("TEST")
        test_x, test_y = self.test_set
        test_pred = self.model.forward(test_x)

        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.asarray(test_y)

        print("[%.2fs] accuracy after %d batches: " %
              (time.time() - self.start_time, self.cnt))
        print(accuracy(test_pred_idx, test_y_idx))
        self.model.set_phase("TRAIN")


@ray.remote
class Worker(object):

    def __init__(self, model, train_set, ps):
        self.ps = ps
        self.model = model
        self.train_set = train_set

        self.iterator = BatchIterator(batch_size=args.batch_size)
        self.batch_gen = None

    def get_params(self, params):
        return self.model.net.params

    def set_params(self, params):
        self.model.net.params = params

    def get_next_batch(self):
        end_epoch = False
        # reset batch generator if needed
        if self.batch_gen is None:
            self.batch_gen = self.iterator(*self.train_set)

        try:
            batch = next(self.batch_gen)
        except StopIteration:
            self.batch_gen = None
            batch, _ = self.get_next_batch()
            end_epoch = True
        return batch, end_epoch

    def compute_grads(self, batch=None):
        if batch is None:
            batch, _ = self.get_next_batch()

        # fetch model params from server
        params = ray.get(self.ps.get_params.remote())
        self.set_params(params)

        # get local gradients
        preds = self.model.forward(batch.inputs)
        _, grads = self.model.backward(preds, batch.targets)
        return grads

    def train_one_epoch(self):
        end_epoch = False
        while not end_epoch:
            batch, end_epoch = self.get_next_batch()
            grads = self.compute_grads(batch)
            self.ps.apply_grads.remote(grads)
        return self.get_params()


def SSGD(ps, workers, iter_each_epoch):
    """Synchronous Stochastic Gradient Descent"""
    for i in range(args.num_ep * iter_each_epoch):
        all_grads = [worker.compute_grads.remote() for worker in workers]
        ps.apply_grads.remote(*all_grads)

        params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(params)


def MA(ps, workers, iter_each_epoch):
    """ 
    Model Average Method
    see: https://www.aclweb.org/anthology/N10-1069.pdf
    """
    for i in range(num_updates):
        all_params = [worker.train_one_epoch.remote() for worker in workers]
        all_params = ray.get(all_params)
        ps.set_params.remote(sum(all_params))

        params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(params)
    

def BMUF():
    pass


def ADMM():
    pass


def EASGD():
    pass


def main():
    if args.seed >= 0:
        random_seed(args.seed)

    # data preparation
    train_set, valid_set, test_set = mnist(args.data_dir)
    train_set = (train_set[0], get_one_hot(train_set[1], 10))

    # get init model
    model = get_model()

    ray.init()
    ps = ParamServer.remote(model=copy.deepcopy(model),
                            test_set=test_set)
    workers = []
    for rank in range(1, args.num_proc + 1):
        worker = Worker.remote(model=copy.deepcopy(model),
                               train_set=train_set,
                               ps=ps)
        workers.append(worker)

    iter_each_epoch = len(train_set[0]) // args.batch_size + 1
    if args.algo == "MA":
        MA(ps, workers, iter_each_epoch)
    else:
        print("Invalid training algorithm.")

    time.sleep(10000)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SSGD",
                        help="[*SSGD|MA|BUMF|ADMM|EASGD]")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of workers.")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    global args
    args = parser.parse_args()
    main()
