"""Synchronous SGD"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import ray

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


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def get_model(lr):
    net = Net([Dense(200), ReLU(), Dense(50), ReLU(), Dense(10)])
    model = Model(net=net, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=lr))
    # init parameters manually
    model.net.init_params(input_shape=(784,))
    return model


@ray.remote
class ParamServer(object):

    def __init__(self, model, ds):
        self.model = model
        self.ds = ds

        self.cnt = 0
        self.start_time = time.time()
        self.last_eval_time = -1

    def get_params(self):
        return self.model.net.params

    def set_params(self, params):
        self.model.net.params = params

    def apply_update(self, values, type_, cnt=1):
        assert type_ in ("grads", "params")
        if type_ == "grads":
            self.apply_grads(values)
        else:
            self.set_params(values)
        # update counts
        self.cnt += cnt

        # evaluate 
        if time.time() - self.last_eval_time > 5:
            self.last_eval_time = time.time()
            self.evaluate()  
    
    def apply_grads(self, grads):
        self.model.apply_grad(grads)

    def evaluate(self):
        self.model.set_phase("TEST")

        test_x, test_y = ray.get(self.ds.test_set.remote())

        test_pred = self.model.forward(test_x)

        test_pred_idx = np.argmax(test_pred, axis=1)
        test_y_idx = np.argmax(test_y, axis=1)

        print("[%.2fs] accuracy after %d batches: " %
              (time.time() - self.start_time, self.cnt))
        print(accuracy(test_pred_idx, test_y_idx))
        self.model.set_phase("TRAIN")


@ray.remote
class Worker(object):

    def __init__(self, model, ds):
        self.model = model
        self.ds = ds

    def get_params(self):
        return self.model.net.params

    def set_params(self, params):
        self.model.net.params = params

    def compute_grads(self, batch):
        preds = self.model.forward(batch.inputs)
        _, grads = self.model.backward(preds, batch.targets)
        return grads

    def apply_grads(self, grads):
        self.model.apply_grad(grads)

    def elastic_update(self, params, alpha=0.5):
        diff = self.model.net.params - params
        self.model.net.params -= alpha * diff


@ray.remote
class DataServer(object):

    def __init__(self, dataset, batch_size, num_ep):
        self._train_set, self._test_set = dataset

        self.iterator = BatchIterator(batch_size)
        self.batch_gen = None

        iter_each_epoch = len(self._train_set[0]) // batch_size + 1
        self._total_iters = num_ep * iter_each_epoch

    def test_set(self):
        return self._test_set

    def total_iters(self):
        return self._total_iters

    def next_batch(self):
        if self.batch_gen is None:
            self.batch_gen = self.iterator(*self._train_set)

        try:
            batch = next(self.batch_gen)
        except StopIteration:
            self.batch_gen = None
            batch = self.next_batch()
        return batch


def SSGD(ds, ps, workers):
    """
    Synchronous Stochastic Gradient Descent
    ref: https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf
    """
    # iterations for each worker
    iterations = ray.get(ds.total_iters.remote())
    for i in range(iterations):
        all_grads = []

        global_params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(global_params)

            batch = ds.next_batch.remote()
            local_grads = worker.compute_grads.remote(batch)

            all_grads.append(local_grads)
        all_grads = ray.get(all_grads)

        # average local gradients
        grads = sum(all_grads) / len(workers)
        # update global model
        ps.apply_update.remote(grads, type_="grads", cnt=len(workers))


def MA(ds, ps, workers):
    """ 
    Model Average Method
    ref: https://www.aclweb.org/anthology/N10-1069.pdf
    """
    comm_interval = 10  # interval of communication
    
    # iterations for each worker
    iterations = ray.get(ds.total_iters.remote())
    for i in range(iterations // comm_interval):
        all_params = []

        global_params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(global_params)
            for _ in range(comm_interval):
                batch = ds.next_batch.remote()
                local_grads = worker.compute_grads.remote(batch)
                worker.apply_grads.remote(local_grads)
            local_params = worker.get_params.remote()
            all_params.append(local_params)

        all_params = ray.get(all_params)

        # average parameters
        params = sum(all_params) / len(workers)
        # update global model
        ps.apply_update.remote(params, type_="params", 
                               cnt=len(workers) * comm_interval)


def BMUF(ds, ps, workers):
    """
    Blcok-wise Model Update Filtering
    ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf
    """
    comm_interval = 10  # interval of communication

    beta = 0.9  # momentum coef
    m = 0  # momentum
    t = 0

    # iterations for each worker
    iterations = ray.get(ds.total_iters.remote())

    for i in range(iterations // comm_interval):
        t += 1
        all_params = []

        global_params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(global_params)
            for _ in range(comm_interval):
                batch = ds.next_batch.remote()
                local_grads = worker.compute_grads.remote(batch)
                worker.apply_grads.remote(local_grads)

            local_params = worker.get_params.remote()
            all_params.append(local_params)

        all_params = ray.get(all_params)

        # parameter with momentum
        params = sum(all_params) / len(workers)
        m = beta * m + (1.0 - beta) * params
        m_ = m / (1 - beta ** t)  # bias correction
        # update global model
        ps.apply_update.remote(m_, type_="params", cnt=len(workers) * comm_interval)


def EASGD(ds, ps, workers):
    """
    Elastic Average Stochastic Gradient Decent
    ref: https://arxiv.org/abs/1412.6651
    """
    alpha = 0.5  # elastic coefficient
    beta = 0.9  # momentum coefficient
    m = 0  # momentum
    t = 0

    # iterations for each worker
    iterations = ray.get(ds.total_iters.remote())
    for i in range(iterations):
        t += 1

        all_params = []
        global_params = ps.get_params.remote()

        for worker in workers:
            batch = ds.next_batch.remote()
            local_grads = worker.compute_grads.remote(batch)
            worker.apply_grads.remote(local_grads)

            # update local parameters elastically
            worker.elastic_update.remote(global_params, alpha)

            local_params = worker.get_params.remote()
            all_params.append(local_params)

        all_params = ray.get(all_params)

        # parameter with momentum
        params = sum(all_params) / len(workers)
        m = beta * m + (1.0 - beta) * params
        m_ = m / (1 - beta ** t)  # bias correction
        # update global model
        ps.apply_update.remote(m_, type_="params", cnt=len(workers))


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    # dataset preparation
    train_set, _, test_set = mnist(args.data_dir)
    train_set = (train_set[0], get_one_hot(train_set[1], 10))
    test_set = (test_set[0], get_one_hot(test_set[1], 10))
    dataset = (train_set, test_set)

    ray.init()
    # init a network model
    model = get_model(args.lr)
    # init the data server
    ds = DataServer.remote(dataset, args.batch_size, args.num_ep)
    # init the parameter server
    ps = ParamServer.remote(model=copy.deepcopy(model), ds=ds)

    # init workers
    workers = []
    for rank in range(1, args.num_proc + 1):
        worker = Worker.remote(model=copy.deepcopy(model), ds=ds)
        workers.append(worker)

    algo_dict = {"SSGD": SSGD, "MA": MA, "BMUF": BMUF, "EASGD": EASGD}
    algo = algo_dict.get(args.algo, None)
    if algo is None:
        print("Error: Invalid training algorithm. "
              "Available choices: [SSGD|MA|BUMF|EASGD]")
        return 

    # run synchronous training
    algo(ds, ps, workers)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SSGD",
                        help="[*SSGD|MA|BUMF|EASGD]")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of workers.")
    parser.add_argument("--num_ep", default=4, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    main(parser.parse_args())
