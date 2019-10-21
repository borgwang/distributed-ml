"""Synchronous training of neural networks."""

import argparse
import copy
import os
import pickle
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


def get_model(lr):
    net = Net([
        Dense(200),
        ReLU(),
        Dense(100),
        ReLU(),
        Dense(70),
        ReLU(),
        Dense(30),
        ReLU(),
        Dense(10)])
    model = Model(net=net, loss=SoftmaxCrossEntropy(),
                  optimizer=Adam(lr=lr))
    # init parameters manually
    model.net.init_params(input_shape=(784,))
    return model


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


@ray.remote
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

    def elastic_update(self, params, alpha=0.5):
        diff = self.model.net.params - params
        self.model.net.params -= alpha * diff


@ray.remote
class DataServer(object):

    def __init__(self, dataset, batch_size, num_ep):
        self._train_set, _, self._test_set = dataset

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


def SSGD(ss, ds, ps, workers):
    """
    Synchronous Stochastic Gradient Descent
    ref: https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf
    """
    history = []
    start_time = time.time()
    batch_cnt = 0

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
        ps.apply_update.remote(grads, type_="grads")
        batch_cnt += len(workers)

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))
        res = {"t": time.time() - start_time, 
               "acc": acc["accuracy"],
               "batch_cnt": batch_cnt}
        print(res)
        history.append(res)
    return history


def MA(ss, ds, ps, workers):
    """ 
    Model Average Method
    ref: https://www.aclweb.org/anthology/N10-1069.pdf
    """
    history = []
    batch_cnt = 0
    start_time = time.time()

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
        ps.apply_update.remote(params, type_="params")
        batch_cnt += len(workers) * comm_interval

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))
        res = {"t": time.time() - start_time, 
               "acc": acc["accuracy"],
               "batch_cnt": batch_cnt}
        print(res)
        history.append(res)

    return history


def BMUF(ss, ds, ps, workers):
    """
    Block-wise Model Update Filtering
    ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf
    """
    history = []
    batch_cnt = 0
    start_time = time.time()

    comm_interval = 10  # interval of communication
    beta = 0.5  # momentum coefficient
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
        ps.apply_update.remote(m_, type_="params")
        batch_cnt += len(workers) * comm_interval

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))
        res = {"t": time.time() - start_time, 
               "acc": acc["accuracy"],
               "batch_cnt": batch_cnt}
        print(res)
        history.append(res)

    return history


def EASGD(ss, ds, ps, workers):
    """
    Elastic Average Stochastic Gradient Descent
    ref: https://arxiv.org/abs/1412.6651
    """
    history = []
    batch_cnt = 0
    start_time = time.time()

    alpha = 0.05  # elastic coefficient
    beta = 0.5  # momentum coefficient
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
        ps.apply_update.remote(m_, type_="params")
        batch_cnt += len(workers)

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))
        res = {"t": time.time() - start_time, 
               "acc": acc["accuracy"],
               "batch_cnt": batch_cnt}
        print(res)
        history.append(res)

    return history


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    # dataset preparation
    dataset = mnist(args.data_dir, one_hot=True)

    ray.init()
    # init a network model
    model = get_model(args.lr)
    # init the statistics server
    ss = StatsServer.remote(model=copy.deepcopy(model))
    # init the data server
    ds = DataServer.remote(dataset, args.batch_size, args.num_ep)
    # init the parameter server
    ps = ParamServer.remote(model=copy.deepcopy(model))
    # init workers
    workers = []
    for rank in range(1, args.num_workers + 1):
        worker = Worker.remote(model=copy.deepcopy(model))
        workers.append(worker)

    algo_dict = {"SSGD": SSGD, "MA": MA, "BMUF": BMUF, "EASGD": EASGD}
    algo = algo_dict.get(args.algo, None)
    if algo is None:
        print("Error: Invalid training algorithm. "
              "Available choices: [SSGD|MA|BUMF|EASGD]")
        return 

    # run synchronous training
    history = algo(ss, ds, ps, workers)
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    save_path = os.path.join(args.result_dir, args.algo)
    with open(save_path, "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SSGD",
                        help="[*SSGD|MA|BUMF|EASGD]")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--result_dir", type=str,
                        default=os.path.join(curr_dir, "result"))
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers.")
    parser.add_argument("--num_ep", default=4, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    main(parser.parse_args())
