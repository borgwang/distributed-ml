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

    def __init__(self, model, test_set):
        self.test_set = test_set
        self.model = model

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

    def __init__(self, model, train_set, ps, batch_size):
        self.model = model
        self.train_set = train_set
        self.ps = ps

        self.iterator = BatchIterator(batch_size=batch_size)
        self.batch_gen = None

    def get_params(self):
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

    def compute_grads(self):
        batch, _ = self.get_next_batch()
        preds = self.model.forward(batch.inputs)
        _, grads = self.model.backward(preds, batch.targets)
        return grads

    def compute_elastic_grads(self, coef=0.1):
        local_grads = self.compute_grads()
        local_params = self.get_params()

        global_params = ray.get(self.ps.get_params.remote())
        elastic = coef * (local_params - global_params)
        return local_grads + elastic

    def apply_grads(self, grads):
        self.model.apply_grad(grads)


def SSGD(ps, workers, iter_each_epoch, num_ep):
    """
    Synchronous Stochastic Gradient Descent
    ref: https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf
    """
    for i in range(num_ep * iter_each_epoch):
        all_grads = []

        global_params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(global_params)
            local_grads = worker.compute_grads.remote()
            all_grads.append(local_grads)
        all_grads = ray.get(all_grads)

        # average local gradients
        grads = sum(all_grads) / len(workers)
        # update global model
        ps.apply_update.remote(grads, type_="grads", cnt=len(workers))


def MA(ps, workers, iter_each_epoch, num_ep):
    """ 
    Model Average Method
    ref: https://www.aclweb.org/anthology/N10-1069.pdf
    """
    comm_interval = 10

    for i in range(num_ep * iter_each_epoch // M):
        all_params = []

        global_params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(global_params)
            for _ in range(comm_interval):
                local_grads = worker.compute_grads.remote()
                worker.apply_grads.remote(local_grads)
            local_params = worker.get_params.remote()
            all_params.append(local_params)

        all_params = ray.get(all_params)

        # average parameters
        params = sum(all_params) / len(workers)
        # update global model
        ps.apply_update.remote(params, type_="params", cnt=len(workers) * M)


def BMUF(ps, workers, iter_each_epoch, num_ep):
    """
    Blcok-wise Model Update Filtering
    ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf
    """
    comm_interval = 10  # communication interval

    beta = 0.9  # momentum coef
    m = 0  # momentum
    t = 0

    for i in range(num_ep * iter_each_epoch // M):
        t += 1
        all_params = []

        global_params = ps.get_params.remote()
        for worker in workers:
            worker.set_params.remote(global_params)
            for _ in range(comm_interval):
                local_grads = worker.compute_grads.remote()
                worker.apply_grads.remote(local_grads)

            local_params = worker.get_params.remote()
            all_params.append(local_params)

        all_params = ray.get(all_params)

        # parameter with momentum
        params = sum(all_params) / len(workers)
        m = beta * m + (1.0 - beta) * params
        m_ = m / (1 - beta ** t)  # bias correction
        # update global model
        ps.apply_update.remote(m_, type_="params", cnt=len(workers) * M)


def EASGD(ps, workers, iter_each_epoch, num_ep):
    """
    Elastic Average Stochastic Gradient Decent
    ref: https://arxiv.org/abs/1412.6651
    """
    alpha = 0.5  # elastic coef
    beta = 0.9  # momentum coef
    m = 0  # momentum
    t = 0

    for i in range(num_ep * iter_each_epoch):
        t += 1

        all_params = []
        for worker in workers:
            grads = worker.compute_elastic_grads.remote(coef=alpha)
            worker.apply_grads.remote(grads)

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

    # data preparation
    train_set, valid_set, test_set = mnist(args.data_dir)
    train_set = (train_set[0], get_one_hot(train_set[1], 10))

    ray.init()

    # init a network model
    model = get_model(args.lr)

    # init the parameter server
    ps = ParamServer.remote(model=copy.deepcopy(model),
                            test_set=test_set)
    # init workers
    workers = []
    for rank in range(1, args.num_proc + 1):
        worker = Worker.remote(model=copy.deepcopy(model),
                               train_set=train_set,
                               ps=ps,
                               batch_size=args.batch_size)
        workers.append(worker)

    iter_each_epoch = len(train_set[0]) // args.batch_size + 1

    algo_dict = {"SSGD": SSGD, "MA": MA, "BMUF": BMUF, "EASGD": EASGD}
    algo = algo_dict.get(args.algo, None)
    if algo is None:
        print("Error: Invalid training algorithm. "
              "Available choices: [SSGD|MA|BUMF|EASGD]")
        return 

    # run sysnchronous training
    algo(ps, workers, iter_each_epoch, args.num_ep)

    time.sleep(10000)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SSGD",
                        help="[*SSGD|MA|BUMF|EASGD]")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(curr_dir, "data"))
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of workers.")
    parser.add_argument("--num_ep", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    main(parser.parse_args())
