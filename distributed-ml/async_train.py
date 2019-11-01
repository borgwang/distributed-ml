"""Asynchronous training of neural networks."""

import argparse
import copy
import os
import pickle
import time

import numpy as np
import ray
from tinynn.utils.dataset import cifar10
from tinynn.utils.seeder import random_seed

from components.data_server import DataServer
from components.param_server import ParamServer
from components.stats_server import StatsServer
from components.worker import Worker
from models import get_mlp


@ray.remote
class AsyncWorker(Worker):

    def __init__(self, model):
        super().__init__(model)
        self.t = 0
        self.mse = 0
        self.beta = 0.95

    def compute_dc_grads(self, batch, global_params):
        self.t += 1
        grads = self.compute_grads(batch)

        # adaptive lambda
        self.mse = self.beta * self.mse + (1 - self.beta) * grads ** 2
        mse = self.mse / (1 - self.beta ** self.t)
        lambda_ = 2.0 / (mse + 1e-7) ** 0.5

        approx_hessian = lambda_ * grads * grads
        return grads + approx_hessian * (self.model.net.params - global_params)


def ASGD(ss, ds, ps, workers):
    """
    Asynchronous Stochastic Gradient Descent
    ref: https://arxiv.org/abs/1104.5525
    """
    history = []
    start_time = time.time()

    iter_each_epoch = ray.get(ds.iter_each_epoch.remote())
    iterations = ray.get(ds.total_iters.remote())
    for i in range(iterations):
        for worker in workers:
            # fetch global params
            global_params = ps.get_params.remote()
            worker.set_params.remote(global_params)

            # update local params
            batch = ds.next_batch.remote()
            grads = worker.compute_grads.remote(batch)

            # push grads to the param server
            ps.apply_update.remote(grads, type_="grads")

        # evaluate
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))["accuracy"]
        epoch = 1.0 * i * len(workers) / iter_each_epoch
        res = {"epoch": epoch, "t": time.time() - start_time, "acc": acc}
        print(res)
        history.append(res)
    return history


def DCASGD(ss, ds, ps, workers):
    """
    Asynchronous Stochastic Gradient Descent with Delay Compensation
    ref: https://arxiv.org/abs/1609.08326
    """
    history = []
    start_time = time.time()

    iter_each_epoch = ray.get(ds.iter_each_epoch.remote())
    iterations = ray.get(ds.total_iters.remote())
    for i in range(iterations):
        for worker in workers:
            # fetch global params
            global_params = ps.get_params.remote()
            worker.set_params.remote(global_params)

            # update local params
            batch = ds.next_batch.remote()
            # compute dc_grads
            dc_grads = worker.compute_dc_grads.remote(
                batch, global_params)

            # push dc_grads to the param server
            ps.apply_update.remote(dc_grads, type_="grads")

        # evaluate
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))["accuracy"]
        epoch = (1.0 * i * len(workers)) / iter_each_epoch
        res = {"epoch": epoch, "t": time.time() - start_time, "acc": acc}
        print(res)
        history.append(res)
    return history


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    ray.init()
    # dataset preparation
    dataset = cifar10(args.data_dir, one_hot=True)
    # init a network model
    model = get_mlp(args.lr)
    input_dim = np.prod(dataset[0][0].shape[1:])
    model.net.init_params(input_shape=(input_dim,))
    # init the statistics server
    ss = StatsServer.remote(copy.deepcopy(model))
    # init the data server
    ds = DataServer.remote(dataset, args.batch_size, 
                           args.num_ep, args.num_workers)
    # init the parameter server
    ps = ParamServer.remote(copy.deepcopy(model))
    # init workers
    workers = []
    for rank in range(1, args.num_workers + 1):
        worker = AsyncWorker.remote(copy.deepcopy(model))
        workers.append(worker)

    algo_dict = {"ASGD": ASGD, "DCASGD": DCASGD}
    algo = algo_dict.get(args.algo, None)
    if algo is None:
        raise ValueError("Error: Invalid training algorithm. "
                         "Available choices: [ASGD|DCASGD]")

    # run asynchronous training
    history = algo(ss, ds, ps, workers)
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    file_name = "%s-%s-%s.log" % (args.algo, args.num_workers, args.num_ep)
    save_path = os.path.join(args.result_dir, file_name)
    with open(save_path, "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ASGD",
                        help="[ASGD|DCASGD]")
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
