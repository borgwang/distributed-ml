"""Synchronous training of neural networks."""

import argparse
import copy
import os
import pickle
import time

import numpy as np
import ray

from components.data_server import DataServer
from components.param_server import ParamServer
from components.stats_server import StatsServer
from components.worker import Worker
from models import get_mlp

# tinynn packages
from utils.dataset import cifar10
from utils.seeder import random_seed


@ray.remote
class SyncWorker(Worker):

    def elastic_update(self, params, alpha=0.5):
        diff = self.model.net.params - params
        self.model.net.params -= alpha * diff


def SSGD(ss, ds, ps, workers):
    """
    Synchronous Stochastic Gradient Descent
    ref: https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf
    """
    history = []
    start_time = time.time()

    iter_each_epoch = ray.get(ds.iter_each_epoch.remote())
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

        # sum of all local gradients
        grads = sum(all_grads)
        # update global model
        ps.apply_update.remote(grads, type_="grads")

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))["accuracy"]
        epoch = (1.0 * i * len(workers)) / iter_each_epoch
        res = {"epoch": epoch, "t": time.time() - start_time, "acc": acc}
        print(res)
        history.append(res)
    return history


def MA(ss, ds, ps, workers):
    """ 
    Model Average Method
    ref: https://www.aclweb.org/anthology/N10-1069.pdf
    """
    history = []
    start_time = time.time()

    comm_interval = 10  # interval of communication
    
    iter_each_epoch = ray.get(ds.iter_each_epoch.remote())
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

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))["accuracy"]
        epoch = (1.0 * i * len(workers) * comm_interval) / iter_each_epoch
        res = {"epoch": epoch, "t": time.time() - start_time, "acc": acc}
        print(res)
        history.append(res)

    return history


def BMUF(ss, ds, ps, workers):
    """
    Block-wise Model Update Filtering
    ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf
    """
    history = []
    start_time = time.time()

    comm_interval = 10  # interval of communication
    beta = 0.5  # momentum coefficient
    m = 0  # momentum
    t = 0

    iter_each_epoch = ray.get(ds.iter_each_epoch.remote())
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

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))["accuracy"]
        epoch = (1.0 * i * len(workers) * comm_interval) / iter_each_epoch
        res = {"epoch": epoch, "t": time.time() - start_time, "acc": acc}
        print(res)
        history.append(res)

    return history


def EASGD(ss, ds, ps, workers):
    """
    Elastic Average Stochastic Gradient Descent
    ref: https://arxiv.org/abs/1412.6651
    """
    history = []
    start_time = time.time()

    alpha = 0.05  # elastic coefficient
    beta = 0.5  # momentum coefficient
    m = 0  # momentum
    t = 0

    iter_each_epoch = ray.get(ds.iter_each_epoch.remote())
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

        # evaluation
        ss.set_params.remote(ps.get_params.remote())
        acc = ray.get(ss.evaluate.remote(ds.test_set.remote()))["accuracy"]
        t = time.time() - start_time
        epoch = (1.0 * i * len(workers)) / iter_each_epoch
        res = {"epoch": epoch, "t": time.time() - start_time, "acc": acc}
        print(res)
        history.append(res)

    return history


def main(args):
    if args.seed >= 0:
        random_seed(args.seed)

    # dataset preparation
    dataset = cifar10(args.data_dir, one_hot=True)

    ray.init()
    # init a network model
    model = get_mlp(args.lr)
    input_dim = np.prod(dataset[0][0].shape[1:])
    model.net.init_params(input_shape=(input_dim,))
    # init the statistics server
    ss = StatsServer.remote(model=copy.deepcopy(model))
    # init the data server
    ds = DataServer.remote(dataset, args.batch_size, 
                           args.num_ep, args.num_workers)
    # init the parameter server
    ps = ParamServer.remote(model=copy.deepcopy(model))
    # init workers
    workers = []
    for rank in range(1, args.num_workers + 1):
        worker = SyncWorker.remote(model=copy.deepcopy(model))
        workers.append(worker)

    algo_dict = {"SSGD": SSGD, "MA": MA, "BMUF": BMUF, "EASGD": EASGD}
    algo = algo_dict.get(args.algo, None)
    if algo is None:
        raise ValueError("Error: Invalid training algorithm. "
                         "Available choices: [SSGD|MA|BUMF|EASGD]")

    # run synchronous training
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
