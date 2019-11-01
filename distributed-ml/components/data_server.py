import ray
from tinynn.utils.data_iterator import BatchIterator


@ray.remote
class DataServer(object):

    def __init__(self, dataset, batch_size, num_ep, num_workers):
        self._train_set, self._test_set = dataset

        self.iterator = BatchIterator(batch_size)
        self.batch_gen = None

        self._iter_each_epoch = len(self._train_set[0]) // batch_size + 1
        self._total_iters = num_ep * self._iter_each_epoch // num_workers + 1

    def test_set(self):
        return self._test_set

    def total_iters(self):
        return self._total_iters

    def iter_each_epoch(self):
        return self._iter_each_epoch

    def next_batch(self):
        if self.batch_gen is None:
            self.batch_gen = self.iterator(*self._train_set)

        try:
            batch = next(self.batch_gen)
        except StopIteration:
            self.batch_gen = None
            batch = self.next_batch()
        return batch
