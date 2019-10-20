### Distributed Training of Machine Learning Models

- Synchronous
    - SSGD [Synchronous Stochastic Gradient Descent](https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf)
    - MA [Model Average Method](https://www.aclweb.org/anthology/N10-1069.pdf)
    - BMUF [Block-wise Model Update Filtering](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf)
    - EASGD [Elastic Average Stochastic Gradient Descent](https://arxiv.org/abs/1412.6651)

- Asynchronous
    - ASGD [Asynchronous Stochastic Gradient Descent](https://arxiv.org/abs/1104.5525)
    - DC-ASGD [Asynchronous Stochastic Gradient Descent with Delay Compensation](https://arxiv.org/abs/1609.08326)

#### Get started

```bash
# git clone
git clone --recursive https://github.com/borgwang/distributed-ml.git

cd distributed-ml
pip install -r requirements.txt

# run
PYTHONPATH=$PYTHONPATH:./tinynn python distributed-ml/sync_run.py --algo SSGD
```
