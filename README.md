### Distributed Training of Machine Learning Models

- Synchronous Training Algorithms
    - [Synchronous Stochastic Gradient Descent](https://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf)
    - [Model Average Method](https://www.aclweb.org/anthology/N10-1069.pdf)
    - [Block-wise Model Update Filtering](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf)
    - [Elastic Average Stochastic Gradient Decent](https://arxiv.org/abs/1412.6651)

- Asynchronous Training Algorithms
    - Asynchronous SGD


#### Get started

```bash
pip install -r requirements.txt

# add tinynn to python path
PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/tinynn python distributed-ml/sync_run.py --algo SSGD
```


