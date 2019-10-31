# tinynn packages
from core.layer import Dense
from core.layer import ReLU
from core.loss import SoftmaxCrossEntropy
from core.model import Model
from core.net import Net
from core.optimizer import Adam


def get_mlp(lr):
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
    return model
