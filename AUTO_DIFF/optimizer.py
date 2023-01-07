import numpy as np

from . import my_auto_grad_v0 as op
from typing import Sequence

eps = 1e-10


class VariableNumber(object):
    def __init__(self, value):
        self.value = value


class SGD:
    def __init__(self, *, learning_rate, trainable_nodes: Sequence[op.Op]):
        self.trainable_nodes = trainable_nodes
        self.learning_rate = learning_rate

    def __call__(self):
        for node in self.trainable_nodes:
            node._v -= self.learning_rate * node._d


class AdaGrad:
    def __init__(self, *, learning_rate, trainable_nodes: Sequence[op.Op]):
        self.trainable_nodes = trainable_nodes
        self.learning_rate = learning_rate
        self.s_init_list = [np.zeros_like(node._v) for node in trainable_nodes]

    def __call__(self):
        for node, s_val in zip(self.trainable_nodes, self.s_init_list):
            assert isinstance(s_val, np.ndarray)
            s_val += np.square(node._d)
            node._v -= (self.learning_rate / (np.sqrt(s_val) + eps)) * node._d


def to_zeros_array(arr):
    return VariableNumber(np.zeros_like(arr))


class Momentum:
    def __init__(self, *, learning_rate, beta=0.9, trainable_nodes: Sequence[op.Op]):
        self.trainable_nodes = trainable_nodes
        self.learning_rate = learning_rate
        self.beta = beta
        self.nbeta = 1.0 - beta
        self.v_list = [to_zeros_array(node._v) for node in trainable_nodes]

    def __call__(self):
        for vobj, node in zip(self.v_list, self.trainable_nodes):
            vobj.value = self.beta * vobj.value + self.nbeta * node._d
            node._v -= self.learning_rate * vobj.value


class Rmsprop:
    def __init__(self, *, learning_rate, beta=0.9, trainable_nodes: Sequence[op.Op]):
        self.trainable_nodes = trainable_nodes
        self.learning_rate = learning_rate
        self.beta = beta
        self.nbeta = 1.0 - beta
        self.s_list = [to_zeros_array(node._v) for node in trainable_nodes]

    def __call__(self):
        for sobj, node in zip(self.s_list, self.trainable_nodes):
            sobj.value = self.beta * sobj.value + self.nbeta * np.square(node._d)
            node._v -= (self.learning_rate / (np.sqrt(sobj.value) + eps)) * node._d


# http://zh-v2.d2l.ai/chapter_optimization/adam.html
class Adam:
    def __init__(self, *, learning_rate, beta1=0.9, beta2=0.999, trainable_nodes: Sequence[op.Op]):
        self.trainable_nodes = trainable_nodes
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.nbeta1 = 1.0 - beta1
        self.beta2 = beta2
        self.nbeta2 = 1.0 - beta2
        self.t = 1
        self.v_list = [to_zeros_array(node._v) for node in trainable_nodes]
        self.s_list = [to_zeros_array(node._v) for node in trainable_nodes]

    def __call__(self):
        for vobj, sobj, node in zip(self.v_list, self.s_list, self.trainable_nodes):
            vobj.value = self.beta1 * vobj.value + self.nbeta1 * node._d
            sobj.value = self.beta2 * sobj.value + self.nbeta2 * np.square(node._d)
            v_sc = vobj.value / (1.0 - np.power(self.beta1, self.t))
            s_sc = sobj.value / (1.0 - np.power(self.beta2, self.t))
            node._v -= (self.learning_rate / (np.sqrt(s_sc) + eps)) * v_sc
        self.t += 1
