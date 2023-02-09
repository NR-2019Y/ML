import numpy as np
import abc
import numbers
import typing
from typing import Union, Tuple
from collections.abc import Iterable
import matplotlib.pyplot as plt
from deprecated import deprecated

# 实现反向自动微分
# 参考
# https://zhuanlan.zhihu.com/p/161635270
# https://github.com/dlsys-course/assignment1-2018/blob/master/autodiff.py

eps = 1e-10


class NoTraining:
    __state_is_training = True

    def __enter__(self):
        NoTraining.__state_is_training = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        NoTraining.__state_is_training = True

    @staticmethod
    def mode_is_training():
        return NoTraining.__state_is_training


mode_is_training = NoTraining.mode_is_training


def get_reduced_result(ori, reduction):
    if reduction is None:
        return ori
    else:
        assert callable(reduction)
        return reduction(ori)


# 前向传播：将梯度置0
# Op.backward : 图遍历，计算梯度

class Op:
    _d: Union[numbers.Integral, np.ndarray]
    _v: Union[numbers.Integral, np.ndarray]
    nodes: Tuple[Op, ...]

    # dfs 实现拓扑排序
    def _dfs(self):
        ord_nodes = []  # 保存拓扑排序结果
        visited = set()
        curr_path = []
        q = [(0, self)]
        while q:
            layer, node = q.pop()
            if node in visited:
                continue
            while len(curr_path) > layer:
                ord_nodes.append(curr_path.pop())
            visited.add(node)
            curr_path.append(node)
            if hasattr(node, "nodes"):
                for nnd in node.nodes:
                    if nnd not in visited:
                        q.append((layer + 1, nnd))
        while curr_path:
            ord_nodes.append(curr_path.pop())
        ord_nodes.reverse()
        self.ord_nodes = ord_nodes

    def backward(self):
        assert hasattr(self, "_d")
        if isinstance(self._v, np.ndarray):
            self._d = np.ones_like(self._v)
        else:
            self._d = 1.0
        self._dfs()
        for node in self.ord_nodes:
            if hasattr(node, "nodes"):
                node.back_calc_grad()

    def back_calc_grad(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._v)

    @property
    def shape(self):
        return np.shape(self._v)

    @property
    def ndim(self):
        return np.ndim(self._v)

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __neg__(self):
        return Neg(self)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return TrueDiv(self, other)

    def __matmul__(self, other):
        return MatMul(self, other)

    def __pow__(self, power, modulo=None):
        return Pow(self, power)

    def __getitem__(self, item):
        return GetItem(item)


class C(Op):
    def __init__(self, val, requires_grad=False, name=None):
        if isinstance(val, np.ndarray) and (val.ndim == 0):
            val = np.sum(val)
        self._v = val
        if requires_grad:
            self._d = 0.0
        if name is not None:
            self.name = name


class Add(Op):
    def __init__(self, node1, node2):
        self._v = node1._v + node2._v
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        if hasattr(node1, "_d"):
            node1._d += self._d
        if hasattr(node2, "_d"):
            node2._d += self._d


class Sub(Op):
    def __init__(self, node1, node2):
        self._v = node1._v - node2._v
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        if hasattr(node1, "_d"):
            node1._d += self._d
        if hasattr(node2, "_d"):
            node2._d -= self._d


class Neg(Op):
    def __init__(self, node):
        self._v = -node._v
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d -= self._d


class Mul(Op):
    def __init__(self, node1, node2):
        self._v = node1._v * node2._v
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        if hasattr(node1, "_d"):
            node1._d += self._d * node2._v
        if hasattr(node2, "_d"):
            node2._d += self._d * node1._v


class TrueDiv(Op):
    def __init__(self, node1, node2):
        self._v = node1._v / node2._v
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        # n3 = n1 / n2
        # dn3 / dn1 = 1 / n2
        # dn3 / dn2 = - n1 / (n2 ** 2)
        node1, node2 = self.nodes
        if hasattr(node1, "_d"):
            node1._d += self._d / node2._v
        if hasattr(node2, "_d"):
            node2._d -= self._d * node1._v / (node2._v ** 2 + eps)


def broad_cast_grad(pgrad, shape, pshape):
    if shape == pshape:
        return pgrad
    if not shape:
        return np.sum(pgrad)
    if len(shape) < len(pshape):
        assert len(shape) == 1
        return np.sum(pgrad, axis=tuple(range(len(pshape) - 1)))
    assert len(shape) == len(pshape)
    sum_axis = tuple(i for i, (sz1, sz2) in enumerate(zip(shape, pshape)) if sz1 < sz2)
    return np.sum(pgrad, axis=sum_axis, keepdims=True)


class BroadcastAdd(Op):
    def __init__(self, node1, node2):
        self._v = node1._v + node2._v
        self._d = 0.0
        self._shape1 = np.shape(node1._v)
        self._shape2 = np.shape(node2._v)
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        pshape = np.shape(self._v)

        if hasattr(node1, "_d"):
            node1._d += broad_cast_grad(self._d, self._shape1, pshape)
        if hasattr(node2, "_d"):
            node2._d += broad_cast_grad(self._d, self._shape2, pshape)


class BroadcastSub(Op):
    def __init__(self, node1, node2):
        self._v = node1._v - node2._v
        self._d = 0.0
        self._shape1 = np.shape(node1._v)
        self._shape2 = np.shape(node2._v)
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        pshape = np.shape(self._v)

        if hasattr(node1, "_d"):
            node1._d += broad_cast_grad(self._d, self._shape1, pshape)
        if hasattr(node2, "_d"):
            node2._d -= broad_cast_grad(self._d, self._shape2, pshape)


class BroadcastMul(Op):
    def __init__(self, node1: Op, node2: Op):
        self._v = node1._v * node2._v
        self._d = 0.0
        self._shape1 = node1.shape
        self._shape2 = node2.shape
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        pshape = np.shape(self._v)
        if hasattr(node1, "_d"):
            node1._d += broad_cast_grad(self._d * node2._v, self._shape1, pshape)
        if hasattr(node2, "_d"):
            node2._d += broad_cast_grad(self._d * node1._v, self._shape2, pshape)


class BroadcastDiv(Op):
    def __init__(self, node1, node2):
        self._v = node1._v / node2._v
        self._d = 0.0
        self._shape1 = np.shape(node1._v)
        self._shape2 = np.shape(node2._v)
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        pshape = np.shape(self._v)

        if hasattr(node1, "_d"):
            node1._d += broad_cast_grad(self._d / node2._v, self._shape1, pshape)
        if hasattr(node2, "_d"):
            node2._d -= broad_cast_grad(self._d * node1._v / (node2._v ** 2 + eps), self._shape2, pshape)


class MatMul(Op):
    def __init__(self, node1, node2):
        assert node1.ndim >= 2 and node2.ndim == 2
        self._v = np.dot(node1._v, node2._v)
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        if hasattr(node1, "_d"):
            node1._d += np.dot(self._d, node2._v.T)
        if hasattr(node2, "_d"):
            node2._d += np.dot(
                node1._v.reshape((-1, node1.shape[-1])).T,
                self._d.reshape((-1, self.shape[-1]))
            )


class Sin(Op):
    def __init__(self, node):
        self._v = np.sin(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        # n2 = sin(n1)
        # dn2 / dn1 = cos(n1)
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * np.cos(self.nodes[0]._v)


class Cos(Op):
    def __init__(self, node):
        self._v = np.cos(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d -= self._d * np.sin(self.nodes[0]._v)


class Exp(Op):
    def __init__(self, node):
        self._v = np.exp(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        # n2 = exp(n1)
        # dn2 / dn1 = n2
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._v


class Square(Op):
    def __init__(self, node: Op):
        self._v = np.square(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += 2.0 * self._d * self.nodes[0]._v


class Sqrt(Op):
    def __init__(self, node: Op):
        self._v = np.sqrt(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += 0.5 * self._d / (self._v + eps)


class Pow(Op):
    def __init__(self, node1, node2):
        self._v = node1._v ** node2._v
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        # n3 = n1 ** n2
        # dn3 / dn1 = n2 * n1 ** (n2 - 1) = n3 * n2 / n1
        # dn3 / dn2 = n1 ** n2 * ln(n1)
        if hasattr(node1, "_d"):
            node1._d += self._d * self._v * node2._v / (node1._v + eps)
            # node1._d += self._d * node2._v * node1._v ** (node2._v - 1)
        if hasattr(node2, "_d"):
            node2._d += self._d * self._v * np.log(node1._v)


class Log(Op):
    def __init__(self, node, base=np.e):
        self._g = 1.0 / np.log(base)
        self._v = self._g * np.log(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._g / self.nodes[0]._v


class ReduceSum(Op):
    def __init__(self, node, axis=None, keepdims=False):
        self._v = np.sum(node._v, axis=axis, keepdims=keepdims)
        if mode_is_training():
            self._d = 0.0
            self.nodes = (node,)
            self._axis_reduce = axis
            if keepdims:
                self._axis_reduce = None
            else:
                self._axis_reduce = axis

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            if self._axis_reduce is None:
                self.nodes[0]._d += self._d * np.ones_like(self.nodes[0]._v)
            else:
                self.nodes[0]._d += np.expand_dims(self._d, self._axis_reduce) * np.ones_like(self.nodes[0]._v)


class ReduceMean(Op):
    def __init__(self, node, axis=None, keepdims=False):
        self._v = np.mean(node._v, axis=axis, keepdims=keepdims)
        if mode_is_training():
            self._d = 0.0
            self.nodes = (node,)
            self._axis_reduce = axis
            if isinstance(axis, numbers.Integral):
                axis = (axis,)
            self._scale_size = 1. / np.prod(node.shape) if axis is None else 1. / np.prod([node.shape[i] for i in axis])
            if keepdims:
                self._axis_reduce = None
            else:
                self._axis_reduce = axis

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            if self._axis_reduce is None:
                self.nodes[0]._d += self._scale_size * self._d * np.ones_like(self.nodes[0]._v)
            else:
                self.nodes[0]._d += self._scale_size * np.expand_dims(self._d, self._axis_reduce) \
                                    * np.ones_like(self.nodes[0]._v)


class ReduceVar(Op):
    def __init__(self, node, axis=None, keepdims=False):
        if not mode_is_training():
            self._v = np.var(node._v, axis=axis, keepdims=keepdims)
        else:
            x_scale = node._v - np.mean(node._v, axis=axis, keepdims=True)
            self._cache_x_scale = x_scale
            self._v = np.mean(np.square(x_scale), axis=axis, keepdims=keepdims)
            self._d = 0.0
            self.nodes = (node,)
            self._axis_reduce = axis
            if isinstance(axis, numbers.Integral):
                axis = (axis,)
            scale_size = 2. / np.prod(node.shape) if axis is None else 2. / np.prod([node.shape[i] for i in axis])
            self._cache_x_scale *= scale_size
            if keepdims:
                self._axis_reduce = None
            else:
                self._axis_reduce = axis

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            if self._axis_reduce is None:
                self.nodes[0]._d += self._d * self._cache_x_scale
            else:
                self.nodes[0]._d += np.expand_dims(self._d, self._axis_reduce) * self._cache_x_scale
            del self._cache_x_scale


class Scaler(Op):
    def __init__(self, node: Op, axis=None):
        x_mean = np.mean(node._v, axis=axis, keepdims=True)
        x_sc = node._v - x_mean
        x_var = np.mean(np.square(x_sc), axis=axis, keepdims=True)
        x_std = np.sqrt(x_var)
        self._v = x_sc / (x_std + eps)
        if mode_is_training():
            self._d = 0.0
            self._bn_mean = x_mean
            self._bn_var = x_var
            self._cache_val = x_std
            self._arg_axis = axis
            self.nodes = (node,)

    def get_grad(self, grad):
        x_std = self._cache_val
        d = grad / x_std - self._v * np.mean(grad * self._v, axis=self._arg_axis, keepdims=True) / x_std
        d -= np.mean(d, axis=self._arg_axis, keepdims=True)
        del self._cache_val
        return d

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0] += self.get_grad(self._d)


class Relu(Op):
    def __init__(self, node):
        self._v = np.maximum(node._v, 0.0)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * (self._v > 0).astype(np.float64)


class Sigmoid(Op):
    def __init__(self, node):
        self._v = 1.0 / (1.0 + np.exp(-node._v))
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._v * (1 - self._v)


class Tanh(Op):
    def __init__(self, node):
        self._v = np.tanh(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * (1.0 - np.square(self._v))


def np_softmax(z):
    z_exp_sc = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return z_exp_sc / np.sum(z_exp_sc, axis=-1, keepdims=True)


class Softmax(Op):
    def __init__(self, node):
        self._v = np_softmax(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._v - np.sum(self._d * np.square(self._v), axis=-1, keepdims=True)


class HingleLoss(Op):
    def __init__(self, node_logits, node_y_sig):
        assert not hasattr(node_y_sig, "_d")
        assert node_logits.shape == node_y_sig.shape
        self._scale = 1. / np.prod(node_logits.shape[:-1])
        self._loss_before_reduce = np.maximum(1. - node_y_sig._v * node_logits._v, 0.)
        self._ysig = node_y_sig._v
        self._v = self._scale * np.sum(self._loss_before_reduce)
        self._d = 0.
        self.nodes = (node_logits,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += -self._scale * self._ysig * (self._loss_before_reduce > 0.)


class CrossEntropyLossLayer(Op):
    def __init__(self, node_logits, node_y):
        assert not hasattr(node_y, "_d")
        self._n_classes = node_logits.shape[-1]
        proba = np_softmax(node_logits._v)
        self._proba = proba
        self._scale_size = np.prod(node_logits.shape[:-1])
        proba_flat = proba.reshape((-1, self._n_classes))
        y_flat = node_y._v.ravel()
        self._v = -np.mean(np.log(proba_flat[range(self._scale_size), y_flat] + eps))
        self._d = 0.0
        self.nodes = (node_logits, node_y)

    def back_calc_grad(self):
        node_logits, node_y = self.nodes
        node_logits._d += (1.0 / self._scale_size) * self._d * (
                self._proba - np.eye(self._n_classes, dtype=np.float64)[node_y._v]
        )


class LinearLayer(Op):
    def __init__(self, node_x, node_w, node_b):
        assert node_x.ndim >= 2
        assert node_w.ndim == 2
        assert node_b.ndim == 1
        assert hasattr(node_w, "_d") and hasattr(node_b, "_d")
        self._v = np.dot(node_x._v, node_w._v) + node_b._v
        self._d = 0.0
        self.nodes = (node_x, node_w, node_b)

    def back_calc_grad(self):
        node_x, node_w, node_b = self.nodes
        if hasattr(node_x, "_d"):
            node_x._d += np.dot(self._d, node_w._v.T)
        node_w._d += np.dot(
            node_x._v.reshape((-1, node_x.shape[-1])).T,
            self._d.reshape((-1, self.shape[-1]))
        )
        node_b._d += np.sum(self._d, axis=tuple(range(node_x.ndim - 1)))


@deprecated('Instead, use AddBiasND')
class AddBias2D(Op):
    def __init__(self, node_ori: Op, node_b: Op):
        assert node_ori.ndim >= 2
        assert node_b.ndim == 1
        self._v = node_ori._v + node_b._v
        self._d = 0.0
        self.nodes = (node_ori, node_b)

    def back_calc_grad(self):
        node_ori, node_b = self.nodes
        if hasattr(node_ori, "_d"):
            node_ori._d += self._d
        if hasattr(node_b, "_d"):
            node_b._d += np.sum(self._d, axis=0)


class AddBiasND(Op):
    def __init__(self, node_ori: Op, node_b: Op):
        assert node_ori.ndim >= 2
        assert node_b.ndim == 1
        self._v = node_ori._v + node_b._v
        self._d = 0.0
        self.nodes = (node_ori, node_b)

    def back_calc_grad(self):
        node_ori, node_b = self.nodes
        if hasattr(node_ori, "_d"):
            node_ori._d += self._d
        if hasattr(node_b, "_d"):
            node_b._d += np.sum(self._d, axis=tuple(range(node_ori.ndim - 1)))


class GetItem(Op):
    def __init__(self, node: Op, arg):
        self._v = node._v[arg]
        self._d = 0
        self.arg = arg
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            if not isinstance(self.nodes[0]._d, np.ndarray):
                self.nodes[0]._d = np.zeros_like(self.nodes[0]._v)
            self.nodes[0]._d[self.arg] += self._d


class ListToTensor(Op):
    def __init__(self, all_nodes):
        self._v = np.array([node._v for node in all_nodes])
        self._d = 0.0
        self.nodes = tuple(all_nodes)

    def back_calc_grad(self):
        for node, grad in zip(self.nodes, self._d):
            if hasattr(node, "_d"):
                node._d += grad


def inv_perm(perm: typing.Union[typing.List, typing.Tuple]):
    new_perm = [0] * len(perm)
    for idx1, idx2 in enumerate(perm):
        new_perm[idx2] = idx1
    return new_perm


class Transpose(Op):
    def __init__(self, node, axes):
        self._v = np.transpose(node._v, axes)
        self._d = 0.0
        self.inv_axes_arg = None if axes is None else inv_perm(axes)
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += np.transpose(self._d, self.inv_axes_arg)


class Reshape(Op):
    def __init__(self, node: Op, new_shape):
        self._ori_shape = node.shape
        self._v = np.reshape(node._v, new_shape)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += np.reshape(self._d, self._ori_shape)


def Flatten(node: Op):
    assert node.ndim >= 2
    if node.ndim == 2:
        return node
    return Reshape(node, (node.shape[0], -1))


class Dropout(Op):
    def __init__(self, node: Op, *, rate):
        if not mode_is_training():
            self._v = node._v
        else:
            p = 1 - rate
            self._v_mul = (1.0 / p) * np.random.binomial(1, p, size=node.shape)
            self._v = node._v * self._v_mul
            self._d = 0.0
            self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._v_mul


class RegularizerL2Wrap(Op):
    def __init__(self, node: Op, lam_and_node: Iterable, *, calc_regularizer_loss=False):
        if not calc_regularizer_loss:
            self._v = node._v
        else:
            assert node.ndim == 0
            self._v = node._v
            for lam, nd in lam_and_node:
                self._v = self._v + 0.5 * lam * np.sum(np.square(nd._v))
        self._d = 0.0
        self.nodes = (node,)
        self._lam_and_node = lam_and_node

    def back_calc_grad(self):
        for lam, node in self._lam_and_node:
            assert not hasattr(node, "nodes") and hasattr(node, "_d")
            node._d += lam * node._v
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d


# x = C(np.linspace(-5, 5, 1000), requires_grad=True)
# y = C(1.0) / (C(1.0) + Exp(-x))
# y.backward()
# # plt.plot(x._v, x._d)
# plt.plot(y._v, x._d)
# plt.show()

# # softmax 测试 OK
def my_test_softmax():
    y = C(np.eye(4)[np.random.randint(0, 4, 15)])
    z = C(np.random.uniform(-10, 10, (15, 4)), requires_grad=True)
    z_exp = Exp(z)
    a = BroadcastDiv(z_exp, ReduceSum(z_exp, axis=1, keepdims=True))
    L = ReduceMean(ReduceSum(y * Log(a), axis=1))
    L.backward()
    print(z._d / (y._v - a._v))


if __name__ == '__main__':
    my_test_softmax()
