import numpy as np
import abc
import matplotlib.pyplot as plt

# 实现反向自动微分
# 参考
# https://zhuanlan.zhihu.com/p/161635270
# https://github.com/dlsys-course/assignment1-2018/blob/master/autodiff.py
eps = 1e-10


# 前向传播：将梯度置0
# Op.backward : 图遍历，计算梯度

class Op:
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
        self._d = 1.0
        self._dfs()
        for node in self.ord_nodes:
            if hasattr(node, "nodes"):
                node.back_calc_grad()

    @abc.abstractmethod
    def back_calc_grad(self):
        pass


class C(Op):
    def __init__(self, val, requires_grad=False):
        self._v = val
        if requires_grad:
            self._d = 0.0


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


class MatMul(Op):
    def __init__(self, node1, node2):
        self._v = np.dot(node1._v, node2._v)
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        if hasattr(node1, "_d"):
            node1._d += np.dot(self._d, node2._v.T)
        if hasattr(node2, "_d"):
            node2._d += np.dot(node1._v.T, self._d)


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


class Sum(Op):
    def __init__(self, node):
        self._v = np.sum(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * np.ones_like(self.nodes[0]._v)


class SumAxis(Op):
    def __init__(self, node, axis=1):
        # 严格来说应该np.repeat, 考虑到numpy的广播机制，不加入repeat
        self._v = np.sum(node._v, axis=axis, keepdims=True)
        self._d = 0.0
        self._sumaxis = axis
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += np.sum(self._d, axis=self._sumaxis, keepdims=True)


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
    z_exp_sc = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z_exp_sc / np.sum(z_exp_sc, axis=1, keepdims=True)


class Softmax(Op):
    def __init__(self, node):
        self._v = np_softmax(node._v)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._v - np.sum(self._d * np.square(self._v), axis=1, keepdims=True)


class CrossEntropyLossLayer(Op):
    def __init__(self, node_logits, node_y):
        self.batch_size, self.n_classes = node_logits._v.shape
        proba = np_softmax(node_logits._v)
        self._proba = proba
        self._v = -np.mean(np.log(proba[range(self.batch_size), node_y._v] + eps))
        self._d = 0.0
        self.nodes = (node_logits, node_y)

    def back_calc_grad(self):
        node_logits, node_y = self.nodes
        assert not hasattr(node_y, "_d")
        node_logits._d += (1.0 / self.batch_size) * self._d * (
                self._proba - np.eye(self.n_classes, dtype=np.float64)[node_y._v])


# class CrossEntropyLossLayer(Op):
#     def __init__(self, node_logits, node_y):
#         self.batch_size, self.n_classes = node_logits._v.shape
#         proba = np_softmax(node_logits._v)
#         self._proba = proba
#         self._v = -(1.0 / self.batch_size) * np.sum(node_y._v * np.log(proba + eps))
#         self._d = 0.0
#         self.nodes = (node_logits, node_y)
#
#     def back_calc_grad(self):
#         node_logits, node_y = self.nodes
#         assert not hasattr(node_y, "_d")
#         node_logits._d += (1.0 / self.batch_size) * self._d * (self._proba - node_y._v)


class LinearLayer(Op):
    def __init__(self, node_x, node_w, node_b):
        self._v = np.dot(node_x._v, node_w._v) + node_b._v
        self._d = 0.0
        self.nodes = (node_x, node_w, node_b)

    def back_calc_grad(self):
        node_x, node_w, node_b = self.nodes
        if hasattr(node_x, "_d"):
            node_x._d += np.dot(self._d, node_w._v.T)
        assert hasattr(node_w, "_d") and hasattr(node_b, "_d")
        node_w._d += np.dot(node_x._v.T, self._d)
        node_b._d += np.sum(self._d, axis=0)


class AddBias2D(Op):
    def __init__(self, node_ori, node_b):
        self._v = node_ori._v + node_b._v
        self._d = 0.0
        self.nodes = (node_ori, node_b)

    def back_calc_grad(self):
        node_ori, node_b = self.nodes
        if hasattr(node_ori, "_d"):
            node_ori._d += self._d
        if hasattr(node_b, "_d"):
            node_b._d += np.sum(self._d, axis=0)


class ListToTensor(Op):
    def __init__(self, all_nodes):
        self._v = np.array([node._v for node in all_nodes])
        self._d = 0.0
        self.nodes = tuple(all_nodes)

    def back_calc_grad(self):
        for node, grad in zip(self.nodes, self._d):
            if hasattr(node, "_d"):
                node._d += grad


class Transpose(Op):
    def __init__(self, node, axes):
        self._v = np.transpose(node._v, axes)
        self._d = 0.0
        self.axes_arg = axes
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += np.transpose(self._d, self.axes_arg)


class Reshape(Op):
    def __init__(self, node, new_shape):
        self._ori_shape = node._v.shape
        # print("self._ori_shape", self._ori_shape)
        self._v = np.reshape(node._v, new_shape)
        # print("self._v.shape", self._v.shape)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += np.reshape(self._d, self._ori_shape)


def OpInit2Func(OpClass):
    return lambda *args, **kwargs: OpClass(*args, **kwargs)


Op.__add__ = OpInit2Func(Add)
Op.__sub__ = OpInit2Func(Sub)
Op.__neg__ = OpInit2Func(Neg)
Op.__mul__ = OpInit2Func(Mul)
Op.__truediv__ = OpInit2Func(TrueDiv)
Op.__matmul__ = OpInit2Func(MatMul)
Op.__pow__ = OpInit2Func(Pow)


# x1 = C(2.0, requires_grad=True)
# x2 = C(3.0, requires_grad=True)
# z = (x1 ** C(2)) * x2 + x2 + C(2.0)
# z.backward()
# print(x1._d, x2._d)

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
    a = z_exp / SumAxis(z_exp, axis=1)
    L = Sum(y * Log(a))
    L.backward()
    print((y._v - a._v) / z._d)


if __name__ == '__main__':
    my_test_softmax()
