import numpy as np
import matplotlib.pyplot as plt

# 实现自动微分
# 参考
# https://zhuanlan.zhihu.com/p/161635270
# https://github.com/dlsys-course/assignment1-2018/blob/master/autodiff.py
eps = 1e-10

class Op:
    # dfs 实现拓扑排序
    def _dfs(self):
        ord_nodes = [] # 保存拓扑排序结果
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
        self.nodes = (node, )
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
        self.nodes = (node, )
    def back_calc_grad(self):
        # n2 = sin(n1)
        # dn2 / dn1 = cos(n1)
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * np.cos(self.nodes[0]._v)

class Cos(Op):
    def __init__(self, node):
        self._v = np.cos(node._v)
        self._d = 0.0
        self.nodes = (node, )
    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d -= self._d * np.sin(self.nodes[0]._v)

class Exp(Op):
    def __init__(self, node):
        self._v = np.exp(node._v)
        self._d = 0.0
        self.nodes = (node, )
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
        self.nodes = (node, )
    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * self._g / self.nodes[0]._v

class Sum(Op):
    def __init__(self, node):
        self._v = np.sum(node._v)
        self._d = 0.0
        self.nodes = (node, )
    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d * np.ones_like(self.nodes[0]._v)

class SumAxis(Op):
    def __init__(self, node, axis=1):
        # 严格来说应该np.repeat, 考虑到numpy的广播机制，不加入repeat
        self._v = np.sum(node._v, axis=axis, keepdims=True)
        self._d = 0.0
        self._sumaxis = axis
        self.nodes = (node, )
    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += np.sum(self._d, axis=self._sumaxis, keepdims=True)

def OpInit2Func(OpClass):
    return lambda *args, **kwargs : OpClass(*args, **kwargs)

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
def test_softmax():
    y = C(np.eye(4)[np.random.randint(0, 4, 15)])
    z = C(np.random.uniform(-10, 10, (15, 4)), requires_grad=True)
    z_exp = Exp(z)
    a = z_exp / SumAxis(z_exp, axis=1)
    L = Sum(y * Log(a))
    L.backward()
    print(np.c_[y._v - a._v, z._d])

if __name__ == '__main__':
    test_softmax()
