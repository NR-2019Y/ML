from AUTO_DIFF import my_auto_grad_v0 as op
import numpy as np
import numba
import math
from typing import Union
import numbers


@numba.njit
def conv2d(X, W):
    h, w = W.shape[0], W.shape[1]
    Y = np.empty((X.shape[0], X.shape[1] - h + 1, X.shape[2] - w + 1, W.shape[-1]), dtype=X.dtype)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                for l in range(Y.shape[3]):
                    Y[i, j, k, l] = np.sum(X[i, j:j + h, k:k + w, :] * W[:, :, :, l])
    return Y


@numba.njit
def grad_conv2d_dx_and_dw(X, W, dy):
    h, w = W.shape[0], W.shape[1]
    y_shape = (X.shape[0], X.shape[1] - h + 1, X.shape[2] - w + 1, W.shape[-1])
    dx = np.zeros_like(X)
    dw = np.zeros_like(W)
    for i in range(y_shape[0]):
        for j in range(y_shape[1]):
            for k in range(y_shape[2]):
                for l in range(y_shape[3]):
                    dw[:, :, :, l] += dy[i, j, k, l] * X[i, j:j + h, k:k + w, :]
                    dx[i, j:j + h, k:k + w, :] += dy[i, j, k, l] * W[:, :, :, l]
    return dx, dw


@numba.njit
def grad_conv2d_dx(X, W, dy):
    h, w = W.shape[0], W.shape[1]
    y_shape = (X.shape[0], X.shape[1] - h + 1, X.shape[2] - w + 1, W.shape[-1])
    dx = np.zeros_like(X)
    for i in range(y_shape[0]):
        for j in range(y_shape[1]):
            for k in range(y_shape[2]):
                for l in range(y_shape[3]):
                    dx[i, j:j + h, k:k + w, :] += dy[i, j, k, l] * W[:, :, :, l]
    return dx


@numba.njit
def grad_conv2d_dw(X, W, dy):
    h, w = W.shape[0], W.shape[1]
    y_shape = (X.shape[0], X.shape[1] - h + 1, X.shape[2] - w + 1, W.shape[-1])
    dw = np.zeros_like(W)
    for i in range(y_shape[0]):
        for j in range(y_shape[1]):
            for k in range(y_shape[2]):
                for l in range(y_shape[3]):
                    dw[:, :, :, l] += dy[i, j, k, l] * X[i, j:j + h, k:k + w, :]
    return dw


class Conv2D(op.Op):
    def __init__(self, node1: op.Op, node2: op.Op):
        '''

        :param node1: X (image)
        :param node2: W (卷积核)
        '''
        self._v = conv2d(node1._v, node2._v)
        self._d = 0.0
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        if hasattr(node1, "_d") and hasattr(node2, "_d"):
            dx, dw = grad_conv2d_dx_and_dw(node1._v, node2._v, self._d)
            node1._d += dx
            node2._d += dw
        elif hasattr(node1, "_d"):
            node1._d += grad_conv2d_dx(node1._v, node2._v, self._d)
        elif hasattr(node2, "_d"):
            node2._d += grad_conv2d_dw(node1._v, node2._v, self._d)


@numba.njit
def avg_pool2d(X: np.ndarray, *, kh, kw, sh, sw):
    osize = X.shape[0], X.shape[1] - kh + 1, X.shape[2] - kw + 1, X.shape[3]
    Y = np.empty((osize[0], math.ceil(osize[1] / sh), math.ceil(osize[2] / sw), osize[3]), dtype=X.dtype)
    for i in range(osize[0]):
        for j in range(0, osize[1], sh):
            for k in range(0, osize[2], sw):
                for l in range(osize[3]):
                    Y[i, j // sh, k // sw, l] = np.mean(X[i, j:j + kh, k:k + kw, l])
    return Y


@numba.njit
def grad_avg_pool2d(X: np.ndarray, *, kh, kw, sh, sw, dy: np.ndarray):
    dx = np.zeros_like(X)
    osize = X.shape[0], X.shape[1] - kh + 1, X.shape[2] - kw + 1, X.shape[3]
    keach = (1.0 / kh / kw)
    for i in range(osize[0]):
        for j in range(0, osize[1], sh):
            for k in range(0, osize[2], sw):
                for l in range(osize[3]):
                    dx[i, j:j + kh, k:k + kw, l] += keach * dy[i, j // sh, k // sw, l]
    return dx


class AvgPool2D(op.Op):
    def __init__(self, node: op.Op, ksize: Union[tuple, list], stride: Union[tuple, list]):
        kh, kw = ksize
        sh, sw = stride
        self._args = kh, kw, sh, sw
        self._v = avg_pool2d(node._v, kh=kh, kw=kw, sh=sh, sw=sw)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            kh, kw, sh, sw = self._args
            self.nodes[0]._d += grad_avg_pool2d(self.nodes[0]._v, kh=kh, kw=kw, sh=sh, sw=sw, dy=self._d)


@numba.njit
def max_pool2d(X: np.ndarray, *, kh, kw, sh, sw):
    osize = X.shape[0], X.shape[1] - kh + 1, X.shape[2] - kw + 1, X.shape[3]
    Y = np.empty((osize[0], math.ceil(osize[1] / sh), math.ceil(osize[2] / sw), osize[3]), dtype=X.dtype)
    for i in range(osize[0]):
        for j in range(0, osize[1], sh):
            for k in range(0, osize[2], sw):
                for l in range(osize[3]):
                    Y[i, j // sh, k // sw, l] = np.max(X[i, j:j + kh, k:k + kw, l])
    return Y


@numba.jit
def max_pool2d_v2(X: np.ndarray, *, kh, kw, sh, sw):
    osize = X.shape[0], X.shape[1] - kh + 1, X.shape[2] - kw + 1, X.shape[3]
    sz_mul0, sz_mul1, sz_mul2 = X.shape[1] * X.shape[2] * X.shape[3], X.shape[2] * X.shape[3], X.shape[3]
    Y = np.empty((osize[0], math.ceil(osize[1] / sh), math.ceil(osize[2] / sw), osize[3]), dtype=X.dtype)
    maxidxs = np.empty_like(Y, dtype=np.int64)
    for i in range(osize[0]):
        for j in range(0, osize[1], sh):
            for k in range(0, osize[2], sw):
                for l in range(osize[3]):
                    cmp_val = X[i, j, k, l]
                    jidx, kidx = j, k
                    for j1 in range(j, j + kh):
                        for k1 in range(k, k + kw):
                            curr_val = X[i, j1, k1, l]
                            if curr_val > cmp_val:
                                jidx, kidx = j1, k1
                                cmp_val = curr_val
                    jy = j // sh
                    ky = k // sw
                    Y[i, jy, ky, l] = cmp_val
                    maxidxs[i, jy, ky, l] = i * sz_mul0 + jidx * sz_mul1 + kidx * sz_mul2 + l
    return maxidxs, Y


@numba.njit
def grad_max_pool2d(X: np.ndarray, maxidxs: np.ndarray, dy: np.ndarray):
    dx = np.zeros_like(X)
    dx_ravel = dx.ravel()
    # osize = dy.shape
    # for i in range(osize[0]):
    #     for j in range(osize[1]):
    #         for k in range(osize[2]):
    #             for l in range(osize[3]):
    #                 dx_ravel[maxidxs[i, j, k, l]] += dy[i, j, k, l]
    for curr_idx, curr_d in zip(maxidxs.ravel(), dy.ravel()):
        dx_ravel[curr_idx] += curr_d
    return dx


class MaxPool2D(op.Op):
    def __init__(self, node: op.Op, ksize: Union[tuple, list], stride: Union[tuple, list]):
        kh, kw = ksize
        sh, sw = stride
        self._idxs_store, self._v = max_pool2d_v2(node._v, kh=kh, kw=kw, sh=sh, sw=sw)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += grad_max_pool2d(self.nodes[0]._v, self._idxs_store, self._d)


def _pad_const(x: np.ndarray, *, pad_width, constant_values=0.):
    xpad = np.pad(x, pad_width, constant_values=constant_values)
    if isinstance(pad_width, numbers.Number):
        pad_before_arr = [pad_width] * x.ndim
    else:
        pad_before_arr = [v[0] for v in pad_width]
    slice_args = tuple(slice(i, i + size) for i, size in zip(pad_before_arr, x.shape))
    return slice_args, xpad


class PadConst(op.Op):
    def __init__(self, node: op.Op, *, pad_width, constant_values=0.):
        self._slice_args, self._v = _pad_const(node._v, pad_width=pad_width, constant_values=constant_values)
        self._d = 0.0
        self.nodes = (node,)

    def back_calc_grad(self):
        if hasattr(self.nodes[0], "_d"):
            self.nodes[0]._d += self._d[self._slice_args]


def conv2d_padding_same_op(X: op.Op, W: op.Op):
    kh, kw = W._v.shape[0], W._v.shape[1]
    pad_width = [[0, 0], [kh // 2, 1 - kh // 2], [kw // 2, 1 - kw // 2], [0, 0]]
    xpad: op.Op = PadConst(X, pad_width=pad_width, constant_values=0.)
    return Conv2D(xpad, W)


def max_pool2d_padding_same_op(X: op.Op, ksize: Union[tuple, list], stride: Union[tuple, list]):
    kh, kw = ksize
    pad_width = [[0, 0], [kh // 2, 1 - kh // 2], [kw // 2, 1 - kw // 2], [0, 0]]
    xpad: op.Op = PadConst(X, pad_width=pad_width, constant_values=0.)
    return MaxPool2D(xpad, ksize, stride)


def avg_pool2d_padding_same_op(X: op.Op, ksize: Union[tuple, list], stride: Union[tuple, list]):
    kh, kw = ksize
    pad_width = [[0, 0], [kh // 2, 1 - kh // 2], [kw // 2, 1 - kw // 2], [0, 0]]
    xpad: op.Op = PadConst(X, pad_width=pad_width, constant_values=0.)
    return AvgPool2D(xpad, ksize, stride)
