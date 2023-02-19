import numbers
import numpy as np
from typing import Union, Tuple, List
from . import my_auto_grad_v0 as op


# github.com/sebgao/cTensor
class Conv2D(op.Op):
    def __init__(self, node1: op.Op, node2: op.Op, *,
                 strides: Union[int, Tuple[int, ...], List[int]] = 1):
        X: np.ndarray = node1._v
        W: np.ndarray = node2._v
        # X: N, ih, iw, ic
        # W: kh, kw, ic, oc
        # Xcol: N, oh, ow, kh, kw, ic
        # -> N, oh, ow, oc
        if isinstance(strides, numbers.Number):
            strides = (strides,) * 2
        assert len(strides) == 2
        ih, iw = X.shape[1], X.shape[2]
        kh, kw = W.shape[0], W.shape[1]
        sh, sw = strides
        oh = (ih - kh) // sh + 1
        ow = (iw - kw) // sw + 1
        oshape = (oh, ow, kh, kw)
        out_im2col_shape = (X.shape[0], oh, ow, kh, kw, X.shape[-1])
        out_im2col_strides = (X.strides[0], X.strides[1] * sh, X.strides[2] * sw,
                              X.strides[1], X.strides[2], X.strides[3])
        Xcol = np.lib.stride_tricks.as_strided(X, out_im2col_shape, out_im2col_strides, writeable=False)

        L = [
            np.tile(
                np.tile(np.arange(oshape[i]).reshape(-1, 1), np.prod(oshape[i + 1:], dtype=np.int32)).ravel(),
                np.prod(oshape[:i], dtype=np.int32)
            ).reshape(oshape) for i in range(4)
        ]
        self._im2col_slice = (slice(None), L[0] * sh + L[2], L[1] * sw + L[3], slice(None))
        self._Xcol = Xcol
        # Xcol = X[self._im2col_slice], 也能计算Xcol
        self._v = np.tensordot(Xcol, W, axes=[[-3, -2, -1], [0, 1, 2]])
        self._d = 0.
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        W: np.ndarray = node2._v
        if hasattr(node1, "_d"):
            if isinstance(node1._d, numbers.Number):
                node1._d = np.zeros_like(node1._v)
            dXcol = np.tensordot(self._d, W, axes=[-1, -1])
            np.add.at(node1._d, self._im2col_slice, dXcol)
        if hasattr(node2, "_d"):
            node2._d += np.tensordot(self._Xcol, self._d, axes=[[0, 1, 2], [0, 1, 2]])


class MaxPool2D(op.Op):
    def __init__(self, node: op.Op, *,
                 ksize: Union[int, Tuple[int, ...], List[int]] = 1,
                 strides: Union[int, Tuple[int, ...], List[int]] = 1):
        # X: N, ih, iw, ic
        # Xcol: N, oh, ow, kh, kw, ic
        # -> N, oh, ow, ic
        if isinstance(ksize, numbers.Number):
            ksize = (ksize,) * 2
        if isinstance(strides, numbers.Number):
            strides = (strides,) * 2
        assert len(ksize) == 2 and len(strides) == 2
        ih, iw = node.shape[1], node.shape[2]
        kh, kw = ksize
        sh, sw = strides
        oh = (ih - kh) // sh + 1
        ow = (iw - kw) // sw + 1
        oshape = (oh, ow, kh, kw)
        X: np.ndarray = node._v
        out_im2col_shape = (X.shape[0], oh, ow, kh, kw, X.shape[-1])
        out_im2col_strides = (X.strides[0], X.strides[1] * sh, X.strides[2] * sw,
                              X.strides[1], X.strides[2], X.strides[3])
        Xcol = np.lib.stride_tricks.as_strided(X, out_im2col_shape, out_im2col_strides, writeable=False)
        L = [
            np.tile(
                np.tile(np.arange(oshape[i]).reshape(-1, 1), np.prod(oshape[i + 1:], dtype=np.int32)).ravel(),
                np.prod(oshape[:i], dtype=np.int32)
            ).reshape(oshape) for i in range(4)
        ]
        self._Xcol_shape = Xcol.shape
        self._im2col_slice = (slice(None), L[0] * sh + L[2], L[1] * sw + L[3], slice(None))
        Xcol_bindkhkw = Xcol.reshape((X.shape[0], oh, ow, -1, X.shape[-1]))
        self._Xcol_bindkhkw_shape = Xcol_bindkhkw.shape
        self._argmax_index = np.expand_dims(Xcol_bindkhkw.argmax(-2), axis=-2)
        self._v = np.take_along_axis(Xcol_bindkhkw, self._argmax_index, axis=-2).squeeze(axis=-2)
        self._d = 0.
        self.nodes = (node,)

    def back_calc_grad(self):
        node, = self.nodes
        if hasattr(node, "_d"):
            if isinstance(node._d, numbers.Number):
                node._d = np.zeros_like(node._v)
            dXcol_bindkhkw = np.zeros(shape=self._Xcol_bindkhkw_shape)
            np.put_along_axis(dXcol_bindkhkw, self._argmax_index, np.expand_dims(self._d, axis=-2), axis=-2)
            np.add.at(node._d, self._im2col_slice, dXcol_bindkhkw.reshape(self._Xcol_shape))
