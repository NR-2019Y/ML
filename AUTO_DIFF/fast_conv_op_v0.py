import numbers
import numpy as np
from typing import Union, Tuple, List, Optional
from . import my_auto_grad_v0 as op
from .layers import Layer


class Conv2DLayer(Layer):
    def __init__(self, *, out_channels: int,
                 ksize: Union[int, Tuple[int, int]],
                 batch_input_size: Tuple[None, int, int, int],
                 strides: Union[int, Tuple[int, ...], List[int]] = 1,
                 padding: int = 0, bias: bool = True):
        if isinstance(ksize, int):
            ksize = (ksize,) * 2
        if isinstance(strides, int):
            strides = (strides,) * 2
        assert len(ksize) == 2 and len(strides) == 2
        _, ih, iw, ic = batch_input_size
        kh, kw = ksize
        sh, sw = strides
        oh = (ih - kh) // sh + 1
        ow = (iw - kw) // sw + 1
        oshape = (oh, ow, kh, kw)
        L = [np.tile(
            np.tile(np.arange(oshape[i]).reshape(-1, 1), np.prod(oshape[i + 1:], dtype=np.int32)).ravel(),
            np.prod(oshape[:i], dtype=np.int32)
        ).reshape(oshape) for i in range(4)]
        self._strides = strides
        self._oshape = oshape
        self._im2col_slice = (slice(None), L[0] * sh + L[2], L[1] * sw + L[3], slice(None))
        self._padding = padding
        W = op.C(np.random.normal(loc=0., scale=0.01, size=(kh, kw, ic, out_channels)), requires_grad=True)
        self._bias = bias
        if bias:
            b = op.C(np.zeros(out_channels), requires_grad=True)
            self.params = (W, b)
        else:
            self.params = W,

    def __call__(self, X: op.Op):
        W = self.params[0]
        L = _Conv2D(X, W, self._oshape, self._im2col_slice, self._strides, self._padding)
        if self._bias:
            b = self.params[-1]
            return op.AddBiasND(L, b)
        else:
            return L


# github.com/sebgao/cTensor
class _Conv2D(op.Op):
    def __init__(self, node1: op.Op, node2: op.Op,
                 oshape: Tuple[int, ...],
                 im2col_slice,
                 strides: Tuple[int, int] = (1, 1),
                 padding: int = 0):
        X: np.ndarray = node1._v
        W: np.ndarray = node2._v
        self._padding = padding
        if padding != 0:
            X = np.pad(X, pad_width=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            self._shape_after_pad = X.shape
        sh, sw = strides
        out_im2col_shape = X.shape[:1] + oshape + X.shape[-1:]
        out_im2col_strides = (X.strides[0], X.strides[1] * sh, X.strides[2] * sw,
                              X.strides[1], X.strides[2], X.strides[3])
        Xcol = np.lib.stride_tricks.as_strided(X, out_im2col_shape, out_im2col_strides, writeable=False)
        # Xcol = X[im2col_slice] 也能够实现im2col
        self._im2col_slice = im2col_slice
        self._Xcol = Xcol
        self._v = np.tensordot(Xcol, W, axes=[[-3, -2, -1], [0, 1, 2]])
        self._d = 0.
        self.nodes = (node1, node2)

    def back_calc_grad(self):
        node1, node2 = self.nodes
        W: np.ndarray = node2._v
        if hasattr(node1, "_d"):
            dXcol = np.tensordot(self._d, W, axes=[-1, -1])
            if self._padding == 0:
                if isinstance(node1._d, numbers.Number):
                    node1._d = np.zeros_like(node1._v)
                np.add.at(node1._d, self._im2col_slice, dXcol)
            else:
                p = self._padding
                d = np.zeros(shape=self._shape_after_pad)
                np.add.at(d, self._im2col_slice, dXcol)
                node1._d += d[:, p:-p, p:-p, :]
        if hasattr(node2, "_d"):
            node2._d += np.tensordot(self._Xcol, self._d, axes=[[0, 1, 2], [0, 1, 2]])


class MaxPool2DLayer(Layer):
    def __init__(self, *,
                 ksize: Union[int, Tuple[int, int]],
                 batch_input_size: Tuple[None, int, int, int],
                 strides: Union[int, Tuple[int, ...], List[int]] = 1,
                 padding: int = 0):
        if isinstance(ksize, int):
            ksize = (ksize,) * 2
        if isinstance(strides, int):
            strides = (strides,) * 2
        assert len(ksize) == 2 and len(strides) == 2
        _, ih, iw, ic = batch_input_size
        kh, kw = ksize
        sh, sw = strides
        oh = (ih - kh) // sh + 1
        ow = (iw - kw) // sw + 1
        oshape = (oh, ow, kh, kw)
        L = [np.tile(
            np.tile(np.arange(oshape[i]).reshape(-1, 1), np.prod(oshape[i + 1:], dtype=np.int32)).ravel(),
            np.prod(oshape[:i], dtype=np.int32)
        ).reshape(oshape) for i in range(4)]
        self._strides = strides
        self._oshape = oshape
        self._im2col_slice = (slice(None), L[0] * sh + L[2], L[1] * sw + L[3], slice(None))
        self._padding = padding

    def __call__(self, X: op.Op):
        return _MaxPool2D(X, self._oshape, self._im2col_slice, self._strides, self._padding)


class _MaxPool2D(op.Op):
    def __init__(self, node: op.Op,
                 oshape: Tuple[int, ...],
                 im2col_slice,
                 strides: Tuple[int, int] = (1, 1),
                 padding: int = 0):
        X: np.ndarray = node._v
        self._padding = padding
        if padding != 0:
            X = np.pad(X, pad_width=[[0, 0], [padding, padding], [padding, padding], [0, 0]])
            self._shape_after_pad = X.shape
        sh, sw = strides
        out_im2col_shape = X.shape[:1] + oshape + X.shape[-1:]
        out_im2col_strides = (X.strides[0], X.strides[1] * sh, X.strides[2] * sw,
                              X.strides[1], X.strides[2], X.strides[3])
        Xcol = np.lib.stride_tricks.as_strided(X, out_im2col_shape, out_im2col_strides, writeable=False)
        # Xcol = X[im2col_slice] 也能够实现im2col
        self._im2col_slice = im2col_slice
        self._Xcol_shape = Xcol.shape
        Xcol_bindkhkw = Xcol.reshape((Xcol.shape[0], Xcol.shape[1], Xcol.shape[2], -1, Xcol.shape[-1]))
        self._Xcol_bindkhkw_shape = Xcol_bindkhkw.shape
        self._argmax_index = np.expand_dims(Xcol_bindkhkw.argmax(-2), axis=-2)
        self._v = np.take_along_axis(Xcol_bindkhkw, self._argmax_index, axis=-2).squeeze(axis=-2)
        self._d = 0.
        self.nodes = (node,)

    def back_calc_grad(self):
        node, = self.nodes
        if hasattr(node, "_d"):
            dXcol_bindkhkw = np.zeros(shape=self._Xcol_bindkhkw_shape)
            np.put_along_axis(dXcol_bindkhkw, self._argmax_index, np.expand_dims(self._d, axis=-2), axis=-2)
            if self._padding == 0:
                if isinstance(node._d, numbers.Number):
                    node._d = np.zeros_like(node._v)
                np.add.at(node._d, self._im2col_slice, dXcol_bindkhkw.reshape(self._Xcol_shape))
            else:
                p = self._padding
                d = np.zeros(shape=self._shape_after_pad)
                np.add.at(d, self._im2col_slice, dXcol_bindkhkw.reshape(self._Xcol_shape))
                node._d += d[:, p:-p, p:-p, :]
