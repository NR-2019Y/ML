import numpy as np
from . import my_auto_grad_v0 as op
import numbers

eps = op.eps


class Layer:
    pass


class Dense(Layer):
    def __init__(self, units, *, input_dim, use_bias=True):
        W = op.C(np.random.normal(loc=0., scale=0.01, size=(input_dim, units)), requires_grad=True)
        if use_bias:
            b = op.C(np.zeros(units), requires_grad=True)
            self.params = (W, b)
        else:
            self.params = (W,)

    def __call__(self, X: op.Op):
        if len(self.params) == 1:
            W = self.params[0]
            return op.MatMul(X, W)
        else:
            W, b = self.params
            return op.LinearLayer(X, W, b)


class BasicRNN(Layer):
    def __init__(self, units, *, input_dim):
        W_xh = op.C(np.random.normal(scale=0.01, size=(input_dim, units)), requires_grad=True)
        W_hh = op.C(np.random.normal(scale=0.01, size=(units, units)), requires_grad=True)
        b_h = op.C(np.zeros(units), requires_grad=True)
        self._n_hiddens = units
        self.params = W_xh, W_hh, b_h

    def __call__(self, X: op.Op):
        W_xh, W_hh, b_h = self.params
        batch_size, n_steps, _ = X.shape
        H = op.C(np.zeros([batch_size, self._n_hiddens]))
        L = []
        for i in range(n_steps):
            xi = X[:, i, :]
            H = op.Tanh(op.AddBiasND(xi @ W_xh + H @ W_hh, b_h))  # (batch_size, n_niddens)
            L.append(H)
        outputs = op.Transpose(op.ListToTensor(L), [1, 0, 2])  # (batch_size, seq_len, n_niddens)
        return outputs


class GRU(Layer):
    @staticmethod
    def gen_rand_w_and_b(n_inputs, n_hiddens):
        return (
            op.C(np.random.normal(scale=0.01, size=(n_inputs, n_hiddens)), requires_grad=True),
            op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_hiddens)), requires_grad=True),
            op.C(np.zeros(shape=(n_hiddens,)), requires_grad=True)
        )

    def __init__(self, units, *, input_dim):
        W_xr, W_hr, b_r = GRU.gen_rand_w_and_b(input_dim, units)
        W_xz, W_hz, b_z = GRU.gen_rand_w_and_b(input_dim, units)
        W_xh, W_hh, b_h = GRU.gen_rand_w_and_b(input_dim, units)
        self._n_hiddens = units
        self.params = W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h

    def __call__(self, X: op.Op):
        W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h = self.params
        batch_size, n_steps, _ = X.shape
        H = op.C(np.zeros(shape=(batch_size, self._n_hiddens)))
        L = []
        for i in range(n_steps):
            xi = X[:, i, :]
            R = op.Sigmoid(op.AddBiasND(op.MatMul(xi, W_xr) + op.MatMul(H, W_hr), b_r))
            Z = op.Sigmoid(op.AddBiasND(op.MatMul(xi, W_xz) + op.MatMul(H, W_hz), b_z))
            H_d = op.Tanh(op.AddBiasND(op.MatMul(xi, W_xh) + op.MatMul(R * H, W_hh), b_h))
            H = Z * H + (op.C(1.0) - Z) * H_d
            L.append(H)
        outputs = op.Transpose(op.ListToTensor(L), [1, 0, 2])  # (batch_size, seq_len, n_niddens)
        return outputs


class LSTM(Layer):
    @staticmethod
    def gen_rand_w_and_b(n_inputs, n_hiddens):
        return (
            op.C(np.random.normal(scale=0.01, size=(n_inputs, n_hiddens)), requires_grad=True),
            op.C(np.random.normal(scale=0.01, size=(n_hiddens, n_hiddens)), requires_grad=True),
            op.C(np.zeros(shape=(n_hiddens,)), requires_grad=True)
        )

    def __init__(self, units, *, input_dim):
        W_xi, W_hi, b_i = LSTM.gen_rand_w_and_b(input_dim, units)
        W_xf, W_hf, b_f = LSTM.gen_rand_w_and_b(input_dim, units)
        W_xo, W_ho, b_o = LSTM.gen_rand_w_and_b(input_dim, units)
        W_xc, W_hc, b_c = LSTM.gen_rand_w_and_b(input_dim, units)
        self._n_hiddens = units
        self.params = W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c

    def __call__(self, X: op.Op):
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c = self.params
        batch_size, n_steps, _ = X.shape
        H = op.C(np.zeros(shape=(batch_size, self._n_hiddens)))
        C = op.C(np.zeros(shape=(batch_size, self._n_hiddens)))
        L = []
        for i in range(n_steps):
            xi = X[:, i, :]
            I = op.Sigmoid(op.AddBiasND(op.MatMul(xi, W_xi) + op.MatMul(H, W_hi), b_i))
            F = op.Sigmoid(op.AddBiasND(op.MatMul(xi, W_xf) + op.MatMul(H, W_hf), b_f))
            O = op.Sigmoid(op.AddBiasND(op.MatMul(xi, W_xo) + op.MatMul(H, W_ho), b_o))
            C_d = op.Tanh(op.AddBiasND(op.MatMul(xi, W_xc) + op.MatMul(H, W_hc), b_c))
            C = F * C + I * C_d
            H = O * op.Tanh(C)
            L.append(H)
        outputs = op.Transpose(op.ListToTensor(L), [1, 0, 2])  # (batch_size, seq_len, n_niddens)
        return outputs


class BatchNorm(Layer):
    def __init__(self, *, axis=-1, input_dim, ndim):
        # axis: 指定通道维
        # input_dim: 通道数
        gamma = op.C(np.ones(input_dim), requires_grad=True)
        beta = op.C(np.zeros(input_dim), requires_grad=True)
        moving_mean: np.ndarray = np.zeros(input_dim)
        moving_var: np.ndarray = np.zeros(input_dim)
        self.params = gamma, beta
        self._const_params = moving_mean, moving_var
        assert isinstance(axis, numbers.Integral)
        if axis < 0:
            axis = ndim - axis
        self._axis_to_reduce = tuple(i for i in range(ndim) if i != in axis)

    def __call__(self, X: op.Op):
        gamma, beta = self.params
        moving_mean, moving_var = self._const_params
        if not op.mode_is_training():
            x_scaled: np.ndarray = (X._v - moving_mean) / (np.sqrt(moving_var) + eps)
            return op.C(x_scaled * gamma + beta)
        x_scaled: op.Scaler = op.Scaler(X, axis=self._axis_to_reduce)
        mean_val, var_val = x_scaled._bn_mean, x_scaled._bn_var
        del x_scaled._bn_mean, x_scaled._bn_var
        Y = op.BroadcastAdd(op.BroadcastMul(gamma, x_scaled), beta)
        momentum = 0.9
        moving_mean = momentum * moving_mean + (1. - momentum) * mean_val
        moving_var = momentum * moving_var + (1. - momentum) * var_val
        self._const_params = moving_mean, moving_var
        return Y
