import numpy as np
from .my_auto_grad_v0 import get_reduced_result


def _calc_binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, threshold, reduction=None):
    eq = y_true == np.where(y_pred > threshold, 1, 0)
    return get_reduced_result(eq, reduction)


# 命名参考keras
def binary_accuracy_from_logits(y_true: np.ndarray, y_pred: np.ndarray, *, reduction=None):
    return _calc_binary_accuracy(y_true, y_pred, threshold=0., reduction=reduction)


def binary_accuracy_from_proba(y_true: np.ndarray, y_pred: np.ndarray, *, reduction=None):
    return _calc_binary_accuracy(y_true, y_pred, threshold=0.5, reduction=reduction)


def categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, reduction=None):
    eq = np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1)
    return get_reduced_result(eq, reduction)


def sparse_categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, reduction=None):
    eq = y_true == np.argmax(y_pred, axis=-1)
    return get_reduced_result(eq, reduction)
