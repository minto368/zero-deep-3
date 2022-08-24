import numpy as np
import leopard
from leopard import utils
from leopard.core import Function, Variable, as_variable, as_array

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.sin(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.cos(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.log(x)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape  # xの形状を覚えておく
        y = x.reshape(self.shape)  # Numpyのreshape関数を使う
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x: np.ndarray, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x: np.ndarray):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x: np.ndarray, axes=None):
    return Transpose(axes)(x)


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape  # xの形状を覚えておく
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        # xp = leopard.cuda.get_array_module(x)
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
