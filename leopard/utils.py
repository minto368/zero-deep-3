import os
import subprocess
import urllib.request
import numpy as np
from leopard import as_variable
from leopard import Variable
from leopard import cuda

# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================
def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for leopard.functions.sum's backward.

    Args:
        gy (leopard.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        leopard.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    # xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    """Test backward procedure of a given function.

    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print('')
        print('========== FAILED (Gradient Check) ==========')
        print('Numerical Grad')
        print(' shape: {}'.format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
        print('Backprop Grad')
        print(' shape: {}'.format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
    return res


def numerical_grad(f, x, *args, **kwargs):
    """Computes numerical gradient by finite differences.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A target `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.

    Returns:
        `ndarray`: Gradient.
    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    xp = cuda.get_array_module(x)
    if xp is not np:
        np_x = cuda.as_numpy(x)
    else:
        np_x = x
    grad = xp.zeros_like(x)

    it = np.nditer(np_x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1 = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2 = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad


def array_equal(a, b):
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays
            to compare

    Returns:
        bool: True if the two arrays are equal.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.array_equal(a, b)


def array_allclose(a, b, rtol=1e-4, atol=1e-5):
    """Returns True if two arrays(or variables) are element-wise equal within a
    tolerance.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays
            to compare
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.

    Returns:
        bool: True if the two arrays are equal within the given tolerance,
            False otherwise.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


# =============================================================================
# download function
# =============================================================================
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end="")


cache_dir = os.path.join(os.path.expanduser("~"), ".leopard")


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind("/") + 1 :]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
