import numpy as np
from pathlib import Path


def positive_sigma(x, alpha):
    """
    Computes the SoftPlus function for non-negative input values.

    Parameters:
    x : array-like or float
        Input value(s) where x >= 0.
    alpha : float
        Smoothing parameter.

    Returns:
    float or ndarray
        The transformed input using the SoftPlus function.
    """
    return 1 / alpha * (alpha * x + np.log1p(np.exp(-alpha * x)))


def negative_sigma(x, alpha):
    """
    Computes the SoftPlus function for negative input values.

    Parameters:
    x : array-like or float
        Input value(s) where x < 0.
    alpha : float
        Smoothing parameter.

    Returns:
    float or ndarray
        The transformed input using the SoftPlus function.
    """
    return 1 / alpha * np.log1p(np.exp(alpha * x))


def sigma(x, alpha=1):
    """
    Computes the SoftPlus function.

    Parameters:
    x : array-like or float
        Input value(s).
    alpha : float, optional (default=1)
        Smoothing parameter.

    Returns:
    float or ndarray
        The transformed input using the SoftPlus function.
    """
    positive = x >= 0
    negative = ~positive

    res = np.empty_like(x, dtype=float)
    res[positive] = positive_sigma(x[positive], alpha)
    res[negative] = negative_sigma(x[negative], alpha)
    return res


def _positive_sigma_der(x, alpha):
    """
    Computes the derivative of the SoftPlus function for non-negative inputs.

    Parameters:
    x : array-like or float
        Input value(s) where x >= 0.
    alpha : float
        Smoothing parameter.

    Returns:
    float or ndarray
        The derivative of sigma at x.
    """
    return 1 / (1 + np.exp(-alpha * x))


def _negative_sigma_der(x, alpha):
    """
    Computes the derivative of the SoftPlus function for negative inputs.

    Parameters:
    x : array-like or float
        Input value(s) where x < 0.
    alpha : float
        Smoothing parameter.

    Returns:
    float or ndarray
        The derivative of sigma at x.
    """
    exp = np.exp(alpha * x)
    return exp / (exp + 1)


def sigma_der(x, alpha=1):
    """
    Computes the derivative of the SoftPlus function .

    Parameters:
    x : array-like or float
        Input value(s).
    alpha : float, optional (default=1)
        Smoothing parameter.

    Returns:
    ndarray
        The derivative of sigma at each x.
    """
    positive = x >= 0
    negative = ~positive

    res = np.empty_like(x, dtype=float)
    res[positive] = _positive_sigma_der(x[positive], alpha)
    res[negative] = _negative_sigma_der(x[negative], alpha)
    return res


def positive_sigma_sec_der(x, alpha=1):
    return alpha * sigma_der(x, alpha) * np.exp(-alpha * x) / (1 + np.exp(-alpha * x))


def negative_sigma_sec_der(x, alpha=1):
    return alpha * sigma_der(x, alpha) / (1 + np.exp(alpha * x))


def sigma_sec_der(x, alpha=1):
    """
    Computes the second derivative of the SoftPlus function .

    Parameters:
    x : array-like or float
        Input value(s).
    alpha : float, optional (default=1)
        Smoothing parameter.

    Returns:
    ndarray
        The second derivative of sigma at each x.
    """
    positive = x >= 0
    negative = ~positive

    res = np.empty_like(x, dtype=float)
    res[positive] = positive_sigma_sec_der(x[positive], alpha)
    res[negative] = negative_sigma_sec_der(x[negative], alpha)
    return res


def sigma_inv_pos(x, alpha=1):
    """
    Computes the inverse of the SoftPlus function for positive inputs.

    Parameters:
    x : array-like or float
        Input value(s) where x >= 0.

    Returns:
    float or ndarray
        The inverse of the SoftPlus function at x.
    """
    return x + np.log(1 - np.exp(-alpha * x)) / alpha


def sigma_inv_neg(x, alpha=1):
    """
    Computes the inverse of the SoftPlus function for negative inputs.

    Parameters:
    x : array-like or float
        Input value(s) where x < 0..

    Returns:
    float or ndarray
        The inverse of the SoftPlus function at x.
    """
    return np.log(np.expm1(alpha * x)) / alpha


def sigma_inv(x, alpha=1):
    """
    Computes the inverse of the SoftPlus function.

    Parameters:
    x : array-like or float
        Input value(s).

    Returns:
    float or ndarray
        The inverse of the SoftPlus function at x.
    """
    positive = x >= 0
    negative = ~positive

    res = np.empty_like(x, dtype=float)
    res[positive] = sigma_inv_pos(x[positive], alpha)
    res[negative] = sigma_inv_neg(x[negative], alpha)
    return res


def stable_softmax(x):
    """
    Compute the softmax function in a numerically stable way.

    Parameters:
    x (numpy.ndarray): Input array of any shape.

    Returns:
    numpy.ndarray: Softmax-transformed array with the same shape as x.
    """
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0)


def create_folders():
    """
    Creates two main directories: 'data' and 'imgs' in the current working directory.
    Each main directory will contain two subdirectories: 'mc' and 'conv'.

    If the directories already exist, no changes are made.
    """
    current_folder = Path.cwd()
    main_folders = ["data", "imgs"]
    subfolders = ["mc", "conv"]

    for main in main_folders:
        main_path = current_folder / main
        main_path.mkdir(exist_ok=True)

        for sub in subfolders:
            subfolder_path = main_path / sub
            subfolder_path.mkdir(exist_ok=True)

    print("Folders created successfully.")
