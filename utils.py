import numpy as np
from pathlib import Path
from scipy.linalg import block_diag


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


def metr_tens(W, X_n, lamb, lambda_W):
    d, n = X_n.shape
    K = W.shape[0]
    inter = np.einsum("ij, jk-> ikj", X_n, X_n.T, order="K").reshape(
        (d, d * n), order="F"
    )
    sig_der_square = np.power(sigma(W @ X_n), 2)
    factors = np.kron(sig_der_square, np.ones((1, d)))
    F_WW = block_diag(
        *[
            np.sum((factors[i, :] * inter).reshape((d, d, n), order="A"), axis=2)
            for i in range(K)
        ]
    ) + lambda_W * np.eye(K * d)
    return np.block(
        [
            [F_WW, np.zeros((K * d, K * n))],
            [np.zeros((K * n, K * d)), lamb * np.eye(K * n)],
        ]
    )


def eff_rad(D, Var_ups, F_inv, n):
    print("D:" + str(D.shape))
    print("F_inv:" + str(F_inv.shape))
    print("Var_ups:" + str(Var_ups.shape))
    B = D @ F_inv @ Var_ups @ F_inv @ D
    return np.sqrt(
        np.linalg.trace(B)
        + 2 * np.sqrt(np.log(n) * np.linalg.trace(B @ B))
        + 2 * np.log(n) * np.linalg.norm(B, 2)
    )


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
