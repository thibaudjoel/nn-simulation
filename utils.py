import numpy as np
from pathlib import Path
from scipy.special import softmax
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
    positive = (x >= 0)
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
    return 1 / (1 + np.exp(- alpha * x))

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
    positive = (x >= 0)
    negative = ~positive

    res = np.empty_like(x, dtype=float)
    res[positive] = _positive_sigma_der(x[positive], alpha)
    res[negative] = _negative_sigma_der(x[negative], alpha)
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
    positive = (x >= 0)
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


def inv_Fisher(W, X_n, lamb, lambda_W, lambda_X):
    d, n = X_n.shape
    K = W.shape[0]
    # matrix of outer products of feature vectors
    inter = np.einsum('ij, jk-> ikj', X_n, X_n.T, order='K').reshape((d, d * n), order='F')
    # matrix of squared derivatives of the activation function
    sig_der_square = np.power(sigma(W @ X_n), 2)
    # augment the matrix for multiplication with the outer products of feature vectors
    factors = np.kron(sig_der_square, np.ones((1, d)))
    
    F_WW = block_diag(*[np.sum((factors[i, :] * inter).reshape((d, d, n), order='A'), axis=2) for i in range(K)]) + lambda_W * np.eye(K * d)
    F_XX = var_Y(W, X_n) + (lamb + lambda_X) * np.eye(K * n)
    v_blocks = [np.hstack([np.outer(X_n[:, i], np.eye(K)[:, k])  for i in range(n)]) for k in range(K)]
    X_augmented = np.vstack(v_blocks)
    sig_der = np.kron(sigma_der(W @ X_n), np.ones((d, K)))
    F_WX = sig_der * X_augmented
    F = np.block([[F_WW, F_WX],
                  [F_WX.T, F_XX]])
    return np.linalg.inv(F)

def metr_tens(W, X_n, lamb, lambda_W):
    d, n = X_n.shape
    K = W.shape[0]
    inter = np.einsum('ij, jk-> ikj', X_n, X_n.T, order='K').reshape((d, d * n), order='F')
    sig_der_square = np.power(sigma(W @ X_n), 2)
    factors = np.kron(sig_der_square, np.ones((1, d)))
    F_WW = block_diag(*[np.sum((factors[i, :] * inter).reshape((d, d, n), order='A'), axis=2) for i in range(K)]) + lambda_W * np.eye(K * d)
    return np.block([[F_WW, np.zeros((K * d, K * n))],
                     [np.zeros((K * n, K * d)), lamb * np.eye(K * n)]])

def var_Y(W, X_n):
    """
    Computes the variance of `Y`.

    Parameters:
    -----------
    W : numpy.ndarray
        The weight matrix. Shape: (m, n), where m is the number of output units and n is the number of input features.
    
    X_n : numpy.ndarray
        The input matrix. Shape: (n, N), where n is the number of input features and N is the number of data samples.

    Returns:
    --------
    numpy.ndarray
        A matrix representing the variance of the output `Y`. Shape: (m, m), where m is the number of output units.

    """
    eta = sigma(W @ X_n)
    probabilities = softmax(eta, axis=0)
    diag = np.diag(probabilities.flatten('F'))
    n = eta.shape[1]
    blocks = [np.outer(probabilities[:, i], probabilities[:, i]) for i in range(n)]
    
    return diag - block_diag(*blocks)


def eff_rad(D, Var_ups, F_inv, n):
    print("D:" + str(D.shape))
    print("F_inv:" + str(F_inv.shape))
    print("Var_ups:" + str(Var_ups.shape))
    B = D @ F_inv @ Var_ups @ F_inv @ D
    return np.sqrt(np.linalg.trace(B) + 2 * np.sqrt(np.log(n) * np.linalg.trace(B @ B)) + 2 * np.log(n) * np.linalg.norm(B, 2))

# def eff_dim():
#     return

# def tau_3(alpha):
#     """
#     Computes the tau_3 value based on two terms: one involving `alpha` and the other involving 
#     the noise level of the model's predictions. The final value returned is the maximum of these two terms.

#     Parameters:
#     -----------
#     alpha : float
#         A parameter that scales the first term of the computation. This should be a scalar.
    
#     W : numpy.ndarray
#         The weight matrix of the model. Shape: (K, d), where K is the number of categories and d is the number of input features.
    
#     X_n : numpy.ndarray
#         The input matrix. Shape: (d, N), where d is the number of features and N is the number of samples.

#     Returns:
#     --------
#     float
#         The tau_3 value, which is the maximum of two computed terms: one depending on `alpha`
#         and the other on the model's noise level.
#     """
#     one = np.sqrt(2) * np.exp(1) * alpha * np.max()
#     two = 2**(3/2) * np.exp(2) * noise_level(W, X_n) * (1 - noise_level(W, X_n))**(3/2)
#     return np.max([one, two])

def create_img_folders():
    """
    Creates a main directory named 'imgs' in the current working directory 
    and three subdirectories: 'good_initialization', 'bad_initialization', 
    and 'monte_carlo'. If the directories already exist, no changes are made.

    """
    current_folder = Path.cwd()
    main = current_folder / "imgs"
    subfolders = ["good_initialization", "bad_initialization", "monte_carlo"]

    # Create main folder
    main.mkdir(exist_ok=True)

    # Create subfolders
    for sub in subfolders:
        subfolder_path = main / sub
        subfolder_path.mkdir(exist_ok=True)
