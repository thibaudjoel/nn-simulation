import numpy as np
from scipy.special import softmax, logsumexp
from scipy.linalg import block_diag

def sigma(x, alpha=1):
    """
    Computes the smooth approximation of the ReLU (SoftPlus) function.

    Parameters:
    x : array-like or float
        Input value(s).
    alpha : float, optional (default=1)
        Smoothing parameter that controls the sharpness of the function.

    Returns:
    float or ndarray
        The transformed input using the SoftPlus-like function.
    """
    return 1 / alpha * np.log1p(np.exp(alpha * x))

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
    Computes the derivative of the SoftPlus function for all input values.

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

def log_lh(W, X, X_n, Y, lamb):
    """
    Computes the log-likelihood.

    Parameters:
    W (ndarray): Weight matrix.
    X (ndarray): Current latent variable representation.
    X_n (ndarray): Feature matrix.
    Y (ndarray): Observed data.
    lamb (float): Regularization coefficient for the structural penalty.

    Returns:
    float: The computed log-likelihood.
    """
    return np.sum(Y * X - logsumexp(X, axis=0)) - lamb / 2 * np.linalg.norm(X - sigma(W @ X_n), 'fro')**2

def pen_log_lh(W, X, X_n, Y, lamb, lamb_X, lamb_W):
    """
    Computes the penalized log-likelihood function with additional regularization terms.

    Parameters:
    W (ndarray): Weight matrix.
    X (ndarray): Current latent variable representation.
    X_n (ndarray): Feature matrix.
    Y (ndarray): Observed data.
    lamb (float): Regularization coefficient for structural penalty.
    lamb_X (float): Regularization coefficient for X.
    lamb_W (float): Regularization coefficient for W.

    Returns:
    float: The penalized log-likelihood.
    """
    return log_lh(W, X, X_n, Y, lamb) - lamb_W / 2 * np.linalg.norm(W, 'fro')**2 - lamb_X / 2 * np.linalg.norm(X, 'fro')**2

def gradient_W(W, X, X_n, lamb, lamb_W):
    """
    Computes the gradient of the loss function with respect to the weight matrix `W`.

    Parameters:
    -----------
    W : numpy.ndarray
        The weight matrix for the model. Shape: (m, n), where m is the number of output units and n is the number of input features.
    
    X : numpy.ndarray
        The feature matrix. Shape: (m, N), where m is the number of output units and N is the number of data samples.
    
    X_n : numpy.ndarray
        The input matrix for the regularization term. Shape: (n, N), where n is the number of input features and N is the number of data samples.
    
    lamb : float
        The scaling factor for the structured loss term. Controls the contribution of the structured loss in the final gradient.
    
    lamb_W : float
        The regularization strength for the weight matrix `W`. Controls the contribution of the L2 regularization term in the final gradient.

    Returns:
    --------
    numpy.ndarray
        The gradient of the loss with respect to the weight matrix `W`. This includes both the structured loss term and the L2 regularization term.
    """
    grad_str = lamb * np.sum(np.einsum('ki, ji->ikj',(X - sigma(W @ X_n)) * sigma_der(W @ X_n), X_n, order='C'), axis=0) # sum over i
    grad_pen = - lamb_W * W
    return grad_str + grad_pen
    
def gradient_X(Y, W, X, X_n, lamb, lamb_X):
    """
    Computes the gradient of the loss function with respect to the nuisance parameter `X`, including:
    - A term related to the model's output (`grad_phi`).
    - A structured loss term (`grad_str`).
    - A regularization term (`grad_pen`).

    Parameters:
    -----------
    Y : numpy.ndarray
        The target/output matrix. Shape: (m, N), where m is the number of output units and N is the number of data samples.
    
    W : numpy.ndarray
        The weight matrix. Shape: (m, n), where m is the number of output units and n is the number of input features.
    
    X : numpy.ndarray
        The input matrix. Shape: (n, N), where n is the number of input features and N is the number of data samples.
    
    X_n : numpy.ndarray
        The feature matrix. Shape: (n, N), where n is the number of input features and N is the number of data samples.
    
    lamb : float
        The scaling factor for the structured loss term. Controls the contribution of the structured loss in the final gradient.
    
    lamb_X : float
        The regularization strength for the input matrix `X`. Controls the contribution of the L2 regularization term in the final gradient.

    Returns:
    --------
    numpy.ndarray
        The gradient of the loss with respect to the input matrix `X`, including both the structured loss term and the L2 regularization term.
    """
    
    grad_phi = Y - X * np.exp(X) / np.sum(np.exp(X), axis=0) # sum over k
    grad_str = - lamb * (X - sigma(W @ X_n)) 
    grad_pen = - lamb_X * X
    return grad_phi + grad_str + grad_pen

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

def sample_features(W, n, K):
    beta = np.random.random((K, n))
    categories = np.random.randint(0, K, n) #classes with high probability
    beta[categories, range(n)] += 6
    X_n = np.linalg.pinv(W) @ beta
    
    return X_n

def sample_Y(W, X_n):
    """
    Samples output values `Y` based on the softmax probabilities of `eta`.

    Parameters:
    -----------
    W : numpy.ndarray
        The weight matrix. Shape: (K, d), where K is the number of classes and d is the number of input features.

    X_n : numpy.ndarray
        The input matrix. Shape: (d, n), where d is the number of input features and n is the number of samples.

    Returns:
    --------
    numpy.ndarray
        A one-hot encoded matrix `Y` of shape (K, n), where K is the number of categories and N is the number of samples.
        Each column represents a sample, with a 1 in the position corresponding to the sampled category and 0 elsewhere.
    """
    
    eta = sigma(W @ X_n)
    K = eta.shape[0]
    Y = np.zeros_like(eta)
    for i in range(eta.shape[1]):
        # sample a category from the probabilities given by the softmax of eta
        probabilities = softmax(eta[:, i])
        sampled_category = np.random.choice(K, 1, p=probabilities)[0]
        # set Y_i entry of sampled category to 1
        Y[sampled_category, i] = 1
    return Y

def grad_asc(Y, X_n, lamb, lamb_W, lamb_X, max_it, eps_W, eps_X, gamma, W_star=None, W=None):
    """
    Performs gradient ascent to find maximizers W and X of the log-likelihood.

    The function computes the gradients of the loss function with respect to `W` and `X`, and updates the matrices
    iteratively based on these gradients. The process includes both structured penalties and log-likelihood regularization
    terms to prevent overfitting. Convergence is determined by checking the changes in `W` and `X` against given thresholds.

    Parameters:
    -----------
    Y : numpy.ndarray
        The target/output matrix. Shape: (K, N), where K is the number of categories and N is the number of samples.
    
    X_n : numpy.ndarray
        The input matrix. Shape: (n, N), where n is the number of input features and N is the number of samples.
    
    lamb : float
        The scaling factor for the structured loss term.
    
    lamb_W : float
        The regularization strength for the weight matrix `W`.
    
    lamb_X : float
        The regularization strength for the input matrix `X`.
    
    max_it : int
        The maximum number of iterations for the optimization process.
    
    eps_W : float
        The convergence threshold for the weight matrix `W`. Optimization stops if the Frobenius norm of the difference
        between successive `W` updates is below this threshold.
    
    eps_X : float
        The convergence threshold for the input matrix `X`. Optimization stops if the Frobenius norm of the difference
        between successive `X` updates is below this threshold.
    
    gamma : float
        The learning rate or step size for updating `W` and `X`.

    Returns:
    --------
    W_cur : numpy.ndarray
        The optimized weight matrix after convergence. Shape: (K, d), where K is the number of categories and d is the number
        of input features.
    
    X_cur : numpy.ndarray
        The optimized input matrix after convergence. Shape: (K, n), where K is the number of categories and n is the number
        of samples.

    struct_pens : list
        A list of the structured penalty values for each iteration.

    pen_logLHs : list
        A list of the penalized log-likelihood values for each iteration.

    diff_Ws : list
        A list of the differences (Frobenius norm) between successive `W` matrices for each iteration.

    diff_Xs : list
        A list of the differences (Frobenius norm) between successive `X` matrices for each iteration.

    i : int
        The number of iterations performed before convergence or reaching the maximum number of iterations.

    """
    struct_pens = []
    pen_logLHs = []
    diff_Ws = []
    diff_Xs = []
    losses = []
    biases = []
    
    # initialization
    # W_cur = np.zeros((K, d))
    W_cur = W
    X_cur = sigma(W @ X_n)
    
    for i in range(max_it):
        W_new = W_cur + gamma * gradient_W(W_cur, X_cur, X_n, lamb, lamb_W)
        X_new = X_cur + gamma * gradient_X(Y, W_cur, X_cur, X_n, lamb, lamb_X)
        
        diff_W = np.linalg.norm(W_new - W_cur, 'fro') 
        diff_X = np.linalg.norm(X_new - X_cur, 'fro')
        struct_pen = np.linalg.norm(X_new - sigma(W_new @ X_n))
        pen_logLH = pen_log_lh(W_cur, X_cur, X_n, Y, lamb, lamb_X, lamb_W)
        
        struct_pens.append(struct_pen)
        pen_logLHs.append(pen_logLH)
        diff_Ws.append(diff_W)
        diff_Xs.append(diff_X)
        if W_star is not None:
            losses.append(np.linalg.norm(W_star - W_cur, 'fro')**2)
        if W is not None:
            biases.append(np.linalg.norm(W - W_cur, 'fro')**2)
        
        if (diff_W < eps_W) and (diff_X < eps_X):
            break
        
        W_cur = W_new
        X_cur = X_new
        
    W_cur = W_new
    X_cur = X_new
    
    struct_pens.append(np.linalg.norm(X_new - sigma(W_new @ X_n)))
    pen_logLHs.append(pen_log_lh(W_cur, X_cur, X_n, Y, lamb, lamb_X, lamb_W))
    
    if W_star is not None:
        losses.append(np.linalg.norm(W_star - W_cur, 'fro')**2)
        
    if W is not None:
        biases.append(np.linalg.norm(W - W_cur, 'fro')**2)
        
    return W_cur, X_cur, struct_pens, pen_logLHs, diff_Ws, diff_Xs, i, losses, biases

def noise_level(W, X_n):
    """
    Computes the noise level of the model's output probabilities.

    This function applies the sigmoid activation to the weighted input (`W @ X_n`),
    then computes the softmax probabilities for each sample. The noise level is defined
    as the minimum of the maximum probabilities for each sample. The lower the noise level,
    the more confident the model is about its predictions.

    Parameters:
    -----------
    W : numpy.ndarray
        The weight matrix of the model. Shape: (K, d), where K is the number of categories and d is the number of input features.
    
    X_n : numpy.ndarray
        The input matrix. Shape: (d, N), where d is the number of input features and N is the number of samples.
    
    sigma : function
        The sigmoid activation function applied to the result of `W @ X_n`. It should be an element-wise function, like `np.sigmoid`.

    softmax : function
        The softmax activation function applied to the output of the sigmoid function (`eta`). It should return a probability distribution for each sample.

    Returns:
    --------
    float
        The noise level of the model's predictions, defined as the minimum of the maximum probabilities across all samples.
        A higher noise level indicates less confidence in the predictions, while a lower noise level indicates higher confidence.
    """
    eta = sigma(W @ X_n)
    probabilities = softmax(eta, axis=0)
    return np.min(np.max(probabilities, axis=0))

def tau_3(alpha):
    """
    Computes the tau_3 value based on two terms: one involving `alpha` and the other involving 
    the noise level of the model's predictions. The final value returned is the maximum of these two terms.

    Parameters:
    -----------
    alpha : float
        A parameter that scales the first term of the computation. This should be a scalar.
    
    W : numpy.ndarray
        The weight matrix of the model. Shape: (K, d), where K is the number of categories and d is the number of input features.
    
    X_n : numpy.ndarray
        The input matrix. Shape: (d, N), where d is the number of features and N is the number of samples.

    Returns:
    --------
    float
        The tau_3 value, which is the maximum of two computed terms: one depending on `alpha`
        and the other on the model's noise level.
    """
    one = np.sqrt(2) * np.exp(1) * alpha * np.max()
    two = 2**(3/2) * np.exp(2) * noise_level(W, X_n) * (1 - noise_level(W, X_n))**(3/2)
    return np.max([one, two])
    