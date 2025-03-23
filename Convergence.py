import json
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.linalg import block_diag
from utils import sigma, sigma_der, sigma_inv, sigma_sec_der
from scipy.optimize import minimize


class Convergence:
    """
    This class implements methods for sampling synthetic categorical data, optimizing model parameters, and evaluating
    performance based on likelihood, loss, and prediction error.

    Attributes:
    -----------
    lamb : float
        Regularization coefficient for the structural penalty.
    lamb_W : float
        Regularization coefficient for the weight matrix W.
    lambda_X: float
        Regularization coefficient for the nuisance parameter X.
    n : int
        Number of samples.
    d : int
        Number of features.
    K : int
        Number of categories.
    s_eta : float
        Probability assigned to a high-probability category.
    rng : numpy.random.Generator
        Random number generator instance for reproducibility.
    exp_Y : np.ndarray
        Expected values of the response.
    Y : np.ndarray
        Observed response.
    W : np.ndarray
        True weight matrix of shape (K, d).
    norm_constants : np.ndarray
        Normalization constants used to sample features.
    X_n : np.ndarray
        Feature matrix.
    W_tilde : np.ndarray
        Estimated weight matrix during optimization.
    X_tilde : np.ndarray
        Estimated nuisance parameter during optimization.
    pen_logLHs : np.ndarray
        Penalized log-likelihood values over iterations.
    logLH_Ws : np.ndarray
        Log-likelihood values for W over iterations.
    iterations : int
        Number of optimization iterations performed.
    losses : numpy.ndarray
        Stores loss values across iterations.
    losses_X : numpy.ndarray
        Stores loss values specific to X across iterations.
    hist : list
        Intermediate values of full dimensional parameter over iterations.
    W_s : numpy.ndarray
        Intermediate values of W over iterations.
    X_s : numpy.ndarray
        Intermediate values of X over iterations.
    cross_entr_s : numpy.ndarray
        Intermediate values of cross entropies specific to W.
    cross_entr_s_X : numpy.ndarray
        Intermediate values of cross entropies specific to X.

    Methods:
    --------
    sample_norm_const():
        Samples normalization constants from a uniform distribution.

    sample_exp_Y():
        Generates categorical probabilities with an emphasis on a high-probability category.

    sample_Y():
        Samples categorical labels based on the generated probabilities.

    sample_W():
        Initializes the weight matrix from a normal distribution.

    sample_features():
        Computes feature matrix X_n using inverse transformations.

    stand_log_lh_W(W):
        Computes the standardized log-likelihood for a given weight matrix.

    stand_log_lh(W, X):
        Computes the full dimensional standardized log-likelihood.

    stand_pen_log_lh(W, X):
        Computes the full dimensional standardized penalized log-likelihood .

    pred_err(W):
        Computes prediction error using W.

    pred_err_X(X):
        Computes prediction error given the nuisance parameter X.

    cross_entr(W):
        Computes the Frobenius norm difference between true and estimated W.

    cross_entr_X(X):
        Computes the Frobenius norm difference between X and its estimated form.

    plot_convergence():
        Plots the convergence of the penalized log-likelihood over iterations.

    gradient_X(W):
        Computes the gradient of the loss function with respect to X.

    gradient_W(W, X):
        Computes the gradient of the loss function with respect to W.

    maximize_newton(max_it=1000, eps=1e-3):
        Optimizes W and X using the Newton-CG method.

    maximize_cg(max_it=1000, eps=1e-3):
        Optimizes W and X using the Conjugate Gradient method.

    fisher(x):
        Computes the Fisher information matrix for second-order optimization.

    var_Y(X):
        Computes the covariance matrix of multinomially distributed data.

    """

    def __init__(self, n, d, K, s_eta, rng, lamb, lambda_W, lambda_X):
        """
        Initializes the class with given parameters and preallocates necessary variables.

        """
        self.rng = rng
        self.n = n
        self.d = d
        self.K = K
        self.s_eta = s_eta
        self.lamb = lamb
        self.lambda_W = lambda_W
        self.lambda_X = lambda_X
        self.exp_Y = self.sample_exp_Y()
        self.Y = self.sample_Y()
        self.W = self.sample_W()
        self.norm_constants = self.sample_norm_const()
        self.X_n = self.sample_features()
        self.W_tilde = np.zeros_like(self.W)
        self.X_tilde = np.zeros_like(self.Y)
        self.iterations = 0
        self.losses = np.zeros(0)
        self.losses_X = np.zeros(self.iterations)
        self.hist = []
        self.W_s = np.zeros(0)
        self.X_s = np.zeros(0)
        self.pen_log_LHs = np.zeros(0)
        self.log_LHs = np.zeros(0)
        self.log_LH_Ws = np.zeros(0)
        self.cross_entr_s = np.zeros(0)
        self.cross_entr_s_X = np.zeros(0)

    def sample_norm_const(self):
        """
        Samples the normalization constant `C` from a uniform distribution.

        Returns:
        --------
        float
            The sampled normalization constant `C`.
        """

        return self.rng.uniform(np.log(self.K), np.log(1e1 * self.K), size=(1, self.n))

    def sample_exp_Y(self):
        """
        Samples category probabilities from a Dirichlet distribution and adjusts them so that
        one randomly chosen category in each sample receives a higher probability (`s_eta`),
        while the remaining categories sum to `1 - s_eta`.

        Returns:
        --------
        np.ndarray
            A 2D array of shape (self.K, self.n), where each column contains
            the adjusted probability distributions over `K` categories.
        """
        # sample classes with high probability
        categories = self.rng.integers(0, self.K, self.n)

        # sample probability vectors
        probabilities = self.rng.dirichlet(1e2 * np.ones(self.K), size=self.n).T

        # Set probabilities of high probability classes to 0 and rescale s.t. sum of other probabilities is 1 - s_eta
        probabilities[categories, np.arange(self.n)] = 0
        probabilities = (1 - self.s_eta) * probabilities / np.sum(probabilities, axis=0)

        # Insert s_eta as probability for chosen high probability classes
        probabilities[categories, np.arange(self.n)] = self.s_eta

        return probabilities

    def sample_W(self):
        """
        Samples the weight matrix `W` from a normal distribution with variance 1/d.

        Returns:
        --------
        np.ndarray
            A 2D array of shape `(K, d)` representing the weight matrix `W` for the model.
            The weights are sampled from a normal distribution with variance 1/d.
        """

        # return self.rng.standard_normal((self.K, self.d)) / (np.sqrt(self.d * self.K)) * 1e-2
        return self.rng.standard_normal((self.K, self.d)) / np.sqrt(self.d)

    def sample_features(self):
        """
        Computes the feature matrix X_n.

        Returns:
        --------
        np.ndarray: The feature matrix X_n.
        """
        X_n = np.linalg.pinv(self.W) @ sigma_inv(
            self.norm_constants
            + np.log(self.exp_Y)
            - np.min(np.log(self.exp_Y), axis=0)
        )

        return X_n

    def sample_Y(self):
        """
        Samples output values `Y` based on the Multinomial distributions with parameters given by the probabilities array.

        Returns:
        --------
        numpy.ndarray
            A one-hot encoded matrix `Y` of shape (K, N), where K is the number of categories and N is the number of samples.
            Each column represents a sample, with a 1 in the position corresponding to the sampled category and 0 elsewhere.
        """
        return self.rng.multinomial(1, self.exp_Y.T).T

    def stand_log_lh_W(self, W):
        """
        Computes the standardized log-likelihood in the non-extended parameter

        Parameters:
        W (ndarray): Weight matrix.

        Returns:
        --------
        float: The computed log-likelihood.
        """
        return (
            np.sum(self.Y * sigma(W @ self.X_n))
            - np.sum(logsumexp(sigma(W @ self.X_n), axis=0))
        ) / self.n

    def stand_log_lh(self, W, X):
        """
        Computes the standardized log-likelihood of the full parameter.

        Parameters:
        W : ndarray
            Weight matrix.
        X : ndarray
            Current latent variable representation.

        Returns:
        --------
        float: The computed log-likelihood.
        """
        return (
            np.sum(self.Y * X)
            - np.sum(logsumexp(X, axis=0))
            - self.lamb / 2 * np.linalg.norm(X - sigma(W @ self.X_n), "fro") ** 2
        ) / self.n

    def stand_pen_log_lh(self, W, X):
        """
        Computes the standardized penalized log-likelihood of the full parameter.

        Parameters:
        W : numpy.ndarray
            Weight matrix.
        X : numpy.ndarray
            Nuisance parameter.

        Returns:
        --------
        float: The penalized log-likelihood.
        """

        return (
            self.stand_log_lh(W, X)
            - self.lambda_X / 2 * np.linalg.norm(X, "fro") ** 2 / self.n
            - self.lambda_W / 2 * np.linalg.norm(W, "fro") ** 2 / self.n
        )

    def cross_entr(self, W):
        return -np.sum(self.Y * np.log(softmax(sigma(W @ self.X_n), axis=0))) / self.n

    def cross_entr_X(self, X):
        return -np.sum(self.Y * np.log(softmax(X, axis=0))) / self.n

    def loss(self, W):
        return np.linalg.norm(self.W - W, "fro")

    def loss_X(self, X):
        return np.linalg.norm(X - sigma(self.W @ self.X_n), "fro")

    def gradient_X(self, W, X):
        """
        Computes the gradient of the loss function with respect to the nuisance parameter `X`

        Parameters:
        -----------

        W : numpy.ndarray
            The weight matrix. Shape: (K, d)

        X : numpy.ndarray
            The input matrix. Shape: (K, n)

        lamb : float
            The scaling factor for the structured loss term.

        lambda_X : float
            The regularization strength for the input matrix `X`.

        Returns:
        --------
        numpy.ndarray
            The gradient of the loss with respect to the input matrix `X`, including both the structured loss term and the L2 regularization term.
        """
        gradient_phi = self.Y - X * softmax(X, axis=0)
        gradient_struc = -self.lamb * (X - sigma(W @ self.X_n))
        gradient_ridge = -self.lambda_X * X

        return gradient_phi + gradient_struc + gradient_ridge

    def gradient_W(self, W, X):
        """
        Computes the gradient of the loss function with respect to the weight matrix `W`.

        Parameters:
        -----------
        W : numpy.ndarray
            The weight matrix for the model. Shape: (m, n), where m is the number of output units and n is the number of input features.

        X : numpy.ndarray
            The feature matrix. Shape: (m, N), where m is the number of output units and N is the number of data samples.

        Returns:
        --------
        numpy.ndarray
            The gradient of the loss with respect to the weight matrix `W`. This includes both the structured loss term and the L2 regularization term.
        """
        # gradient of structural penalty
        gradient_struc = self.lamb * np.sum(
            np.einsum(
                "ki, ji->ikj",
                (X - sigma(W @ self.X_n)) * sigma_der(W @ self.X_n),
                self.X_n,
                order="C",
            ),
            axis=0,
        )  # sum over i
        # gradient of ridge penalty
        gradient_ridge = -self.lambda_W * W

        return gradient_struc + gradient_ridge

    def callback(self, ups):
        """
        Stores the current optimization state and prints the current status.

        Parameters:
        -----------
        ups : numpy.ndarray
            The vectorized full dimensional parameter.

        Returns:
        --------
        None
            Updates the history (`self.hist`) and prints the current status.
        """
        self.hist.append(ups)
        self.print_status(ups)

    def sample_W_0(self):
        """
        Samples an initial weight matrix W_0 with small perturbations from true W.

        Returns:
        --------
        numpy.ndarray
            A perturbed version of the weight matrix W, sampled from a normal distribution.
        """

        return (
            self.W
            + self.rng.standard_normal((self.K, self.d))
            / np.sqrt(self.d * self.K)
            * 1e1
        )

    def extract_W(self, ups):
        """
        Extracts and reshapes the weight matrix W from vectorized full dimensional parameter.

        Parameters:
        -----------
        ups : numpy.ndarray
            Vectorized full dimensional parameter.

        Returns:
        --------
        numpy.ndarray
            Weight matrix W of shape (K, d).
        """
        return ups[: self.K * self.d].reshape(self.W.shape)

    def extract_X(self, ups):
        """
        Extracts and reshapes the latent variable matrix X from a flattened parameter vector.

        Parameters:
        -----------
        ups : numpy.ndarray
            Vectorized full dimensional parameter.

        Returns:
        --------
        numpy.ndarray
            Nuisance parameter X of shape (K, n).
        """
        return ups[self.K * self.d :].reshape(self.Y.shape)

    def gradient_ups(self, ups):
        """
        Computes the gradient of the penalized log-likelihood at ups.

        Parameters:
        -----------
        ups : numpy.ndarray
            Vectorized full dimensional parameter.

        Returns:
        --------
        numpy.ndarray
            Vectorized gradient of the penalized log-likelihood at ups.
        """
        W = self.extract_W(ups)
        X = self.extract_X(ups)
        return self.vectorize(self.gradient_W(W, X), self.gradient_X(W, X))

    def vectorize(self, W, X):
        """
        Flattens and concatenates W and X into a single vector.

        Parameters:
        -----------
        W : numpy.ndarray
            Weight matrix of shape (K, d).
        X : numpy.ndarray
            Nuisance parameter of shape (K, n).

        Returns:
        --------
        numpy.ndarray
            Flattened full dimensional parameter.
        """
        return np.concatenate((W.flatten(), X.flatten()))

    def print_status(self, ups):
        """
        Prints the current status of the optimization, including loss and prediction errors.

        Parameters:
        -----------
        ups : numpy.ndarray
            Flattened full dimensional parameter.

        Returns:
        --------
        None
            Outputs the current loss, prediction errors, and log-likelihood values.
        """
        W = self.extract_W(ups)
        X = self.extract_X(ups)
        print(f"pred_err: {self.cross_entr(W)}")
        print(f"pred_err_X: {self.cross_entr_X(X)}")
        print(f"loss: {self.loss(W)}")
        print(f"loss_X: {self.loss_X(X)}")
        print(f"-f(x) = {-self.stand_pen_log_lh(W, X)}")
        print(f"-logLH_W = {-self.stand_log_lh_W(W)}")

    def set_value_hist(self, iterations):
        """
        Stores the optimization history, including loss, log-likelihood, and predictions.

        Parameters:
        -----------
        iterations : int
            The number of iterations performed during optimization.

        Returns:
        --------
        None
            Updates the history of W, X, loss values, and log-likelihoods.
        """
        self.iterations = iterations
        self.W_s = np.zeros((self.iterations, self.K, self.d))
        self.X_s = np.zeros((self.iterations, self.K, self.n))
        self.losses = np.zeros(self.iterations)
        self.losses_X = np.zeros(self.iterations)
        self.cross_entr_s = np.zeros(self.iterations)
        self.cross_entr_s_X = np.zeros(self.iterations)
        self.pen_log_LHs = np.zeros(self.iterations)
        self.log_LHs = np.zeros(self.iterations)
        self.log_LH_Ws = np.zeros(self.iterations)

        for i in range(self.iterations):
            W_tilde = self.hist[i][: self.K * self.d].reshape((self.K, self.d))
            X_tilde = self.hist[i][self.K * self.d :].reshape((self.K, self.n))
            self.W_s[i, :, :] = W_tilde
            self.X_s[i, :, :] = X_tilde
            self.losses[i] = self.loss(W_tilde)
            self.losses_X[i] = self.loss_X(X_tilde)
            self.cross_entr_s[i] = self.cross_entr(W_tilde)
            self.cross_entr_s_X[i] = self.cross_entr_X(X_tilde)
            self.pen_log_LHs[i] = self.stand_pen_log_lh(W_tilde, X_tilde)
            self.log_LHs[i] = self.stand_log_lh(W_tilde, X_tilde)
            self.log_LH_Ws[i] = self.stand_log_lh_W(W_tilde)

        self.W_tilde = self.W_s[-1, :, :]
        self.X_tilde = self.X_s[-1, :, :]

    def maximize_newton(self, max_it=1000, eps=1e-3):
        """
        Maximizes the penalized log-likelihood function using the Newton-CG optimization method.

        Returns:
        --------
        None
            The function updates internal attributes.
        """
        W_0 = self.sample_W_0()
        X_0 = sigma(W_0 @ self.X_n)
        ups_0 = self.vectorize(W_0, X_0)
        self.print_status(ups_0)

        res = minimize(
            fun=lambda ups: -self.stand_pen_log_lh(
                self.extract_W(ups), self.extract_X(ups)
            ),
            x0=ups_0,
            method="Newton-CG",
            jac=lambda ups: -self.gradient_ups(ups),
            hess=lambda ups: self.fisher(ups),
            tol=eps,
            callback=lambda x: self.callback(x),
            options={"maxiter": max_it, "disp": True, "return_all": True},
        )

        self.set_value_hist(res.nit)

    def maximize_cg(self, max_it=10000, gtol=1e-3):
        """
        Performs constrained maximization of the penalized log-likelihood function
        using the Conjugate Gradient (CG) optimization method.

        Parameters:
        -----------
        max_it : int, optional (default=10000)
            The maximum number of iterations for the CG optimization.
        gtol : float, optional (default=1e-3)
            The gradient norm tolerance for stopping criteria.

        """

        W_0 = self.sample_W_0()
        X_0 = sigma(W_0 @ self.X_n)
        ups_0 = self.vectorize(W_0, X_0)
        self.print_status(ups_0)

        res = minimize(
            fun=lambda ups: -self.stand_pen_log_lh(
                self.extract_W(ups), self.extract_X(ups)
            ),
            x0=ups_0,
            method="CG",
            jac=lambda ups: -self.gradient_ups(ups),
            callback=lambda x: self.callback(x),
            options={
                "disp": True,
                "maxiter": max_it,
                "gtol": gtol,
                "norm": np.inf,
                "return_all": True,
            },
        )

        self.set_value_hist(res.nit)

    def fisher(self, ups):
        """
        Computes the Fisher Information Matrix (FIM) for the model @ ups = vec(W, X)

        Parameters:
        -----------
        ups : numpy.ndarray
            The full dimensional parameter as a flattened vector.

        Returns:
        --------
        numpy.ndarray
            The Fisher Information Matrix.
        """
        W = self.extract_W(ups)
        X = self.extract_X(ups)
        # matrix of outer products of feature vectors
        inter = np.einsum("ij, jk-> ikj", self.X_n, self.X_n.T, order="K").reshape(
            (self.d, self.d * self.n), order="F"
        )
        # matrix of squared derivatives of the activation function
        sig_der_square = np.power(sigma(W @ self.X_n), 2)

        # matrix of structural term
        non_struct = (
            -self.lamb * (X - sigma(W @ self.X_n)) * sigma_sec_der(W @ self.X_n)
        )
        # augment the matrix for multiplication with the outer products of feature vectors
        factors = np.kron(sig_der_square + non_struct, np.ones((1, self.d)))

        F_WW = self.lamb * block_diag(
            *[
                np.sum(
                    (factors[i, :] * inter).reshape(
                        (self.d, self.d, self.n), order="A"
                    ),
                    axis=2,
                )
                for i in range(self.K)
            ]
        ) + self.lambda_W * np.eye(self.K * self.d)
        F_XX = self.var_Y(X) + (self.lamb + self.lambda_X) * np.eye(self.K * self.n)
        v_blocks = [
            np.hstack(
                [np.outer(self.X_n[:, i], np.eye(self.K)[:, k]) for i in range(self.n)]
            )
            for k in range(self.K)
        ]
        X_augmented = np.vstack(v_blocks)
        sig_der = np.kron(sigma_der(W @ self.X_n), np.ones((self.d, self.K)))
        F_WX = sig_der * X_augmented
        F = np.block([[F_WW, F_WX], [F_WX.T, F_XX]])

        return F

    def var_Y(self, X):
        """
        Computes the covariance matrix of n multidimensional distributed
        random vectors whose expected values are proportional to the exponential
        of the columns of X.

        Parameters:
        -----------
        X : numpy.ndarray, shape (K, n)
            A matrix where each column represents a probability distribution
            (before applying softmax).

        Returns:
        --------
        numpy.ndarray, shape (Kn, Kn)
            The covariance matrix of the random vectors.
        """
        probabilities = softmax(X, axis=0)
        diag = np.diag(probabilities.flatten("F"))

        blocks = [
            np.outer(probabilities[:, i], probabilities[:, i]) for i in range(self.n)
        ]

        return diag - block_diag(*blocks)

    def save_results_as_dic(self, folder=""):
        result_dic = {
            "n": self.n,
            "d": self.d,
            "K": self.K,
            "s_eta": self.s_eta,
            "lamb": self.lamb,
            "lambda_W": self.lambda_W,
            "lambda_X": self.lambda_X,
            "exp_Y": self.exp_Y.tolist(),
            "Y": self.Y.tolist(),
            "W": self.W.tolist(),
            "X_n": self.X_n.tolist(),
            "W_tilde": self.W_tilde.tolist(),
            "X_tilde": self.X_tilde.tolist(),
            "iterations": self.iterations,
            "losses": self.losses.tolist(),
            "losses_X": self.losses_X.tolist(),
            "pen_log_LHs": self.pen_log_LHs.tolist(),
            "log_LHs": self.log_LHs.tolist(),
            "log_LH_Ws": self.log_LH_Ws.tolist(),
            "cross_entr_s": self.cross_entr_s.tolist(),
            "cross_entr_s_X": self.cross_entr_s_X.tolist(),
        }
        name = f"data/conv/{folder + '/' if folder else ""}n_{self.n}_d_{self.d}_K_{self.K}s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"

        with open(name, "w") as json_file:
            json.dump(result_dic, json_file, indent=4)
