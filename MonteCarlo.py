import numpy as np
import json
from utils import sigma_inv, sigma, sigma_der, sigma_sec_der
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize
from scipy.linalg import block_diag


class MonteCarlo:
    def __init__(
        self,
        trials,
        lamb,
        lambda_W,
        lambda_X,
        n,
        d,
        K,
        s_eta,
        rng,
        max_it=1000,
        eps=1e-7,
    ):
        self.rng = rng
        self.trials = trials
        self.lamb = lamb
        self.lambda_W = lambda_W
        self.lambda_X = lambda_X
        self.n = n
        self.d = d
        self.K = K
        self.s_eta = s_eta
        self.eps = eps
        self.max_it = max_it
        self.exp_Y = self.sample_exp_Y()
        self.Y = self.sample_Y()
        self.W = self.sample_W()
        self.norm_constants = self.sample_norm_const()
        self.X_n = self.sample_features()
        self.W_0 = self.sample_W_0()
        self.W_tilde_s = np.zeros((self.trials, self.K, self.d))
        self.X_tilde_s = np.zeros((self.trials, self.K, self.n))
        self.losses = np.zeros(trials)
        self.cross_entr_s = np.zeros(trials)
        self.pen_log_LHs = np.zeros(trials)
        self.log_LHs = np.zeros(trials)
        self.log_LH_Ws = np.zeros(trials)
        self.iterations = np.zeros(trials)

        # self.D = metr_tens(self.W, self.X_n, self.lamb, self.lambda_W)
        # self.Sigma = var_Y(self.W, self.X_n)
        # self.Var_ups = block_diag(np.zeros((K * d, K * d)), self.Sigma)
        # self.F_inv = inv_Fisher(self.W, self.X_n, self.lamb, self.lambda_W, self.lambda_X)
        # self.eff_rad = eff_rad(self.D, self.Var_ups, self.F_inv, self.n)
        # self.eff_dim

    def sample_norm_const(self):
        """
        Samples the normalization constants `C` from a uniform distribution.

        The values are drawn from a uniform distribution in the range
        [log(K), log(10 * K)]

        Returns:
        --------
        np.ndarray
            An array of sampled normalization constants `C`
            with shape (1, self.n).
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
        np.ndarray: Array of shape (self.d, self.n) containing the feature matricx.
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
        Y = np.zeros((self.trials, self.K, self.n))
        for trial in range(self.trials):
            Y[trial, :, :] = self.rng.multinomial(1, self.exp_Y.T).T
        return Y

    def stand_log_lh_W(self, W, trial):
        """
        Computes the standardized log-likelihood in the non-extended parameter

        Parameters:
        W (ndarray): Weight matrix.

        Returns:
        --------
        float: The computed log-likelihood.
        """
        return (
            np.sum(self.Y[trial, :, :] * sigma(W @ self.X_n))
            - np.sum(logsumexp(sigma(W @ self.X_n), axis=0))
        ) / self.n

    def stand_log_lh(self, W, X, trial):
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
            np.sum(self.Y[trial, :, :] * X)
            - np.sum(logsumexp(X, axis=0))
            - self.lamb / 2 * np.linalg.norm(X - sigma(W @ self.X_n), "fro") ** 2
        ) / self.n

    def stand_pen_log_lh(self, W, X, trial):
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
            self.stand_log_lh(W, X, trial)
            - self.lambda_X / 2 * np.linalg.norm(X, "fro") ** 2 / self.n
            - self.lambda_W / 2 * np.linalg.norm(W, "fro") ** 2 / self.n
        )

    def cross_entr(self, W, trial):
        return (
            -np.sum(self.Y[trial, :, :] * np.log(softmax(sigma(W @ self.X_n), axis=0)))
            / self.n
        )

    def loss(self, W):
        return np.linalg.norm(self.W - W, "fro")

    def gradient_X(self, W, X, trial):
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
        gradient_phi = self.Y[trial, :, :] - X * softmax(X, axis=0)
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
        return ups[self.K * self.d :].reshape((self.K, self.n))

    def gradient_ups(self, ups, trial):
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
        return self.vectorize(self.gradient_W(W, X), self.gradient_X(W, X, trial))

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

    def maximize_newton(self, trial):
        """
        Maximizes the penalized log-likelihood function using the Newton-CG optimization method.

        Returns:
        --------
        None
            The function updates internal attributes.
        """
        W_0 = self.W_0
        X_0 = sigma(W_0 @ self.X_n)
        ups_0 = self.vectorize(W_0, X_0)

        result = minimize(
            fun=lambda ups: -self.stand_pen_log_lh(
                self.extract_W(ups), self.extract_X(ups), trial
            ),
            x0=ups_0,
            method="Newton-CG",
            jac=lambda ups: -self.gradient_ups(ups, trial),
            hess=lambda ups: self.fisher(ups),
            tol=self.eps,
            options={"maxiter": self.max_it, "disp": True, "return_all": False},
        )

        return result

    def set_results(self, results):
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
        for trial in range(self.trials):
            W_tilde = results[trial].x[: self.K * self.d].reshape((self.K, self.d))
            X_tilde = results[trial].x[self.K * self.d :].reshape((self.K, self.n))
            self.iterations[trial] = results[trial].nit
            self.losses[trial] = self.loss(W_tilde)
            self.cross_entr_s[trial] = self.cross_entr(W_tilde, trial)
            self.pen_log_LHs[trial] = self.stand_pen_log_lh(W_tilde, X_tilde, trial)
            self.log_LHs[trial] = self.stand_log_lh(W_tilde, X_tilde, trial)
            self.log_LH_Ws[trial] = self.stand_log_lh_W(W_tilde, trial)

    # def maximize_cg(self, max_it=10000, gtol=1e-3):
    #     """
    #     Performs constrained maximization of the penalized log-likelihood function
    #     using the Conjugate Gradient (CG) optimization method.

    #     Parameters:
    #     -----------
    #     max_it : int, optional (default=10000)
    #         The maximum number of iterations for the CG optimization.
    #     gtol : float, optional (default=1e-3)
    #         The gradient norm tolerance for stopping criteria.

    #     """

    #     W_0 = self.sample_W_0()
    #     X_0 = sigma(W_0 @ self.X_n)
    #     ups_0 = self.vectorize(W_0, X_0)
    #     self.print_status(ups_0)

    #     res = minimize(
    #         fun=lambda ups: -self.stand_pen_log_lh(
    #             self.extract_W(ups), self.extract_X(ups)
    #         ),
    #         x0=ups_0,
    #         method="CG",
    #         jac=lambda ups: -self.gradient_ups(ups),
    #         callback=lambda x: self.callback(x),
    #         options={
    #             "disp": True,
    #             "maxiter": max_it,
    #             "gtol": gtol,
    #             "norm": np.inf,
    #             "return_all": True,
    #         },
    #     )

    #     self.set_value_hist(res.nit)

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
            np.hstack([np.outer(self.X_n, np.eye(self.K)[:, k]) for i in range(self.n)])
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
            "trials": self.trials,
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
            "W_tilde_s": self.W_tilde_s.tolist(),
            "X_tilde_s": self.X_tilde_s.tolist(),
            "iterations": self.iterations,
            "losses": self.losses.tolist(),
            "pen_log_LHs": self.pen_log_LHs.tolist(),
            "log_LHs": self.log_LHs.tolist(),
            "log_LH_Ws": self.log_LH_Ws.tolist(),
            "cross_entr_s": self.cross_entr_s.tolist(),
        }
        name = f"data/mc/{folder + '/' if folder else ""}n_{self.n}_d_{self.d}_K_{self.K}s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"
        with open(name, "w") as json_file:
            json.dump(result_dic, json_file, indent=4)

    def simulate(self):
        num_workers = min(
            multiprocessing.cpu_count(), self.trials
        )  # Get available cores
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(self.maximize_newton)(i) for i in tqdm(range(self.trials))
        )
        print("Done!")
        self.set_results(results)
