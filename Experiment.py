import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.linalg import block_diag
from utils import sigma, sigma_der, sigma_inv, sigma_sec_der
from scipy.optimize import minimize

class Experiment:
    def __init__(self, n, d, K, s_eta, rng):
        self.rng = rng
        self.n = n
        self.d = d
        self.K = K
        self.s_eta = s_eta
        self.exp_Y = self.sample_exp_Y()
        self.Y = self.sample_Y()
        self.W = self.sample_W()
        self.norm_constants = self.sample_norm_const()
        self.X_n = self.sample_features()
        self.W_tilde = np.zeros((K, d))
        self.X_tilde = np.zeros_like(self.Y)
        self.pen_logLHs = np.zeros(1)
        self.logLH_Ws = np.zeros(1)
        self.prediction_err_s = np.zeros(1)
        self.iterations = 0
        self.losses = np.zeros(1)
        self.hist = []

    def sample_norm_const(self):
        """
        Samples the normalization constant `C` from a uniform distribution.

        Returns:
        --------
        float
            The sampled normalization constant `C`.
        """

        return 1e-4 #self.rng.uniform(np.log(1e0 * self.K), np.log(1e6 * self.K), size=(1, self.n))

    def sample_exp_Y(self):
        """
        Samples category probabilities from a Dirichlet distribution and scales them so that one category
        receives a higher probability (`s_eta`) while the remaining categories sum to `1 - s_eta`.

        This function first samples `n` Dirichlet distributions over `K` categories. Then, for each sample,
        it sets the probability of the chosen category (for that sample) to 0, scales the remaining probabilities
        so that their sum is `1 - s_eta`, and finally assigns `s_eta` to the chosen category.

        """
        # sample classes with high probability
        categories = self.rng.integers(0, self.K, self.n)

        # sample probability vectors
        probabilities = self.rng.dirichlet(np.ones(self.K), size=self.n).T + 1 / self.K
        # probabilities = self.rng.dirichlet(1e2 * self.K * np.ones(self.K), size=self.n).T

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
        return self.rng.standard_normal((self.K, self.d)) #/ np.sqrt(self.d * self.K) 

    def sample_features(self):
        """
        Computes the feature matrix X_n
        Returns:
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
        float: The computed log-likelihood.
        """
        return (
            np.sum(self.Y * sigma(W @ self.X_n))
            - np.sum(logsumexp(sigma(W @ self.X_n), axis=0))
        ) / self.n

    def stand_log_lh(self, W, X, lamb):
        """
        Computes the standardized log-likelihood.

        Parameters:
        W (ndarray): Weight matrix.
        X (ndarray): Current latent variable representation.
        lamb (float): Regularization coefficient for the structural penalty.

        Returns:
        float: The computed log-likelihood.
        """
        return (
            np.sum(self.Y * X)
            - np.sum(logsumexp(X, axis=0))
            - lamb / 2 * np.linalg.norm(X - sigma(W @ self.X_n), "fro") ** 2
        ) / self.n

    def stand_pen_log_lh(self, W, X, lamb, lambda_X, lambda_W):
        """
        Computes the standardized penalized log-likelihood function with additional regularization terms.

        Parameters:
        W (ndarray): Weight matrix.
        X (ndarray): Current latent variable representation.
        lamb (float): Regularization coefficient for structural penalty.
        lamb_X (float): Regularization coefficient for X.
        lamb_W (float): Regularization coefficient for W.

        Returns:
        float: The penalized log-likelihood.
        """

        return (
            self.stand_log_lh(W, X, lamb)
            - lambda_X / 2 * np.linalg.norm(X, "fro") ** 2 / self.n
            - lambda_W / 2 * np.linalg.norm(W, "fro") ** 2 / self.n
        )
    
    def pred_err(self, W):
        return -np.sum(self.Y * np.log(softmax(sigma(W @ self.X_n)))) / self.n #np.linalg.norm(softmax(sigma(W @ self.X_n)) - self.Y, 1) / self.n
    
    def pred_err_X(self, X):
        return -np.sum(self.Y * np.log(softmax(X))) / self.n # np.linalg.norm(softmax(X) - self.Y, 1) / self.n
    
    def loss(self, W):
        return np.linalg.norm(self.W - W, 'fro') / np.linalg.norm(self.W, 'fro')
    
    def loss_X(self, X):
        return np.linalg.norm(X - sigma(self.W @ self.X_n), 'fro') / np.linalg.norm(sigma(self.W @ self.X_n), 'fro')

    def plot_convergence(self):
        log_lh_optimum = self.stand_log_lh(self.W, sigma(self.W @ self.X_n), lamb=1)
        print(log_lh_optimum)

        plt.figure(figsize=(4, 3), dpi=300)

        plt.plot(
            self.pen_logLHs,
            label=r"$\frac{1}{n}\mathcal{L}_{\mathcal{G}}(W_i, \mathbb{X}_i)$",
            linewidth=0.5,
            alpha=0.8,
            color="navy",
        )
        # plt.plot(self.logLH_Ws, label=r"$\frac{1}{n}\mathbb{E}\mathcal{L}(W_i)$",
        #         linewidth=.5, alpha=0.8, color='darkred')

        # Optimal log-likelihood reference line
        # plt.hlines(log_lh_optimum, 0, self.iterations, color='skyblue',
        #         linestyle='dashed', linewidth=1, label=r"$\frac{1}{n}\mathcal{L}(W^*, \mathbb{X}^*)$")

        # plt.title("Convergence of Log-Likelihood", fontsize=12, fontweight='bold')

        plt.xlabel(r"$i$", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12, loc="best", frameon=True)

        # Tight layout for better appearance
        plt.tight_layout()

        # Show plot
        plt.show()
        # plt.savefig(f"imgs/convergence_{self.n}_{self.lamb}_{self.lambda_W}_{self.lambda_X}_{self.d}_{self.K}_{self.max_it}_{self.eps_W}_{self.eps_X}_{self.gamma}.png")
        # plt.close()

    def gradient_X(self, W, X, lamb, lambda_X):
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
        gradient_struc = -lamb * (X - sigma(W @ self.X_n))
        gradient_ridge = -lambda_X * X

        return gradient_phi + gradient_struc + gradient_ridge

    def gradient_W(self, W, X, lamb, lambda_W):
        """
        Computes the gradient of the loss function with respect to the weight matrix `W`.

        Parameters:
        -----------
        W : numpy.ndarray
            The weight matrix for the model. Shape: (m, n), where m is the number of output units and n is the number of input features.

        X : numpy.ndarray
            The feature matrix. Shape: (m, N), where m is the number of output units and N is the number of data samples.

        lamb : float
            The scaling factor for the structured loss term. Controls the contribution of the structured loss in the final gradient.

        lamb_W : float
            The regularization strength for the weight matrix `W`. Controls the contribution of the L2 regularization term in the final gradient.

        Returns:
        --------
        numpy.ndarray
            The gradient of the loss with respect to the weight matrix `W`. This includes both the structured loss term and the L2 regularization term.
        """
        # gradient of structural penalty
        gradient_struc = lamb * np.sum(
            np.einsum(
                "ki, ji->ikj",
                (X - sigma(W @ self.X_n)) * sigma_der(W @ self.X_n),
                self.X_n,
                order="C",
            ),
            axis=0,
        )  # sum over i
        # gradient of ridge penalty
        gradient_ridge = -lambda_W * W

        return gradient_struc + gradient_ridge

   
    def maximize(self, lamb, lambda_W, lambda_X, max_it=1000, eps=1e-3):
        def callback(xk):
            W = xk[:self.K * self.d].reshape(self.W.shape)
            X = xk[self.K * self.d:].reshape(self.Y.shape)
            self.hist.append(xk)
            print(f"pred_err: {self.pred_err(W)}")
            print(f"pred_err_X: {self.pred_err_X(X)}")
            print(f"loss: {self.loss(W)}")
            print(f"loss_X: {self.loss_X(X)}")
            print(f"f(x) = {-objective(xk)}") 
            
        # W_0 = self.W + self.rng.standard_normal((self.K, self.d)) / np.sqrt(self.d * self.K) *1e-1
        W_0 = self.sample_W() #* 1e1
        X_0 = sigma(W_0 @ self.X_n)
        x_0 = np.concatenate((W_0.flatten(), X_0.flatten()))
        grad_x = lambda x: - np.concatenate((self.gradient_W(x[:self.K * self.d].reshape(self.W.shape), x[self.K * self.d:].reshape(self.Y.shape), lamb, lambda_W).flatten(), self.gradient_X(x[:self.K * self.d].reshape(self.W.shape), x[self.K * self.d:].reshape(self.Y.shape), lamb, lambda_X).flatten()))
        hess_x = lambda x: self.fisher(x, lamb, lambda_W, lambda_X)
        objective = lambda x: - self.stand_pen_log_lh(x[:self.K * self.d].reshape(self.W.shape), x[self.K * self.d:].reshape(self.Y.shape), lamb, lambda_X, lambda_W)
        res = minimize(objective, x_0, args=(), method='Newton-CG', jac=grad_x, hess=hess_x, tol=eps, callback=callback, options={'maxiter': max_it, 'disp': True, 'return_all': True})
        
        self.iterations = res.nit
        self.W_s = np.zeros((self.iterations, self.K, self.d))
        self.X_s = np.zeros((self.iterations, self.K, self.n))
        self.losses = np.zeros(self.iterations)
        self.losses_X = np.zeros(self.iterations)
        self.pred_err_s = np.zeros(self.iterations)
        self.pred_err_s_X = np.zeros(self.iterations)
        self.pen_log_LHs = np.zeros(self.iterations)
        self.log_LHs = np.zeros(self.iterations)
        self.log_LH_Ws = np.zeros(self.iterations)


        for i in range(self.iterations):
            W_tilde = self.hist[i][:self.K * self.d].reshape((self.K, self.d))
            X_tilde = self.hist[i][self.K * self.d:].reshape((self.K, self.n))
            self.W_s[i, :, :] = W_tilde
            self.X_s[i, :, :] = X_tilde
            self.losses[i] = self.loss(W_tilde)
            self.losses_X[i] = self.loss_X(X_tilde)
            self.pred_err_s[i] = self.pred_err(W_tilde)
            self.pred_err_s_X[i] = self.pred_err_X(X_tilde)
            self.pen_log_LHs[i] = self.stand_pen_log_lh(W_tilde, X_tilde, lamb, lambda_X, lambda_W)
            self.log_LHs[i] = self.stand_log_lh(W_tilde, X_tilde, lamb)
            self.log_LH_Ws[i] = self.stand_log_lh_W(W_tilde)

        self.W_tilde = self.W_s[-1,:,:]
        self.X_tilde = self.X_s[-1,:,:]
    
    def fisher(self, x, lamb, lambda_W, lambda_X):
        W = x[:self.K * self.d].reshape(self.W.shape)
        X = x[self.K * self.d:].reshape(self.Y.shape)
        # matrix of outer products of feature vectors
        inter = np.einsum("ij, jk-> ikj", self.X_n, self.X_n.T, order="K").reshape(
            (self.d, self.d * self.n), order="F"
        )
        # matrix of squared derivatives of the activation function
        sig_der_square = np.power(sigma(W @ self.X_n), 2)
        
        # matrix of structural term
        non_struct = - lamb * (X - sigma(W @ self.X_n)) * sigma_sec_der(W @ self.X_n)
        # augment the matrix for multiplication with the outer products of feature vectors
        factors = np.kron(sig_der_square + non_struct, np.ones((1, self.d))) 

        F_WW = lamb * block_diag(
            *[
                np.sum((factors[i, :] * inter).reshape((self.d, self.d, self.n), order="A"), axis=2)
                for i in range(self.K)
            ]
        ) + lambda_W * np.eye(self.K * self.d) 
        F_XX = self.var_Y(X) + (lamb + lambda_X) * np.eye(self.K * self.n)
        v_blocks = [
            np.hstack([np.outer(self.X_n[:, i], np.eye(self.K)[:, k]) for i in range(self.n)])
            for k in range(self.K)
        ]
        X_augmented = np.vstack(v_blocks)
        sig_der = np.kron(sigma_der(W @ self.X_n), np.ones((self.d, self.K)))
        F_WX = sig_der * X_augmented
        F = np.block([[F_WW, F_WX], [F_WX.T, F_XX]])

        return F
    
    def var_Y(self, X):
        """
        Computes the covariance matrix of n Multidimensional(1) distributed
        random vectors whose expected values are proportional to the columns of X.

        Parameters:
        -----------
       
        Returns:
        --------
        numpy.ndarray

        """
        probabilities = softmax(X, axis=0)
        diag = np.diag(probabilities.flatten("F"))

        blocks = [np.outer(probabilities[:, i], probabilities[:, i]) for i in range(self.n)]

        return diag - block_diag(*blocks)
        
