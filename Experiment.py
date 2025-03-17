import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, softmax
from utils import stable_softmax, sigma, sigma_der, sigma_inv
class Experiment():
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
        
    def sample_norm_const(self):
        """
        Samples the normalization constant `C` from a uniform distribution.

        Returns:
        --------
        float
            The sampled normalization constant `C`.
        """
        return self.rng.uniform(2 * np.log(self.K), 3 * np.log(self.K), size=(1, self.n))

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

        return self.rng.standard_normal((self.K, self.d)) / np.sqrt(self.d)

    def sample_features(self):
        """
        Computes the feature matrix X_n
        Returns:
            np.ndarray: The feature matrix X_n.
        """
        X_n = np.linalg.pinv(self.W) @ sigma_inv(self.norm_constants + np.log(self.exp_Y) - np.min(np.log(self.exp_Y), axis=0))

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
        return (np.sum(self.Y * sigma(W @ self.X_n)) - np.sum(logsumexp(sigma(W @ self.X_n), axis=0))) / self.n
    
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
        return (np.sum(self.Y * X) - np.sum(logsumexp(X, axis=0)) - lamb / 2 * np.linalg.norm(X - sigma(W @ self.X_n), 'fro')**2) / self.n

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

        return self.stand_log_lh(W, X, lamb) - lambda_X / 2 * np.linalg.norm(X, 'fro')**2 / self.n - lambda_W / 2 * np.linalg.norm(W, 'fro')**2 / self.n
        
    def plot_convergence(self):
        log_lh_optimum = self.stand_log_lh_W(self.W)

        plt.figure(figsize=(4, 3), dpi=300)

        plt.plot(self.pen_logLHs, label=r"$\frac{1}{n}\mathcal{L}_{\mathcal{G}}(W_i, \mathbb{X}_i)$", 
                linewidth=.5, alpha=0.8, color='navy')
        plt.plot(self.logLH_Ws, label=r"$\frac{1}{n}\mathbb{E}\mathcal{L}(W_i)$", 
                linewidth=.5, alpha=0.8, color='darkred')

        # Optimal log-likelihood reference line
        plt.hlines(log_lh_optimum, 0, self.iterations, color='skyblue', 
                linestyle='dashed', linewidth=1, label=r"$\frac{1}{n}\mathbb{E}\mathcal{L}(W^*)$")


        # plt.title("Convergence of Log-Likelihood", fontsize=12, fontweight='bold')

        plt.xlabel(r"$i$", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
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
        grad_phi = self.Y - X * softmax(X, axis=0)
        grad_str = - lamb * (X - sigma(W @ self.X_n)) 
        grad_pen = - lambda_X * X

        return (grad_phi + grad_str + grad_pen) / self.n
    
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
        grad_str = lamb * np.sum(np.einsum('ki, ji->ikj', (X - sigma(W @ self.X_n)) * sigma_der(W @ self.X_n), self.X_n, order='C'), axis=0) # sum over i
        
        # gradient of ridge penalty
        grad_pen = - lambda_W * W
        
        return (grad_str + grad_pen) / self.n
    
    def maximize(self, lamb, lambda_W, lambda_X, max_it, eps, gamma):
        """
        Performs gradient ascent to find maximizers W and X of the log-likelihood.

        The function computes the gradients of the loss function with respect to `W` and `X`, and updates the matrices
        iteratively based on these gradients. The process includes both structured penalties and log-likelihood regularization
        terms to prevent overfitting. Convergence is determined by checking the changes in `W` and `X` against given thresholds.

        Parameters:
        -----------
        
        lamb : float
            The scaling factor for the structured loss term.
        
        lambda_W : float
            The regularization strength for the weight matrix `W`.
        
        lambda_X : float
            The regularization strength for the input matrix `X`.
        
        max_it : int
            The maximum number of iterations for the optimization process.
        
        eps : float
            The convergence threshold
        
        gamma : float
            The step size for updating `W` and `X`.
            
        """
        
        self.pen_logLHs = np.zeros(max_it)
        self.logLH_Ws = np.zeros(max_it)
        self.prediction_err_s = np.zeros(max_it)
        self.losses = np.zeros(max_it)
        
        # initialization
        W_cur = self.W + self.rng.standard_normal((self.K, self.d)) / np.sqrt(self.d) * 1e-2
        # W_cur = self.W * 1
        X_cur = sigma(W_cur @ self.X_n)
        
        for i in range(max_it):
            # if i % 20 == 0:
            print(self.stand_pen_log_lh(W_cur, X_cur, lamb, lambda_X, lambda_W))

            # gradient ascent step
            W_old = W_cur
            X_old = X_cur
            grad_W = self.gradient_W(W_cur, X_cur, lamb, lambda_W)
            grad_X = self.gradient_X(W_cur, X_cur, lamb, lambda_X)
            W_cur = W_old + gamma * grad_W
            X_cur = X_old + gamma * grad_X

            self.losses[i] = np.linalg.norm(self.W - W_cur, 'fro')**2
            self.logLH_Ws[i] = self.stand_log_lh_W(W_cur)
            self.pen_logLHs[i] = self.stand_pen_log_lh(W_cur, X_cur, lamb, lambda_X, lambda_W)
            self.prediction_err_s[i] = np.linalg.norm(softmax(sigma(W_cur @ self.X_n), axis=0) - self.Y, 'fro')**2 / self.n

            if i == 0:
                continue
            else:
                diff = np.abs(self.stand_pen_log_lh(W_cur, X_cur, lamb, lambda_X, lambda_W) - self.stand_pen_log_lh(W_old, X_old, lamb, lambda_X, lambda_W))
                if (diff < eps):
                    break
                
        # save computed maximizers of the pen. log-likelihood  
        self.W_tilde = W_cur
        self.X_tilde = X_cur
        self.losses = self.losses[:i]
        self.logLH_Ws = self.logLH_Ws[:i]
        self.pen_logLHs = self.pen_logLHs[:i]
        self.prediction_err_s = self.prediction_err_s[:i]
        self.iterations = i
