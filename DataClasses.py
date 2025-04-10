import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class ConvergenceData:
    """
    Class to load convergence data from JSON files.

    Attributes:
        n (int): Number of samples.
        d (int): Number of features.
        K (int): Number of categories.
        s_eta (float): Probability assigned to a high-probability category.
        lamb (float): Regularization coefficient for the structural penalty.
        lambda_W (float): Regularization coefficient for the weight matrix W.
        lambda_X (float):  Regularization coefficient for the nuisance parameter X.
        W_0_factor (float): Initial weight matrix.
        filename (str): Filename generated based on parameters.
    """

    def __init__(self, n, d, K, s_eta, lamb, lambda_W, lambda_X, W_0_factor):
        """
        Initialize the class with given parameters and load data from JSON.
        """
        self.n = n
        self.d = d
        self.K = K
        self.s_eta = s_eta
        self.lamb = lamb
        self.lambda_W = lambda_W
        self.lambda_X = lambda_X
        self.W_0_factor = W_0_factor
        self.filename = self.filname_from_attrs()
        self.load_from_json(filename=self.filename)

    def from_dict(self, data):
        """
        Populate the instance attributes using a dictionary of convergence data.

        Args:
            data (dict): Dictionary containing the convergence data
        """
        self.iterations = np.array(data["iterations"])
        self.exp_Y = np.array(data["exp_Y"])
        self.Y = np.array(data["Y"])
        self.W = np.array(data["W"])
        self.X_n = np.array(data["X_n"])
        self.W_tilde = np.array(data["W_tilde"])
        self.X_tilde = np.array(data["X_tilde"])
        self.iterations = np.array(data["iterations"])
        self.losses = np.array(data["losses"])
        self.losses_X = np.array(data["losses_X"])
        self.pen_log_LHs = np.array(data["pen_log_LHs"])
        self.log_LHs = np.array(data["log_LHs"])
        self.log_LH_Ws = np.array(data["log_LH_Ws"])
        self.cross_entr_s = np.array(data["cross_entr_s"])
        self.cross_entr_s_X = np.array(data["cross_entr_s_X"])

    def load_from_json(self, filename):
        """
        Load convergence data from JSON file and fill class attributes with the data from the file.

        Args:
            filename (str): Path to the JSON file containing the convergence data.
        """
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        self.from_dict(data)

    def filname_from_attrs(self):
        """
        Generates the filename with the convergence data corresponding to the attributes.

        Returns:
            str: A string with the name of the JSON file
        """
        return f"data/conv/n_{self.n}_d_{self.d}_K_{self.K}_factor_{self.W_0_factor}_s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"


class ConvergenceDataMultiple:
    """
    A class to and compare multiple convergence results from varying hyperparameter settings.

    Attributes:
        n_s (list[int]): List of number of samples.
        d (int): List of number of features.
        K (int): List of number of categories.
        s_eta_s (list[float]): List of probabilities assigned to high probability category.
        lamb_s (list[float]): List of regularization coefficients for structural penalty.
        lambda_W_s (list[float]): List of regularization coefficient for the weight matrix W.
        lambda_X_s (list[float]): List of regularization coefficient for the nuisance parameter X.
        W_0_factor_s (list[float]): List of factors for initial weight matrix
        conv_results (list[ConvergenceData]): List of ConvergenceData objects generated from parameter combinations.
        legend_values (list): Values used for labeling the legend in plots based on comparison parameter.
        legend_label (str): LaTeX-style label for the legend based on the comparison parameter.
    """

    def __init__(
        self,
        n_s,
        d,
        K,
        s_eta_s,
        lamb_s,
        lambda_W_s,
        lambda_X_s,
        W_0_factor_s,
        compare="lambda",
    ):
        """
        Initialize the class and load multiple ConvergenceData instances.

        Args:
            n_s (list[int]): List of number of samples.
            d (int): List of number of features.
            K (int): List of number of categories.
            s_eta_s (list[float]): List of probabilities assigned to high probability category.
            lamb_s (list[float]): List of regularization coefficients for structural penalty.
            lambda_W_s (list[float]): List of regularization coefficient for the weight matrix W.
            lambda_X_s (list[float]): List of regularization coefficient for the nuisance parameter X.
            W_0_factor_s (list[float]): List of factors for initial weight matrix
            compare (str): Parameter to use for differentiating curves in the plot legend.
                           Must be one of "n", "s_eta", "lambda", "lambda_W", "lambda_X", or "W_0".
        """
        self.n_s = n_s
        self.d = d
        self.K = K
        self.s_eta_s = s_eta_s
        self.lamb_s = lamb_s
        self.lambda_W_s = lambda_W_s
        self.lambda_X_s = lambda_X_s
        self.W_0_factor_s = W_0_factor_s
        self.conv_results = []
        for n, s_eta, lamb, lambda_W, lambda_X, W_0_factor in product(
            self.n_s,
            self.s_eta_s,
            self.lamb_s,
            self.lambda_W_s,
            self.lambda_X_s,
            self.W_0_factor_s,
        ):
            self.conv_results.append(
                ConvergenceData(
                    n,
                    self.d,
                    self.K,
                    s_eta,
                    lamb,
                    lambda_W,
                    lambda_X,
                    W_0_factor,
                )
            )

        self.legend_values = {
            "n": self.n_s,
            "s_eta": self.s_eta_s,
            "lambda": self.lamb_s,
            "lambda_W": self.lambda_W_s,
            "lambda_X": self.lambda_X_s,
            "W_0": self.W_0_factor_s,
        }[compare]
        self.legend_label = {
            "n": r"$n",
            "s_eta": r"$s(\eta)$",
            "lambda": r"$\lambda$",
            "lambda_W": r"$\lambda_W$",
            "lambda_X": r"$\lambda_{{\mathbb{X}}}$",
            "W_0": r"$\gamma$",
        }[compare]

    def create_plots(self, save=False):
        """
        Create comparison plots for penalized log-likelihood and loss across multiple configurations.

        Args:
            save (bool): If True, saves the generated plot as a PDF file in the 'imgs/conv/' directory.
                         Filename is generated based on input hyperparameters.
        """
        plt.style.use("ggplot")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "axes.titlesize": 12,
                "axes.labelsize": 12,
                "legend.fontsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "text.latex.preamble": r"\usepackage{amsfonts, bm, mathpazo}",
                "axes.facecolor": "white",
                "grid.color": "grey",
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=600)

        for i, conv_result in enumerate(self.conv_results):
            # penalized Likelihoods
            axes[0].plot(
                conv_result.pen_log_LHs,
                label=self.legend_label + "$=$" + str(self.legend_values[i]),
                linewidth=1.5,
            )
            # Losses
            axes[1].plot(
                conv_result.losses,
                linewidth=1.5,
                label=self.legend_label + "$=$" + str(self.legend_values[i]),
            )
        axes[0].spines["top"].set_color("black")
        axes[0].spines["right"].set_color("black")
        axes[0].spines["bottom"].set_color("black")
        axes[0].spines["left"].set_color("black")
        axes[0].set_xlabel(r"$i$")
        axes[0].set_ylabel(
            r"$\frac{1}{n} \mathcal{L}_\mathcal{G}(\bm{\tilde{\upsilon}}_i)$"
        )
        axes[0].legend(frameon=True, fontsize=10, loc="best")
        axes[0].grid(True, linestyle="-", linewidth=0.3, alpha=0.7)
        axes[0].set_title("penalized log-likelihood")
        axes[1].spines["top"].set_color("black")
        axes[1].spines["right"].set_color("black")
        axes[1].spines["bottom"].set_color("black")
        axes[1].spines["left"].set_color("black")
        axes[1].set_xlabel(r"$i$")
        axes[1].set_ylabel(r"$\|\widetilde{W}_i-W^*\|_F$")
        axes[1].legend(frameon=True, fontsize=10, loc="best", )
        axes[1].grid(True, linestyle="-", linewidth=0.3, alpha=0.7)
        axes[1].set_title("loss")
        plt.tight_layout()
        if save:
            path = f"imgs/conv/n_{self.n_s}_d_{self.d}_K_{self.K}_factor_{self.W_0_factor_s}_s_eta_{self.s_eta_s}_la_{self.lamb_s}_laW_{self.lambda_W_s}_lax_{self.lambda_X_s}.pdf"
            plt.savefig(path, dpi=600)

        plt.show()


class MonteCarloDataSingle:
    """
    Class to load results from a single Monte Carlo simulation.

    Attributes:
        n (int): Number of samples.
        d (int): Number of features.
        K (int): Number of categories.
        s_eta (float): Probability for high probability category.
        lamb (float): Regularization coefficient for the structural penalty.
        lambda_W (float): Regularization coefficient for the weight matrix W.
        lambda_X (float): Regularization coefficient the nuisance parameter X.
        filename (str): Path to the JSON file containing the Monte Carlo results.
    """

    def __init__(self, n, d, K, s_eta, lamb, lambda_W, lambda_X):
        """
        Initialize class and load data from JSON file.
        """
        self.n = n
        self.d = d
        self.K = K
        self.s_eta = s_eta
        self.lamb = lamb
        self.lambda_W = lambda_W
        self.lambda_X = lambda_X
        self.filename = self.filename_from_attrs()
        self.load_from_json(filename=self.filename)

    def from_dict(self, data):
        """
        Populate attributes from a dictionary.

        Args:
            data (dict): Dictionary containing simulation results.
        """
        self.trials = np.array(data["trials"])
        self.exp_Y = np.array(data["exp_Y"])
        self.Y = np.array(data["Y"])
        self.W = np.array(data["W"])
        self.X_n = np.array(data["X_n"])
        self.W_tilde_s = np.array(data["W_tilde_s"])
        self.X_tilde_s = np.array(data["X_tilde_s"])
        self.iterations = np.array(data["iterations"])
        self.losses = np.array(data["losses"])
        self.pen_log_LHs = np.array(data["pen_log_LHs"])
        self.log_LHs = np.array(data["log_LHs"])
        self.log_LH_Ws = np.array(data["log_LH_Ws"])
        self.cross_entr_s = np.array(data["cross_entr_s"])

    def load_from_json(self, filename):
        """
        Load simulation results from JSON file.

        Args:
            filename (str): Path to the JSON file containing simulation results.
        """
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        self.from_dict(data)

    def filename_from_attrs(self):
        """
        Generates the filename string with the Monte Carlo data corresponding to the class's attributes.

        Returns:
            str: A string with the name of the JSON file.
        """
        return f"data/mc/n_{self.n}_d_{self.d}_K_{self.K}_s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"


class MonteCarloData:
    """
    A class to manage and visualize Monte Carlo simulation results for varying hyperparameter settings.

    This class aggregates multiple instances of MonteCarloDataSingle, corresponding to different
    combinations of hyperparameters, and supports comparative analysis via histogram plotting.

    Attributes:
        n_s (list[int]): List of number of samples.
        d (int): List of number of features.
        K (int): List of number of categories.
        s_eta_s (list[float]): List of probabilities assigned to high probability category.
        lamb_s (list[float]): List of regularization coefficients for structural penalty.
        lambda_W_s (list[float]): List of regularization coefficient for the weight matrix W.
        lambda_X_s (list[float]): List of regularization coefficient for the nuisance parameter X.
        mc_results (list[MonteCarloDataSingle]): List of results for each parameter combination.
        legend_values (list): The values to be shown in the plot legend, depending on the comparison dimension.
        legend_label (str): The LaTeX-formatted label for the comparison parameter.
    """

    def __init__(
        self,
        n_s,
        d,
        K,
        s_eta_s,
        lamb_s,
        lambda_W_s,
        lambda_X_s,
        compare="lambda",
    ):
        """
        Initialize a MonteCarloData instance and populate results for each parameter combination.

        Args:
        n_s (list[int]): List of number of samples.
        d (int): List of number of features.
        K (int): List of number of categories.
        s_eta_s (list[float]): List of probabilities assigned to high probability category.
        lamb_s (list[float]): List of regularization coefficients for structural penalty.
        lambda_W_s (list[float]): List of regularization coefficient for the weight matrix W.
        lambda_X_s (list[float]): List of regularization coefficient for the nuisance parameter X.
            compare (str): The parameter to use for comparing results in plots.
                           One of ["n", "s_eta", "lambda", "lambda_W", "lambda_X"].
        """
        self.n_s = n_s
        self.d = d
        self.K = K
        self.s_eta_s = s_eta_s
        self.lamb_s = lamb_s
        self.lambda_W_s = lambda_W_s
        self.lambda_X_s = lambda_X_s
        self.mc_results = []
        for n, s_eta, lamb, lambda_W, lambda_X in product(
            self.n_s, self.s_eta_s, self.lamb_s, self.lambda_W_s, self.lambda_X_s
        ):
            self.mc_results.append(
                MonteCarloDataSingle(n, self.d, self.K, s_eta, lamb, lambda_W, lambda_X)
            )

        self.legend_values = {
            "n": self.n_s,
            "s_eta": self.s_eta_s,
            "lambda": self.lamb_s,
            "lambda_W": self.lambda_W_s,
            "lambda_X": self.lambda_X_s,
        }[compare]
        self.legend_label = {
            "n": r"$n",
            "s_eta": r"$s(\eta)$",
            "lambda": r"$\lambda$",
            "lambda_W": r"$\lambda_W$",
            "lambda_X": r"$\lambda_{{\mathbb{X}}}$",
        }[compare]

    def create_plots(self, save=False):
        """
        Create and display histograms of the loss values from the Monte Carlo simulations.

        The visualization compares the distributions of losses under different hyperparameter
        settings as specified by the 'compare' argument in the constructor.

        Args:
            save (bool): Whether to save the generated plot as a PDF file.
        """
        plt.style.use("ggplot")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "text.latex.preamble": r"\usepackage{amsfonts, bm, mathpazo}",
                "axes.facecolor": "white",
            }
        )

        plt.figure(figsize=(3, 2.16), dpi=600)

        for i, mc_result in enumerate(self.mc_results):
            plt.hist(
                mc_result.losses,
                range=(np.min(mc_result.losses), np.max(mc_result.losses)),
                bins="auto",
                edgecolor="black",
                alpha=0.4,
                label=self.legend_label + r"$=$" + rf"{self.legend_values[i]}",
            )

        plt.xlabel(r"$\|\widetilde{W} - W^*\|_F$", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)

        plt.legend(frameon=True, fontsize=10, loc="best")
        ax = plt.gca()
        ax.spines["top"].set_color("black")
        ax.spines["right"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")

        plt.grid(False)
        plt.tight_layout()
        if save:
            path = f"imgs/mc/n_{self.n_s}_d_{self.d}_K_{self.K}_s_eta_{self.s_eta_s}_la_{self.lamb_s}_laW_{self.lambda_W_s}_lax_{self.lambda_X_s}.pdf"
            plt.savefig(path, dpi=600, bbox_inches="tight")
        plt.show()
