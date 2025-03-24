import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class ConvergenceData:
    def __init__(self, folder, n, d, K, s_eta, lamb, lambda_W, lambda_X, W_0_factor):
        self.folder = folder
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
        """Set attributes from dictionary"""
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
        """Load instance from JSON file"""
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        self.from_dict(data)

    def filname_from_attrs(self):
        return f"data/conv/{self.folder}/n_{self.n}_d_{self.d}_K_{self.K}_factor_{self.W_0_factor}_s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"

    def create_plot(self, save_path=None):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "mathpazo",
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "text.latex.preamble": r"\usepackage{amsfonts, bm}",
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        viridis = plt.cm.get_cmap("viridis", 3)
        color1, color2, color3 = viridis(0), viridis(0.5), viridis(1)

        # Log Likelihoods
        axes[0].plot(
            self.pen_log_LHs,
            label=r"$\frac{1}{n} \mathcal{L}_\mathcal{G}(\bm{\upsilon}_i)$",
            color=color1,
            linewidth=1.5,
        )
        axes[0].plot(
            self.log_LHs,
            label=r"$\frac{1}{n} \mathcal{L}(\bm{\upsilon}_i)$",
            linestyle="--",
            color=color2,
            linewidth=1.5,
        )
        axes[0].set_xlabel(r"$i$")
        axes[0].legend(frameon=False)
        axes[0].grid(True, linestyle=":", linewidth=0.7)
        axes[0].set_title("(penalized) log-likelihood")

        # Losses
        axes[1].plot(self.losses, label=r"$\|W_i-W^*\|_F$", color=color3, linewidth=1.5)
        axes[1].set_xlabel(r"$i$")
        axes[1].legend(frameon=False)
        axes[1].grid(True, linestyle=":", linewidth=0.7)
        axes[1].set_title("Loss")

        fig.suptitle(
            f"Model performance for $\\lambda={self.lamb}$, $\\lambda_W={self.lambda_W}$, $\\lambda_{{\\mathbb{{X}}}}={self.lambda_X}$, $s(\\bm{{\\eta}})={self.s_eta}$, factor: {self.W_0_factor}",
            fontsize=16,
            y=1.05,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


class ConvergenceDataMultiple:
    def __init__(
        self,
        folder,
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
        self.folder = folder
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
                    self.folder,
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

    def create_plots(self, save_path=None):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "mathpazo",
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "text.latex.preamble": r"\usepackage{amsfonts, bm}",
            }
        )

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.conv_results)))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, conv_result in enumerate(self.conv_results):
            # penalized Likelihoods
            axes[0].plot(
                conv_result.pen_log_LHs,
                label=self.legend_label + "$=$" + str(self.legend_values[i]),
                color=colors[i],
                linewidth=1.5,
            )
            axes[1].plot(
                conv_result.log_LHs,
                label=self.legend_label + "$=$" + self.legend_values[i],
                color=colors[i],
                linewidth=1.5,
            )

            # Losses
            axes[2].plot(
                conv_result.losses,
                color=colors[i],
                linewidth=1.5,
                label=self.legend_label + "$=$" + self.legend_values[i],
            )

        axes[0].set_xlabel(r"$i$")
        axes[0].set_ylabel(r"$\frac{1}{n} \mathcal{L}_\mathcal{G}(\bm{\upsilon}_i)$")
        axes[0].legend(frameon=False)
        axes[0].grid(True, linestyle=":", linewidth=0.7)
        axes[0].set_title("penalized log-likelihoods")
        axes[1].set_xlabel(r"$i$")
        axes[1].set_ylabel(r"$\frac{1}{n} \mathcal{L}(\bm{\upsilon}_i)$")
        axes[1].legend(frameon=False)
        axes[1].grid(True, linestyle=":", linewidth=0.7)
        axes[1].set_title("log-likelihoods")
        axes[2].set_xlabel(r"$i$")
        axes[2].set_ylabel(r"\|W_i-W^*\|")
        axes[2].legend(frameon=False)
        axes[2].grid(True, linestyle=":", linewidth=0.7)
        axes[2].set_title("Loss")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.tight_layout()
        plt.show()


class MonteCarloDataSingle:
    def __init__(self, folder, n, d, K, s_eta, lamb, lambda_W, lambda_X):
        self.folder = folder
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
        """Set attributes from dictionary"""
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
        """Load instance from JSON file"""
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        self.from_dict(data)

    def filename_from_attrs(self):
        return f"data/mc/{self.folder}/n_{self.n}_d_{self.d}_K_{self.K}_s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"


class MonteCarloData:
    def __init__(
        self,
        folder,
        n_s,
        d,
        K,
        s_eta_s,
        lamb_s,
        lambda_W_s,
        lambda_X_s,
        compare="lambda",
    ):
        self.folder = folder
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
                MonteCarloDataSingle(
                    self.folder, n, self.d, self.K, s_eta, lamb, lambda_W, lambda_X
                )
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

    def create_plots(self):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "mathpazo",
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "text.latex.preamble": r"\usepackage{amsfonts, bm}",
            }
        )

        colors = plt.cm.viridis(np.linspace(0, 1, len(self.mc_results)))

        plt.figure(figsize=(6, 4), dpi=300)

        for i, mc_result in enumerate(self.mc_results):
            plt.hist(
                mc_result.losses,
                range=(np.min(mc_result.losses), np.max(mc_result.losses)),
                bins="auto",
                color=colors[i],
                edgecolor="black",
                alpha=0.4,
                label=self.legend_label + r"$=$" + rf"{self.legend_values[i]}",
            )

        plt.xlabel(r"$\|\widetilde{W} - W^*\|_F$", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)

        plt.legend(frameon=True, fontsize=10, loc="upper right")

        plt.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
