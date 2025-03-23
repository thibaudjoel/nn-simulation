import json
import numpy as np
import matplotlib.pyplot as plt

class ConvergenceData:
    def __init__(self, folder, n, d, K, s_eta, lamb, lambda_W, lambda_X):
        self.folder = folder
        self.n = n
        self.d = d
        self.K = K
        self.s_eta = s_eta
        self.lamb = lamb
        self.lambda_W = lambda_W
        self.lambda_X = lambda_X
        self.filename = self.filname_from_attrs()
        self.load_from_json(filename=self.filename)
    
    def from_dict(self, data):
        """ Create an instance from a dictionary """
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

    def load_from_json(self, filename="result.json"):
        """ Load instance from JSON file """
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        return self.from_dict(data)
    
    def filname_from_attrs(self):
        return f"results/{self.folder}/n_{self.n}_d_{self.d}_K_{self.K}s_eta_{self.s_eta}_la_{self.lamb}_laW_{self.lambda_W}_lax_{self.lambda_X}.json"
    

    def plot(self, save_path=None, dpi=300):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "mathpazo",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            'text.latex.preamble': r'\usepackage{amsfonts, bm}'
        })

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        # Log Likelihoods
        axes[0].plot(self.pen_log_LHs, label=r"$\frac{1}{n} \mathcal{L}_\mathcal{G}(\bm{\upsilon}_i)$", color=colors[0], linewidth=1.5)
        axes[0].plot(self.log_LHs, label=r"$\frac{1}{n} \mathcal{L}(\bm{\upsilon}_i)$", color=colors[1], linestyle="--", linewidth=1.5)
        axes[0].set_xlabel(r"$i$")
        axes[0].legend(frameon=False)
        axes[0].grid(True, linestyle=":", linewidth=0.7)
        axes[0].set_title("(penalized) log-ikelihood")

        # Losses
        axes[1].plot(self.losses, label=r"$\|W_i-W^*\|_F$", color=colors[2], linewidth=1.5)
        axes[1].set_xlabel(r"$i$")
        axes[1].legend(frameon=False)
        axes[1].grid(True, linestyle=":", linewidth=0.7)
        axes[1].set_title("Loss")

        # Cross entropies
        axes[2].plot(self.cross_entr_s, label=r"$-\mathcal{L}(W_i)$", color=colors[3], linewidth=1.5)
        axes[2].set_xlabel("i")
        axes[2].legend(frameon=False)
        axes[2].grid(True, linestyle=":", linewidth=0.7)
        axes[2].set_title("Cross entropy")
        
        fig.suptitle(
            f"Model performance for $\\lambda={self.lamb}$, $\\lambda_W={self.lambda_W}$, $\\lambda_{{\\mathbb{{X}}}}={self.lambda_X}$, $s(\\bm{{\\eta}})={self.s_eta}$",
            fontsize=16, y=1.05
        )

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        plt.show()
