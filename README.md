# nn-simulation

This repository contains the code used for the numerical experiments in my Master thesis:

> **_“Non-asymptotic guarantees for parameter inference in neural networks”_**

---

## Repository Overview

### Core Modules

- **`utils.py`**  
  - Helper functions to compute the activation function $\sigma$ and its derivatives.
  - Sets up folder structure for saving experiment results.

- **`Convergence.py`**  
  - Simulates Multinomial data.
  - Tracks the convergence of the MLE procedure for the neural network model.
  - Saves results as `.json`.

- **`MonteCarlo.py`**  
  - Runs Monte Carlo simulations for MLE procedures with various parameters.
  - Stores outcomes in `.json` format.

- **`DataClasses.py`**  
  - Loads experiment data from JSON files.
  - Visualizes results, with options to compare parameters and export plots.

---

## Notebooks

- **`experiments.ipynb`**
  - Sets up folder structure.
  - Runs both convergence and Monte Carlo simulations.

- **`plots.ipynb`**
  - Generates plots based on experimental results.

---

## Folder Structure

Before running experiments, ensure the following directory structure is created using `create_folders()` from `utils.py`:

```text
data/
├── conv/    # Convergence experiment results
├── mc/      # Monte Carlo simulation results

imgs/
├── conv/    # Convergence plots
├── mc/      # Monte Carlo plots
```


> **Important:**  
> Run `create_folders()` before executing any simulations. Otherwise, the code will fail when trying to save output files.

---

## Parallel Execution

The code is designed to run **in parallel** for faster computation.  
It is **highly recommended** to use a machine with **multiple CPU cores** to speed up processing time.
