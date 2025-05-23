# Reservoir Computing and Control of Chaotic Systems

This repository contains a collection of Jupyter notebooks, scripts, and figures for experiments in reservoir computing, control of chaotic systems (primarily the Lorenz system), and related data-driven methods. The codebase includes simulations, control experiments, metric analyses, and visualizations for various dynamical systems.

## Table of Contents
- [Project Overview](#project-overview)
- [File Index](#file-index)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project explores the use of reservoir computing and other data-driven methods to control and forecast chaotic dynamical systems. The main focus is on the Lorenz system, but other systems such as Rossler and Sprott-Linz are also included. The repository features:
- Simulations of chaotic and periodic regimes
- Control strategies (including PID and model predictive control)
- Metric analysis and visualization
- Parameter variation studies
- Translation of MATLAB models to Python

## File Index
See `INDEX.md` for a detailed, one-line description of every file and folder in the repository.

## How to Use
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies:**
   Most notebooks require Python 3, NumPy, SciPy, and Matplotlib. Install with:
   ```sh
   pip install numpy scipy matplotlib jupyter
   ```
3. **Run Jupyter notebooks:**
   ```sh
   jupyter notebook
   ```
   Open any `.ipynb` file to explore simulations and analyses.

## Requirements
- Python 3.x
- Jupyter Notebook
- NumPy, SciPy, Matplotlib
- (Optional) ipywidgets, plotly for interactive dashboards
- (Optional) MATLAB/Simulink for `.m` and `.slx` files

## Repository Structure
- **Jupyter Notebooks:** Main experiments, simulations, and analyses
- **PNG/PDF/HTML:** Plots and exports of results
- **reservoir-computing/**: Python functions for reservoir computing
- **Seb/**: Additional scripts, figures, and MATLAB code for Sprott-Linz and Lorenz96 systems
- **INDEX.md:** Full file index with descriptions

## Contributing
Contributions are welcome! Please update `INDEX.md` and add clear descriptions for any new files or experiments.

## License
This project is for academic and research purposes. See individual files for license details if provided.
