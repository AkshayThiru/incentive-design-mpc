# Incentive Design for Model Predictive Control of EV Charging Stations

This repository contains a Python implementation of the example presented in the paper 'Dynamic Incentive Selection for Hierarchical Convex Model Predictive Control.'
The code demonstrates how to optimize real-time pricing to balance electricity grid demand and social objectives.

## Project Structure

- `chargingstation/`
  - Core modules for the hierarchical MPC problem: lower-level and bilevel MPC, charging station model, and electricity price solver and regularizer.
  - `data/`: Contains sample demand/load data.
  - `example/`: Example scripts for running real-time price control and generating plots.
  - `plots/`: Scripts and images for visualizing results.
  - `test/`: Unit tests for key modules.

## Getting Started

1. **Environment Setup**
   - Use the provided `environment.yml` to set up the required Python environment:
     ```bash
     conda env create -f environment.yml
     conda activate incentive
     ```

2. **Running the Example**
   - Set the Python path:
     ```
     export PYTHONPATH=/path/to/this/directory:$PYTHONPATH
     ```
   - Execute the main example script:
     ```bash
     python chargingstation/example/real_time_price_control.py
     ```
   - Plots and logs will be generated in the `plots/` and `example/` directories.


## Paper Reference

This code is based on the example presented in the following paper:

> **Dynamic Incentive Selection for Hierarchical Convex Model Predictive Control**  
> Thirugnanam, Akshay, and Koushil Sreenath  
> arXiv preprint, 2025.  
> arXiv:2502.04642

## BibTeX Citation

You can cite the paper as follows:

```bibtex
@article{thirugnanam2025dynamic,
  title={Dynamic Incentive Selection for Hierarchical Convex Model Predictive Control},
  author={Thirugnanam, Akshay and Sreenath, Koushil},
  journal={arXiv preprint arXiv:2502.04642},
  year={2025}
}
```
