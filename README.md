# Quasigroup with Holes Solver using D-Wave Leap Hybrid Sampler

This repository contains Python scripts for solving Quasigroup With Holes problems using both classical and D-Wave's Quantum Annealing approaches.

## Requirements

- Python 3.6 or higher
- dimod
- dwave-system
- dwave-hybrid
- dwave-cloud-client
- numpy
- matplotlib

## Installation

It is recommended to use a virtual environment to install the required libraries.

```bash
pip install virtualenv
virtualenv quasigroup_env
source quasigroup_env/bin/activate  # On Windows: quasigroup_env\Scripts\activate
pip install dimod dwave-system dwave-hybrid dwave-cloud-client numpy matplotlib
```

## Usage

There are three main Python scripts:

quasi_library.py: Contains common functions for both classical and quantum solvers.
quantum_solver.py: A quantum solver for the Quasigroup With Holes problem.
classic_solver.py: A classical solver for the Quasigroup With Holes problem.

To run the quantum solver with a specific problem file:

```python quantum_solver.py --file <path_to_problem_file>```

To run the classical solver with a specific problem file:

```python classic_solver.py --file <path_to_problem_file>```

To run either solver in test mode, where it will read all .txt files in the tests folder and check if the solutions are correct:


```python quantum_solver.py --test```

```python classic_solver.py --test```

To run performance tests with either solver:

```python quantum_solver.py --performance dim_start dim_end ratio_start ratio_end```

```python classic_solver.py --performance dim_start dim_end ratio_start ratio_end```

Problem File Format

The problem file should contain an NxN matrix where the numbers represent the known values in the Quasigroup With Holes problem and the zeros represent empty spaces. Each row should be on a separate line and the numbers should be separated by spaces. See the example files in the tests folder for the correct format.
