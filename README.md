# Quasigroup with Holes Solver using D-Wave Leap Hybrid Sampler

This repository contains a Python script for solving Quasigroup With Holes problems using D-Wave's Quantum Annealing.

## Requirements

- Python 3.6 or higher
- dimod
- dwave-system
- dwave-hybrid
- dwave-cloud-client

## Installation

It is recommended to use a virtual environment to install the required libraries.

```bash
pip install virtualenv
virtualenv quasigroup_env
source quasigroup_env/bin/activate  # On Windows: quasigroup_env\Scripts\activate
pip install dimod dwave-system dwave-hybrid dwave-cloud-client
```
## Usage

The quasigroup_solver.py script can be executed from the command line. To run the script with a specific problem file:

```
python solver.py <path_to_problem_file>
```
To run the script in test mode, where it will read all .txt files in the tests folder and check if the solutions are correct:

```
python solver.py --test
```
## Problem File Format

The problem file should contain an NxN matrix where the numbers represent the known values in the Quasigroup With Holes problem and the zeros represent empty spaces. Each row should be on a separate line and the numbers should be separated by spaces. See the example files in the tests folder for the correct format.
