# Quasigroup with Holes Solver using D-Wave Leap Hybrid Sampler

This program solves Quasigroup with Holes (QWH) problems using the D-Wave Leap Hybrid Sampler. It takes an initial QWH problem as a NumPy array and returns a solution that satisfies the problem if one exists.

## Prerequisites

- Python 3.7 or higher
- D-Wave Ocean SDK
- Access to a D-Wave quantum annealer

## Installation

1. Install the D-Wave Ocean SDK by following the instructions in the [official documentation](https://docs.ocean.dwavesys.com/en/latest/overview/install.html).
2. Set up your D-Wave API token by following the instructions in the [official documentation](https://docs.ocean.dwavesys.com/en/latest/overview/dwave_cloud.html).

## Usage

1. Provide the initial Quasigroup with Holes problem as a NumPy array:

```python
qwh = np.array([
    [1, 2, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [4, 1, 0, 0]
])
n = 4
```
Run the Python script:

  ```python quasigroup_solver.py```
  
This will solve the predefined QWH problem and output the solution.

Modifying the QWH Problem

To solve a different QWH problem, modify the qwh variable in the Python script. The qwh variable should be a NumPy array representing the initial QWH problem.

For example:

```python

qwh = np.array([
    [1, 2, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [4, 1, 0, 0]
])
n = 4
```
