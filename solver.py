from dwave.system import LeapHybridSampler
from dimod import BinaryQuadraticModel
import numpy as np
import dimod

def quasi_group_to_qubo(initial_problem, n):
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Create a dictionary to store variables
    variables = {}

    rows = range(n)
    columns = range(n)
    numbers = range(1, n + 1)

    # Iterate through each cell in the quasi-group grid
    for i in rows:
        for j in columns:
            for k in numbers:
                # Create a variable for each possible number k in cell (i, j)
                var_name = f'x_{i}_{j}_{k}'
                variables[var_name] = (i, j, k)
                bqm.add_variable(var_name, 0)

    # Row constraint: Every number between 1 and n (inclusive) must appear exactly once in each row
    for i in rows:
        for k in numbers:
            bqm.add_linear_equality_constraint(
                [(variables[f'x_{i}_{j}_{k}'], 1) for j in columns],
                lagrange_multiplier=50,
                constant = -1
            )

    # Column constraint: Every number between 1 and n (inclusive) must appear exactly once in each column
    for j in columns:
        for k in numbers:
            bqm.add_linear_equality_constraint(
                [(variables[f'x_{i}_{j}_{k}'], 1) for i in rows],
                lagrange_multiplier=50,
                constant=-1
            )

    return bqm, variables


def solve_quasi_group_bqm(bqm, variables, n):

    # Run on hybrid sampler
    print("\nRunning hybrid solver...")
    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm)

    # Get the best sample (lowest energy)
    best_sample = sampleset.first.sample

    # Convert the sample to a human-readable solution
    solution = np.zeros((n, n), dtype=int)

    for var_name, value in best_sample.items():
        if value == 1 and var_name in variables:
            i, j, k = variables[var_name]
            solution[i, j] = k

    return solution


def is_solution_valid(solution):
    n = solution.shape[0]
    for row in range(n):
        if len(set(solution[row, :])) != n:
            return False

    for col in range(n):
        if len(set(solution[:, col])) != n:
            return False

    return True

# Example Quasigroup with Holes problem
qwh = np.array([
    [1, 2, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [4, 1, 0, 0]
])
n = 4
# Create the BQM for the quasi-group with holes
bqm, variables = quasi_group_to_qubo(qwh, n)

# Solve the BQM
solution = solve_quasi_group_bqm(bqm, variables, n)

print("Solution:")
print(solution)
print("Solution is valid:", is_solution_valid(solution))
