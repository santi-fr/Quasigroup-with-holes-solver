import os
import argparse
import copy
import dimod
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from dimod.generators.constraints import combinations
from hybrid.reference import KerberosSampler

def generate_quasigroup_problem(dimension, num_variables):
    # Genera un cuadrado latino completo
    latin_square = [[(i + j) % dimension + 1 for i in range(dimension)] for j in range(dimension)]
    
    # Revuelve el cuadrado latino
    random.shuffle(latin_square)
    for row in latin_square:
        random.shuffle(row)

    # Quita algunas celdas para crear el problema de quasigroup with holes
    variables_to_remove = num_variables
    while variables_to_remove > 0:
        i, j = random.randint(0, dimension - 1), random.randint(0, dimension - 1)
        if latin_square[i][j] != 0:
            latin_square[i][j] = 0
            variables_to_remove -= 1

    return latin_square

def run_performance_tests(start_dim, end_dim, start_vars, end_vars):
    performance_data = np.zeros((end_dim - start_dim + 1, end_vars - start_vars + 1))

    for dim in range(start_dim, end_dim + 1):
        for num_vars in range(start_vars, end_vars + 1):
            times = []
            for _ in range(10):
                matrix = generate_quasigroup_problem(dim, num_vars)
                start_time = time.time()
                bqm = build_bqm(matrix)
                result = solve_quasigroup(bqm, matrix)
                end_time = time.time()

                execution_time = end_time - start_time
                times.append(execution_time)

            avg_time = np.mean(times)
            performance_data[dim - start_dim, num_vars - start_vars] = avg_time
            print(f"Dimension: {dim}, Variables: {num_vars}, Avg. Time: {avg_time}")

    return performance_data

def plot_performance(performance_data, start_dim, end_dim, start_vars, end_vars):
    fig, ax = plt.subplots()
    cax = ax.imshow(performance_data, cmap="viridis", aspect="auto", origin="lower",
                    extent=[start_vars - 0.5, end_vars + 0.5, start_dim - 0.5, end_dim + 0.5])

    ax.set_xticks(range(start_vars, end_vars + 1))
    ax.set_yticks(range(start_dim, end_dim + 1))
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Dimension")
    ax.set_title("Quantum Quasigroup Solver Performance")

    cbar = fig.colorbar(cax)
    cbar.set_label("Average Execution Time (s)")

    plt.savefig("quantum_performance_heatmap.png")
    plt.show()


def get_label(row, col, digit, order):
    '''
    Get label for a cell in the matrix
    Args:   row: Row index
            col: Column index
            digit: Digit
            order: Order of the matrix
    Returns:  Label for the cell
    '''

    return "{row},{col}_{digit}".format(**locals())

def get_matrix(filename):
    '''
    Read Quasigroup problem as matrix
    Args:  filename: Quasigroup problem file
    Returns:  Quasigroup problem as matrix
    '''

    with open(filename, "r") as f:
        content = f.readlines()

    lines = []
    for line in content:
        new_line = line.rstrip()

        if new_line:
            new_line = list(map(int, new_line.split(' ')))
            lines.append(new_line)

    return lines

def is_correct(matrix):
    '''
    Verify if the solution is correct
    Args:  matrix: Quasigroup problem as matrix
    Returns:  True if the solution is correct, False otherwise
    '''

    order = len(matrix)
    digits = set(range(1, order + 1))

    for row in matrix:
        if set(row) != digits:
            print("Error in row: ", row)
            return False

    for col_idx in range(order):
        col = [matrix[row_idx][col_idx] for row_idx in range(order)]
        if set(col) != digits:
            print("Error in col: ", col)
            return False

    return True

def build_bqm(matrix):
    '''
    Build BQM for Quasigroup problem
    Args:  matrix: Quasigroup problem as matrix
    Returns:  Binary quadratic model
    '''

    order = len(matrix)
    digits = range(1, order + 1)

    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

    # Constraint: Each cell can only select one digit
    for row in range(order):
        for col in range(order):
            node_digits = [get_label(row, col, digit, order) for digit in digits]
            one_digit_bqm = combinations(node_digits, 1)
            bqm.update(one_digit_bqm)

    # Constraint: Each row of cells cannot have duplicate digits
    for row in range(order):
        for digit in digits:
            row_nodes = [get_label(row, col, digit, order) for col in range(order)]
            row_bqm = combinations(row_nodes, 1)
            bqm.update(row_bqm)

    # Constraint: Each column of cells cannot have duplicate digits
    for col in range(order):
        for digit in digits:
            col_nodes = [get_label(row, col, digit, order) for row in range(order)]
            col_bqm = combinations(col_nodes, 1)
            bqm.update(col_bqm)

    # Constraint: Fix known values
    for row, line in enumerate(matrix):
        for col, value in enumerate(line):
            if value > 0:
                bqm.fix_variable(get_label(row, col, value, order), 1)

    return bqm

def run_test(filename):
    matrix = get_matrix(filename)
    bqm = build_bqm(matrix)
    start_time = time.time()
    result = solve_quasigroup(bqm, matrix)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution time:", execution_time)

    if result:
        for line in result:
            print(*line, sep=" ")
        if is_correct(result):
            print("The solution is correct")
        else:
            print("The solution is incorrect")
    else:
        print("No solution found")

def solve_quasigroup(bqm, matrix):
    '''
    Solve Quasigroup problem using KerberosSampler
    Args:  bqm: Binary quadratic model
           matrix: Quasigroup problem as matrix
    Returns:  Solution matrix
    '''

    # Solve BQM
    solution = KerberosSampler().sample(bqm,
                                        max_iter=10,
                                        convergence=3,
                                        qpu_params={'label': 'Example - Quasigroup'})
    best_solution = solution.first.sample
    solution_list = [k for k, v in best_solution.items() if v == 1]

    result = copy.deepcopy(matrix)

    # Update matrix with solution
    for label in solution_list:
        coord, digit = label.split('_')
        row, col = map(int, coord.split(','))

        if result[row][col] > 0:
            continue

        result[row][col] = int(digit)

    return result

def main(args):
    if args.test:
        test_folder = "tests"
        for file in os.listdir(test_folder):
            if file.endswith(".txt"):
                print(f"Testing {file}:")
                run_test(os.path.join(test_folder, file))
                print()
    elif args.performance:
        start_dim, end_dim, start_vars, end_vars = args.performance
        performance_data = run_performance_tests(start_dim, end_dim, start_vars, end_vars)
        plot_performance(performance_data, start_dim, end_dim, start_vars, end_vars)
    else:
        run_test(args.file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Quasigroup with Holes using a quantum annealing approach with D-Wave")
    parser.add_argument("--file", type=str, default="example.txt", help="Path to the problem file")
    parser.add_argument("--test", action="store_true", help="Run tests from the 'tests' folder")
    parser.add_argument("--performance", nargs=4, metavar=("START_DIM", "END_DIM", "START_VARS", "END_VARS"), type=int, help="Run performance tests with specified dimensions and number of variables")

    args = parser.parse_args()
    main(args)