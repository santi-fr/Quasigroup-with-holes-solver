import os
import argparse
import copy
import dimod
import time
import quasi_library as ql
import numpy as np
from dimod.generators.constraints import combinations
from hybrid.reference import KerberosSampler

def run_performance_tests(start_dim, end_dim, start_fill_ratio, end_fill_ratio, num_tests=10):
    '''
    Run performance tests for a range of dimensions and fill ratios
    Args:  start_dim: Start dimension
            end_dim: End dimension
            start_fill_ratio: Start fill ratio
            end_fill_ratio: End fill ratio
            num_tests: Number of tests to run
    Returns:  List of performance data
    '''
    performance_data = np.zeros((end_dim - start_dim + 1, int((end_fill_ratio - start_fill_ratio) / 0.1) + 2))

    for dim in range(start_dim, end_dim + 1):
        for fill_ratio_index, fill_ratio in enumerate(np.arange(start_fill_ratio, end_fill_ratio + 0.1, 0.1)):
            times = []
            for _ in range(num_tests):
                num_vars = int(dim * dim * fill_ratio)
                matrix = ql.generate_quasigroup_problem(dim, num_vars)
                start_time = time.time()
                bqm = build_bqm(matrix)
                _ = solve_quasigroup(bqm, matrix)
                end_time = time.time()

                execution_time = end_time - start_time
                times.append(execution_time)

            avg_time = np.mean(times)
            performance_data[dim - start_dim, fill_ratio_index] = avg_time
            print(f"Dimension: {dim}, Fill Ratio: {fill_ratio}, Avg. Time: {avg_time}")

    return performance_data

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
    matrix = ql.get_matrix(filename)
    start_time = time.time()
    bqm = build_bqm(matrix)
    result = solve_quasigroup(bqm, matrix)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution time:", execution_time)

    if result:
        for line in result:
            print(*line, sep=" ")
        if ql.is_correct(result):
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
        dim_start, dim_end = args.dims
        fill_ratio_start, fill_ratio_end = args.fill_ratios
        performance_data = run_performance_tests(dim_start, dim_end, fill_ratio_start, fill_ratio_end)
        ql.plot_performance(performance_data, dim_start, dim_end, fill_ratio_start, fill_ratio_end, "quantum_performance.png")
    else:
        run_test(args.file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Quasigroup with Holes using a quantum annealing approach with D-Wave")
    parser.add_argument("--file", type=str, default="example.txt", help="Path to the problem file")
    parser.add_argument("--test", action="store_true", help="Run tests from the 'tests' folder")
    parser.add_argument("--performance", action="store_true", help="Measure performance for a range of dimensions and fill ratios")
    parser.add_argument("--dims", nargs=2, type=int, metavar=("DIM_START", "DIM_END"), help="Range of dimensions to test")
    parser.add_argument("--fill_ratios", nargs=2, type=float, metavar=("FILL_RATIO_START", "FILL_RATIO_END"), help="Range of fill ratios to test")

    args = parser.parse_args()
    main(args)