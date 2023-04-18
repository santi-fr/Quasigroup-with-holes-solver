import os
import argparse
import copy
import dimod
import time
from dimod.generators.constraints import combinations
from hybrid.reference import KerberosSampler

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

def main(filename):
    # Read Quasigroup problem as matrix
    matrix = get_matrix(filename)

    # Solve BQM and update matrix
    bqm = build_bqm(matrix)

    start_time = time.time()  # Start time measurement
    result = solve_quasigroup(bqm, matrix)
    end_time = time.time()  # End time measurement

    elapsed_time = end_time - start_time
    print("Quantum annealing took {:.2f} seconds to solve the problem.".format(elapsed_time))

    # Print solution
    print("Solution for", filename)
    for line in result:
        print(*line, sep=" ")   # Print list without commas or brackets

    # Verify
    if is_correct(result):
        print("The solution is correct\n")
    else:
        print("The solution is incorrect\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quasigroup With Holes problem solver")
    parser.add_argument("filename", nargs="?", default="example.txt", help="Quasigroup problem file")
    parser.add_argument("--test", action="store_true", help="Run tests from the 'tests' folder")

    args = parser.parse_args()

    if args.test:
        test_folder = "tests"
        if not os.path.exists(test_folder):
            print("Error: 'tests' folder not found.")
        else:
            for file in os.listdir(test_folder):
                if file.endswith(".txt"):
                    filepath = os.path.join(test_folder, file)
                    main(filepath)
    else:
        main(args.filename)