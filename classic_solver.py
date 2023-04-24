import os
import argparse
from constraint import Problem, AllDifferentConstraint
import time
import numpy as np
import quasi_library as ql

def quasigroup_solver(matrix):
    '''
    Solve Quasigroup problem
    Args:  matrix: Quasigroup problem as matrix
    Returns:  Solution to the problem
    '''
    n = len(matrix)
    problem = Problem()
    
    # Definir variables y dominios
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 0:
                problem.addVariable((i, j), range(1, n + 1))
            else:
                problem.addVariable((i, j), [matrix[i][j]])
    
    # Restricciones de filas
    for i in range(n):
        problem.addConstraint(AllDifferentConstraint(), [(i, j) for j in range(n)])
    
    # Restricciones de columnas
    for j in range(n):
        problem.addConstraint(AllDifferentConstraint(), [(i, j) for i in range(n)])
    
    # Encontrar soluciones
    solutions = problem.getSolutions()

    if solutions:
        # Llenar la matriz con la solución
        for key, value in solutions[0].items():
            i, j = key
            matrix[i][j] = value
        return matrix
    else:
        return None

def run_test(filename):
    '''
    Run test with a given file
    Args:  filename: Name of the file
    '''
    matrix = ql.get_matrix(filename)
    start_time = time.time()
    result = quasigroup_solver(matrix)
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

def test_quasigroup_solver(quasigroup_solver, dim, num_vars, num_tests=100):
    '''
    Test the performance of a quasigroup solver
    Args:  quasigroup_solver: Function that solves a quasigroup problem
           dim: Dimension of the quasigroup
           num_vars: Number of variables in the problem
           num_tests: Number of tests to run
    Returns:  List of execution times
    '''
    times = []

    for _ in range(num_tests):
        problem = ql.generate_quasigroup_problem(dim, num_vars)
        start_time = time.time()
        _ = quasigroup_solver(problem)
        end_time = time.time()
        times.append(end_time - start_time)

    return times


def run_performance_tests(quasigroup_solver, start_dim, end_dim, start_fill_ratio, end_fill_ratio, steps=5):
    '''
    Run performance tests for a range of dimensions and fill ratios
    Args:  quasigroup_solver: Function that solves a quasigroup problem
              start_dim: Start dimension
              end_dim: End dimension
              start_fill_ratio: Start fill ratio
              end_fill_ratio: End fill ratio
              steps: Number of steps
    Returns:  List of performance data
    '''
    fill_ratios = np.linspace(start_fill_ratio, end_fill_ratio, steps)
    performance_data = []

    for dim in range(start_dim, end_dim + 1):
        dim_performance = []
        for fill_ratio in fill_ratios:
            num_vars = int(dim * dim * fill_ratio)
            times = test_quasigroup_solver(quasigroup_solver, dim, num_vars, num_tests=10)
            dim_performance.append(np.mean(times))
        performance_data.append(dim_performance)

    return performance_data


def main():
    parser = argparse.ArgumentParser(description="Quasigroup With Holes solver")
    parser.add_argument("--file", type=str, help="Quasigroup problem file")
    parser.add_argument("--test", action="store_true", help="Test the solver with example problems")
    parser.add_argument("--performance", action="store_true", help="Measure performance for a range of dimensions and fill ratios")
    parser.add_argument("--dims", nargs=2, type=int, metavar=("DIM_START", "DIM_END"), help="Range of dimensions to test")
    parser.add_argument("--fill_ratios", nargs=2, type=float, metavar=("FILL_RATIO_START", "FILL_RATIO_END"), help="Range of fill ratios to test")

    args = parser.parse_args()

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
        performance_data = run_performance_tests(quasigroup_solver, dim_start, dim_end, fill_ratio_start, fill_ratio_end)
        print(performance_data)
        ql.plot_performance(performance_data, dim_start, dim_end, fill_ratio_start, fill_ratio_end, "classic_performance.png")

    elif args.file:
        run_test(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()