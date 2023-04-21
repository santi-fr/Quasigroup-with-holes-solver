import os
import argparse
from constraint import Problem, AllDifferentConstraint
import time
import quasi_library

def quasigroup_solver(matrix):
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
    matrix = get_matrix(filename)
    start_time = time.time()
    result = quasigroup_solver(matrix)
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


def run_performance_tests(solver_function, start_dim, end_dim, start_fill_ratio, end_fill_ratio, num_tests=10):
    fill_ratios = np.linspace(start_fill_ratio, end_fill_ratio, 5)
    performance_data = []

    for dim in range(start_dim, end_dim + 1):
        dim_data = []
        for fill_ratio in fill_ratios:
            execution_times = []
            for _ in range(num_tests):
                problem = generate_quasigroup_problem(dim, fill_ratio)
                start_time = time.time()
                _ = solver_function(problem)
                end_time = time.time()

                execution_times.append(end_time - start_time)

            avg_execution_time = sum(execution_times) / num_tests
            dim_data.append(avg_execution_time)
            print(f"Dimension: {dim}, Fill Ratio: {fill_ratio}, Avg Time: {avg_execution_time}")

        performance_data.append(dim_data)

    plot_performance(performance_data, start_dim, end_dim, start_fill_ratio, end_fill_ratio, "classic_performance.png")


def main():
    parser = argparse.ArgumentParser(description="Quasigroup With Holes solver")
    parser.add_argument("--file", type=str, help="Quasigroup problem file")
    parser.add_argument("--test", action="store_true", help="Test the solver with example problems")
    parser.add_argument("--performance", nargs=4, type=int, metavar=("DIM_START", "DIM_END", "VAR_START", "VAR_END"), help="Measure performance for a range of dimensions and variables")
    args = parser.parse_args()

    if args.test:
        test_folder = "tests"
        for file in os.listdir(test_folder):
            if file.endswith(".txt"):
                print(f"Testing {file}:")
                run_test(os.path.join(test_folder, file))
                print()
    elif args.performance:
        dim_start, dim_end, fill_ratio_start, fill_ratio_end = args.performance
        run_performance_tests(quasigroup_solver, dim_start, dim_end, fill_ratio_start, fill_ratio_end)


    elif args.file:
        run_test(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()