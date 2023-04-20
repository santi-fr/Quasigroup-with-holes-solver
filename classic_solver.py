import os
import argparse
from constraint import Problem, AllDifferentConstraint
import time
import matplotlib.pyplot as plt
import random

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

# Ejemplo de uso
dimension = 4
num_variables = 5
problem = generate_quasigroup_problem(dimension, num_variables)
for row in problem:
    print(row)


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

def plot_performance(performance_data, start_dim, end_dim, start_vars, end_vars):
    fig, ax = plt.subplots()
    cax = ax.imshow(performance_data, cmap="viridis", aspect="auto", origin="lower",
                    extent=[start_vars - 0.5, end_vars + 0.5, start_dim - 0.5, end_dim + 0.5])

    ax.set_xticks(range(start_vars, end_vars + 1))
    ax.set_yticks(range(start_dim, end_dim + 1))
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Dimension")
    ax.set_title("Quasigroup Solver Performance")

    cbar = fig.colorbar(cax)
    cbar.set_label("Average Execution Time (s)")

    plt.savefig("performance_heatmap.png")
    plt.show()


def run_performance_tests(start_dim, end_dim, start_vars, end_vars, num_tests=10):
    performance_data = []

    for dim in range(start_dim, end_dim + 1):
        dim_data = []
        for num_vars in range(start_vars, end_vars + 1):
            execution_times = []
            for _ in range(num_tests):
                problem = generate_quasigroup_problem(dim, num_vars)
                start_time = time.time()
                _ = quasigroup_solver(problem)
                end_time = time.time()

                execution_times.append(end_time - start_time)

            avg_execution_time = sum(execution_times) / num_tests
            dim_data.append(avg_execution_time)
            print(f"Dimension: {dim}, Variables: {num_vars}, Avg Time: {avg_execution_time}")

        performance_data.append(dim_data)

    plot_performance(performance_data, start_dim, end_dim, start_vars, end_vars)


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
        dim_start, dim_end, var_start, var_end = args.performance
        run_performance_tests(dim_start, dim_end, var_start, var_end)

    elif args.file:
        run_test(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()