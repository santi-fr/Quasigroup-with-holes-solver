import random
import matplotlib.pyplot as plt
import numpy as np

def generate_quasigroup_problem(dimension, fill_ratio):
    # Genera un cuadrado latino completo
    latin_square = [[(i + j) % dimension + 1 for i in range(dimension)] for j in range(dimension)]

    # Revuelve el cuadrado latino
    random.shuffle(latin_square)
    for row in latin_square:
        random.shuffle(row)

    # Calcula el número de variables a eliminar en función del ratio de llenado
    num_variables_to_remove = int(dimension * dimension * (1 - fill_ratio))

    # Quita algunas celdas para crear el problema de quasigroup with holes
    while num_variables_to_remove > 0:
        i, j = random.randint(0, dimension - 1), random.randint(0, dimension - 1)
        if latin_square[i][j] != 0:
            latin_square[i][j] = 0
            num_variables_to_remove -= 1

    return latin_square


def plot_performance(performance_data, start_dim, end_dim, start_fill_ratio, end_fill_ratio, filename):
    fig, ax = plt.subplots()
    
    fill_ratios = np.linspace(start_fill_ratio, end_fill_ratio, len(performance_data[0]))
    
    for dim, dim_performance in enumerate(performance_data, start_dim):
        ax.plot(fill_ratios, dim_performance, label=f"Dimension {dim}")

    ax.set_xlabel("Fill Ratio")
    ax.set_ylabel("Average Execution Time (s)")
    ax.set_title("Quantum Quasigroup Solver Performance")
    ax.legend()

    plt.savefig(filename)
    plt.show()


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