import sys
import numpy as np
import symnmfmodule

ERROR_MESSAGE = "An Error Has Occurred"


def read_points(file_name):
    """Read a comma-separated dataset as a two-dimensional array.

    Arguments: path for input file.
    Returns: n x d array
    """
    return np.loadtxt(file_name, delimiter=",", ndmin=2)


def format_matrix(matrix):
    """Return the matrix in the required output format.

    Arguments: 2D list of numbers.
    Returns: formatted string with commas between values.
    """
    return "\n".join(",".join(f"{float(value):.4f}" for value in row) for row in matrix)


def initialize_h(w, k):
    """Initialize H

    Arguments: W: normalized similarity matrix. k: cluster count.
    Returns:  n x k nested list. m is the mean of W. 
    """
    np.random.seed(1234)
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(len(w), k)).tolist()


def calculate_goal(points, goal, k):
    """Dispatch a requested goal to the C extension.

    Arguments: poins: data points, goal: name of the goal, k: cluster count.
    Returns: the matrix as a nested list. if unknown goal - raise error.
    """
    points_list = points.tolist()
    if goal == "sym":
        return symnmfmodule.sym(points_list)
    if goal == "ddg":
        return symnmfmodule.ddg(points_list)
    if goal == "norm":
        return symnmfmodule.norm(points_list)
    if goal == "symnmf":
        w = symnmfmodule.norm(points_list)
        return symnmfmodule.symnmf(initialize_h(w, k), w)
    raise ValueError("illegal goal")


def parse_k(value):
    """Parse the k input from user

    Arguments: k: from command line input.
    Returns: int k if ok. if wrong - raise error.
    """
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError("illegal k")
    return int(number)


def main():
    """user interface

    Gets user input: k, input file, goal
    Prints matrix or error message.
    """
    try:
        if len(sys.argv) != 4:
            raise ValueError("wrong number of arguments")
        try:
            k = parse_k(sys.argv[1])
        except (ValueError, OverflowError):
            print("Incorrect number of clusters!")
            return
        goal = sys.argv[2]
        points = read_points(sys.argv[3])
        if k <= 1 or k >= len(points):
            print("Incorrect number of clusters!")
            return
        result = calculate_goal(points, goal, k)
        print(format_matrix(result))
    except Exception:
        print(ERROR_MESSAGE)


if __name__ == "__main__":
    main()
