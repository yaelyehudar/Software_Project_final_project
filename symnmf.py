import sys
import numpy as np
import symnmfmodule as symnmf

np.random.seed(1234)

ERROR_MESSAGE = "An Error Has Occurred"


def read_points(file_name):
    """Read a comma-separated dataset as a two-dimensional array."""
    return np.loadtxt(file_name, delimiter=",", ndmin=2)


def format_matrix(matrix):
    """Return the matrix in the output format required by the project."""
    return "\n".join(",".join(f"{float(value):.4f}" for value in row) for row in matrix)


def initialize_h(w, k):
    """Initialize H exactly as specified in Section 1.4.1."""
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(len(w), k)).tolist()


def calculate_goal(points, goal, k):
    """Dispatch a requested goal to the C extension."""
    points_list = points.tolist()
    if goal == "sym":
        return symnmf.sym(points_list)
    if goal == "ddg":
        return symnmf.ddg(points_list)
    if goal == "norm":
        return symnmf.norm(points_list)
    if goal == "symnmf":
        w = symnmf.norm(points_list)
        h = initialize_h(w, k)
        return symnmf.symnmf(h, w)
    raise ValueError("illegal goal")


def main():
    """Run the command-line interface."""
    try:
        if len(sys.argv) != 4:
            raise ValueError("wrong number of arguments")
        k = int(sys.argv[1])
        goal = sys.argv[2]
        points = read_points(sys.argv[3])
        if k <= 1 or k >= len(points):
            raise ValueError("illegal k")
        result = calculate_goal(points, goal, k)
        print(format_matrix(result))
    except Exception:
        print(ERROR_MESSAGE)


if __name__ == "__main__":
    main()
