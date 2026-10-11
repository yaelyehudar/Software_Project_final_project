import sys
import numpy as np
import symnmfmodule as symnmf

ERROR_MESSAGE = "An Error Has Occurred"


def read_points(file_name):
    """Read a comma-separated dataset as a two-dimensional array."""
    return np.loadtxt(file_name, delimiter=",", ndmin=2)


def format_matrix(matrix):
    """Return the matrix in the output format required by the project."""
    return "\n".join(",".join(f"{float(value):.4f}" for value in row) for row in matrix)


def initialize_h(w, k):
    """Initialize H exactly as specified in Section 1.4.1."""
    np.random.seed(1234)
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


def parse_k(value):
    """Accept integer-valued numeric input, including values such as 3.0."""
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError("illegal k")
    return int(number)


def main():
    """Run the command-line interface."""
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
