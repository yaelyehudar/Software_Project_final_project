#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Read matrix dimensions from a Python sequence; return success flag. */
static int sequence_shape(PyObject *obj, int *rows, int *cols)
{
    PyObject *outer;
    PyObject *first;
    outer = PySequence_Fast(obj, "expected a matrix");
    if (outer == NULL || PySequence_Fast_GET_SIZE(outer) == 0) {
        Py_XDECREF(outer);
        return 0;
    }
    first = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, 0), "expected a row");
    if (first == NULL || PySequence_Fast_GET_SIZE(first) == 0) {
        Py_XDECREF(first);
        Py_DECREF(outer);
        return 0;
    }
    *rows = (int)PySequence_Fast_GET_SIZE(outer);
    *cols = (int)PySequence_Fast_GET_SIZE(first);
    Py_DECREF(first);
    Py_DECREF(outer);
    return 1;
}

/* Convert Python matrix entries into preallocated C matrix. */
static int fill_c_matrix(PyObject *obj, double **matrix, int rows, int cols)
{
    PyObject *outer;
    PyObject *row;
    PyObject *item;
    int i;
    int j;
    outer = PySequence_Fast(obj, "expected a matrix");
    if (outer == NULL || PySequence_Fast_GET_SIZE(outer) != rows) {
        Py_XDECREF(outer);
        return 0;
    }
    for (i = 0; i < rows; ++i) {
        row = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, i), "expected a row");
        if (row == NULL || PySequence_Fast_GET_SIZE(row) != cols) {
            Py_XDECREF(row);
            Py_DECREF(outer);
            return 0;
        }
        for (j = 0; j < cols; ++j) {
            item = PySequence_Fast_GET_ITEM(row, j);
            matrix[i][j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(row);
                Py_DECREF(outer);
                return 0;
            }
        }
        Py_DECREF(row);
    }
    Py_DECREF(outer);
    return 1;
}

/* Allocate C matrix and copy Python values; return NULL on failure. */
static double **py_to_c_matrix(PyObject *obj, int *rows, int *cols)
{
    double **matrix;
    if (!sequence_shape(obj, rows, cols)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid matrix shape");
        }
        return NULL;
    }
    matrix = allocate_matrix(*rows, *cols);
    if (matrix == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    if (!fill_c_matrix(obj, matrix, *rows, *cols)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid matrix rows");
        }
        free_matrix(matrix, *rows);
        return NULL;
    }
    return matrix;
}

/* Convert a C matrix to a nested Python list. */
static PyObject *c_to_py_matrix(double **matrix, int rows, int cols)
{
    PyObject *outer;
    PyObject *row;
    PyObject *value;
    int i;
    int j;
    outer = PyList_New(rows);
    if (outer == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; ++i) {
        row = PyList_New(cols);
        if (row == NULL) {
            Py_DECREF(outer);
            return NULL;
        }
        for (j = 0; j < cols; ++j) {
            value = PyFloat_FromDouble(matrix[i][j]);
            if (value == NULL) {
                Py_DECREF(row);
                Py_DECREF(outer);
                return NULL;
            }
            PyList_SET_ITEM(row, j, value);
        }
        PyList_SET_ITEM(outer, i, row);
    }
    return outer;
}

/* Parse points and return a Python matrix for the requested goal. */
static PyObject *matrix_goal(PyObject *args, int goal)
{
    PyObject *points_obj;
    PyObject *result_obj;
    double **points;
    double **result;
    int n;
    int d;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) {
        return NULL;
    }
    points = py_to_c_matrix(points_obj, &n, &d);
    if (points == NULL) {
        return NULL;
    }
    result = goal == 0 ? sym(points, n, d) :
             (goal == 1 ? ddg(points, n, d) : norm(points, n, d));
    free_matrix(points, n);
    if (result == NULL) {
        return PyErr_NoMemory();
    }
    result_obj = c_to_py_matrix(result, n, n);
    free_matrix(result, n);
    return result_obj;
}

/* Python wrapper for similarity matrix; args contain points. */
static PyObject *py_sym(PyObject *self, PyObject *args)
{
    (void)self;
    return matrix_goal(args, 0);
}

/* Python wrapper for degree matrix; args contain points. */
static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    (void)self;
    return matrix_goal(args, 1);
}

/* Python wrapper for normalized matrix; args contain points. */
static PyObject *py_norm(PyObject *self, PyObject *args)
{
    (void)self;
    return matrix_goal(args, 2);
}

/* Return whether H and W dimensions are compatible. */
static int valid_symnmf_shapes(int hn, int hk, int wn, int wk)
{
    return hn == wn && wn == wk && hk > 0;
}

/* Run SymNMF from Python-supplied H and W matrices. */
static PyObject *py_symnmf(PyObject *self, PyObject *args)
{
    PyObject *h_obj;
    PyObject *w_obj;
    PyObject *result_obj;
    double **h;
    double **w;
    double **result;
    int hn, hk, wn, wk;
    hn = hk = wn = wk = 0;
    h = NULL;
    w = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &h_obj, &w_obj)) {
        return NULL;
    }
    h = py_to_c_matrix(h_obj, &hn, &hk);
    w = py_to_c_matrix(w_obj, &wn, &wk);
    if (h == NULL || w == NULL || !valid_symnmf_shapes(hn, hk, wn, wk)) {
        free_matrix(h, hn);
        free_matrix(w, wn);
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "invalid matrix dimensions");
        }
        return NULL;
    }
    result = symnmf(h, w, hn, hk, SYMNMF_EPSILON, SYMNMF_MAX_ITER);
    free_matrix(h, hn);
    free_matrix(w, wn);
    if (result == NULL) {
        return PyErr_NoMemory();
    }
    result_obj = c_to_py_matrix(result, hn, hk);
    free_matrix(result, hn);
    return result_obj;
}

static PyMethodDef SymNMFMethods[] = {
    {"symnmf", py_symnmf, METH_VARARGS, "Optimize H for SymNMF."},
    {"sym", py_sym, METH_VARARGS, "Calculate the similarity matrix."},
    {"ddg", py_ddg, METH_VARARGS, "Calculate the diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "Calculate normalized similarity."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymNMFMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Initialize and return the extension module. */
PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}
