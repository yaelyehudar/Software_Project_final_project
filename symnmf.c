#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

#define INITIAL_CAPACITY 16
#define ERROR_MESSAGE "An Error Has Occurred"

typedef struct PointData {
    double **points;
    int rows;
    int cols;
} PointData;

static void print_error(void)
{
    printf("%s\n", ERROR_MESSAGE);
}

double **allocate_matrix(int rows, int cols)
{
    double **matrix;
    double *data;
    int i;
    matrix = (double **)malloc((size_t)rows * sizeof(double *));
    data = (double *)calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (matrix == NULL || data == NULL) {
        free(matrix);
        free(data);
        return NULL;
    }
    for (i = 0; i < rows; ++i) {
        matrix[i] = data + ((size_t)i * (size_t)cols);
    }
    return matrix;
}

void free_matrix(double **matrix, int rows)
{
    (void)rows;
    if (matrix != NULL) {
        free(matrix[0]);
        free(matrix);
    }
}

static double squared_distance(const double *a, const double *b, int d)
{
    double sum;
    double diff;
    int i;
    sum = 0.0;
    for (i = 0; i < d; ++i) {
        diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

double **sym(double **points, int n, int d)
{
    double **a;
    double value;
    int i;
    int j;
    a = allocate_matrix(n, n);
    if (a == NULL) {
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            value = exp(-0.5 * squared_distance(points[i], points[j], d));
            a[i][j] = value;
            a[j][i] = value;
        }
    }
    return a;
}

static double *degrees_from_similarity(double **a, int n)
{
    double *degrees;
    int i;
    int j;
    degrees = (double *)calloc((size_t)n, sizeof(double));
    if (degrees == NULL) {
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            degrees[i] += a[i][j];
        }
    }
    return degrees;
}

double **ddg(double **points, int n, int d)
{
    double **a;
    double **degree_matrix;
    double *degrees;
    int i;
    a = sym(points, n, d);
    if (a == NULL) {
        return NULL;
    }
    degrees = degrees_from_similarity(a, n);
    free_matrix(a, n);
    if (degrees == NULL) {
        return NULL;
    }
    degree_matrix = allocate_matrix(n, n);
    if (degree_matrix != NULL) {
        for (i = 0; i < n; ++i) {
            degree_matrix[i][i] = degrees[i];
        }
    }
    free(degrees);
    return degree_matrix;
}

static int fill_normalized(double **w, double **a, double *degrees, int n)
{
    double denominator;
    int i;
    int j;
    for (i = 0; i < n; ++i) {
        if (degrees[i] <= 0.0) {
            return 0;
        }
        for (j = i + 1; j < n; ++j) {
            denominator = sqrt(degrees[i] * degrees[j]);
            if (denominator <= 0.0) {
                return 0;
            }
            w[i][j] = a[i][j] / denominator;
            w[j][i] = w[i][j];
        }
    }
    return 1;
}

double **norm(double **points, int n, int d)
{
    double **a;
    double **w;
    double *degrees;
    a = sym(points, n, d);
    if (a == NULL) {
        return NULL;
    }
    degrees = degrees_from_similarity(a, n);
    w = allocate_matrix(n, n);
    if (degrees == NULL || w == NULL || !fill_normalized(w, a, degrees, n)) {
        free(degrees);
        free_matrix(a, n);
        free_matrix(w, n);
        return NULL;
    }
    free(degrees);
    free_matrix(a, n);
    return w;
}

static double **multiply(double **a, double **b, int r, int m, int c)
{
    double **result;
    int i;
    int j;
    int t;
    result = allocate_matrix(r, c);
    if (result == NULL) {
        return NULL;
    }
    for (i = 0; i < r; ++i) {
        for (t = 0; t < m; ++t) {
            for (j = 0; j < c; ++j) {
                result[i][j] += a[i][t] * b[t][j];
            }
        }
    }
    return result;
}

static double **transpose_multiply(double **h, int n, int k)
{
    double **result;
    int i;
    int j;
    int r;
    result = allocate_matrix(k, k);
    if (result == NULL) {
        return NULL;
    }
    for (r = 0; r < n; ++r) {
        for (i = 0; i < k; ++i) {
            for (j = 0; j < k; ++j) {
                result[i][j] += h[r][i] * h[r][j];
            }
        }
    }
    return result;
}

static double update_entries(double **h, double **next_h,
                             double **wh, double **hhth, int n, int k)
{
    double diff;
    double squared_change;
    double ratio;
    int i;
    int j;
    squared_change = 0.0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            if (hhth[i][j] <= 0.0) {
                return -1.0;
            }
            ratio = wh[i][j] / hhth[i][j];
            next_h[i][j] = h[i][j] * (1.0 - SYMNMF_BETA + SYMNMF_BETA * ratio);
            diff = next_h[i][j] - h[i][j];
            squared_change += diff * diff;
        }
    }
    return squared_change;
}

static double **single_update(double **h, double **w, int n, int k,
                              double *change)
{
    double **wh;
    double **hth;
    double **hhth;
    double **next_h;
    wh = multiply(w, h, n, n, k);
    hth = transpose_multiply(h, n, k);
    hhth = hth == NULL ? NULL : multiply(h, hth, n, k, k);
    next_h = allocate_matrix(n, k);
    if (wh == NULL || hth == NULL || hhth == NULL || next_h == NULL) {
        *change = -1.0;
    } else {
        *change = update_entries(h, next_h, wh, hhth, n, k);
    }
    free_matrix(wh, n);
    free_matrix(hth, k);
    free_matrix(hhth, n);
    if (*change < 0.0) {
        free_matrix(next_h, n);
        return NULL;
    }
    return next_h;
}

double **symnmf(double **h, double **w, int n, int k,
                double epsilon, int max_iter)
{
    double **current;
    double **next_h;
    double change;
    int iteration;
    current = allocate_matrix(n, k);
    if (current == NULL) {
        return NULL;
    }
    memcpy(current[0], h[0], (size_t)n * (size_t)k * sizeof(double));
    for (iteration = 0; iteration < max_iter; ++iteration) {
        next_h = single_update(current, w, n, k, &change);
        if (next_h == NULL) {
            free_matrix(current, n);
            return NULL;
        }
        free_matrix(current, n);
        current = next_h;
        if (change < epsilon) {
            break;
        }
    }
    return current;
}

static char *read_line(FILE *file)
{
    char *line;
    char *grown;
    int ch;
    int length;
    int capacity;
    capacity = 128;
    length = 0;
    line = (char *)malloc((size_t)capacity);
    if (line == NULL) {
        return NULL;
    }
    while ((ch = fgetc(file)) != EOF && ch != '\n') {
        if (length + 1 >= capacity) {
            capacity *= 2;
            grown = (char *)realloc(line, (size_t)capacity);
            if (grown == NULL) {
                free(line);
                return NULL;
            }
            line = grown;
        }
        line[length++] = (char)ch;
    }
    if (length == 0 && ch == EOF) {
        free(line);
        return NULL;
    }
    line[length] = '\0';
    return line;
}

static int count_columns(const char *line)
{
    int columns;
    const char *p;
    if (line[0] == '\0') {
        return 0;
    }
    columns = 1;
    for (p = line; *p != '\0'; ++p) {
        if (*p == ',') {
            ++columns;
        }
    }
    return columns;
}

static int parse_line(const char *line, double *row, int cols)
{
    char *end;
    const char *cursor;
    int j;
    cursor = line;
    for (j = 0; j < cols; ++j) {
        row[j] = strtod(cursor, &end);
        if (end == cursor) {
            return 0;
        }
        cursor = end;
        if (j < cols - 1) {
            if (*cursor != ',') {
                return 0;
            }
            ++cursor;
        }
    }
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r') {
        ++cursor;
    }
    return *cursor == '\0';
}

static int append_row(double ***rows, int *size, int *capacity, double *row)
{
    double **grown;
    if (*size == *capacity) {
        *capacity *= 2;
        grown = (double **)realloc(*rows, (size_t)(*capacity) * sizeof(double *));
        if (grown == NULL) {
            return 0;
        }
        *rows = grown;
    }
    (*rows)[*size] = row;
    ++(*size);
    return 1;
}

static void free_rows(double **rows, int count)
{
    int i;
    for (i = 0; i < count; ++i) {
        free(rows[i]);
    }
    free(rows);
}

static int read_rows(FILE *file, double ***rows_out, int *n_out, int *d_out)
{
    double **rows;
    double *row;
    char *line;
    int n;
    int d;
    int capacity;
    n = 0;
    d = 0;
    capacity = INITIAL_CAPACITY;
    rows = (double **)malloc((size_t)capacity * sizeof(double *));
    if (rows == NULL) {
        return 0;
    }
    while ((line = read_line(file)) != NULL) {
        if (d == 0) {
            d = count_columns(line);
        }
        row = (double *)malloc((size_t)d * sizeof(double));
        if (d == 0 || row == NULL || !parse_line(line, row, d) ||
            !append_row(&rows, &n, &capacity, row)) {
            free(line);
            free(row);
            free_rows(rows, n);
            return 0;
        }
        free(line);
    }
    *rows_out = rows;
    *n_out = n;
    *d_out = d;
    return n > 0;
}

static int copy_rows(PointData *data, double **rows)
{
    int i;
    data->points = allocate_matrix(data->rows, data->cols);
    if (data->points == NULL) {
        return 0;
    }
    for (i = 0; i < data->rows; ++i) {
        memcpy(data->points[i], rows[i], (size_t)data->cols * sizeof(double));
    }
    return 1;
}

static int load_points(const char *filename, PointData *data)
{
    FILE *file;
    double **rows;
    int ok;
    file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }
    rows = NULL;
    ok = read_rows(file, &rows, &data->rows, &data->cols);
    fclose(file);
    if (!ok) {
        return 0;
    }
    ok = copy_rows(data, rows);
    free_rows(rows, data->rows);
    return ok;
}

static void print_matrix(double **matrix, int rows, int cols)
{
    int i;
    int j;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

static double **run_goal(const char *goal, PointData *data)
{
    if (strcmp(goal, "sym") == 0) {
        return sym(data->points, data->rows, data->cols);
    }
    if (strcmp(goal, "ddg") == 0) {
        return ddg(data->points, data->rows, data->cols);
    }
    if (strcmp(goal, "norm") == 0) {
        return norm(data->points, data->rows, data->cols);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    PointData data;
    double **result;
    if (argc != 3) {
        print_error();
        return 1;
    }
    data.points = NULL;
    if (!load_points(argv[2], &data)) {
        print_error();
        return 1;
    }
    result = run_goal(argv[1], &data);
    if (result == NULL) {
        free_matrix(data.points, data.rows);
        print_error();
        return 1;
    }
    print_matrix(result, data.rows, data.rows);
    free_matrix(result, data.rows);
    free_matrix(data.points, data.rows);
    return 0;
}
