#ifndef SYMNMF_H
#define SYMNMF_H

#define SYMNMF_EPSILON 0.0001
#define SYMNMF_MAX_ITER 300
#define SYMNMF_BETA 0.5

double **allocate_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
double **sym(double **points, int n, int d);
double **ddg(double **points, int n, int d);
double **norm(double **points, int n, int d);
double **symnmf(double **h, double **w, int n, int k,
                double epsilon, int max_iter);

#endif
