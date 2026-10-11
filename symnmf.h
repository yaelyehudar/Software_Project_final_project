#ifndef SYMNMF_H
#define SYMNMF_H

#define SYMNMF_EPSILON 0.0001
#define SYMNMF_MAX_ITER 300
#define SYMNMF_BETA 0.5

/* Input: positive rows, cols. Output: zero-initialized matrix or NULL. */
double **allocate_matrix(int rows, int cols);
/* Input: allocated matrix or NULL, row count. Output: frees its memory. */
void free_matrix(double **matrix, int rows);
/* Input: n points of dimension d. Output: n x n similarity matrix. */
double **sym(double **points, int n, int d);
/* Input: n points of dimension d. Output: diagonal degree matrix. */
double **ddg(double **points, int n, int d);
/* Input: n points of dimension d. Output: normalized similarity matrix. */
double **norm(double **points, int n, int d);
/* Input: initial h, normalized w, dimensions and stopping criteria.
   Output: optimized n x k h or NULL. */
double **symnmf(double **h, double **w, int n, int k,
                double epsilon, int max_iter);

#endif
