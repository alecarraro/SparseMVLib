#ifndef SPARSE_MATRIX_MULTIPLICATION_H
#define SPARSE_MATRIX_MULTIPLICATION_H

#include "utils.h"

void spmv_classic(const CSRMatrix *matrix, const double *x, double *y, int num_threads);
void spmv_relaxed(const CSRMatrix *matrix, const double *x, double *y, int num_threads);
void spmv_strict(const CSRMatrix *matrix, const double *x, double *y, int num_threads);
#endif