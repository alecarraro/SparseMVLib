/******************************************************************************
* Parallel Sparse Matrix-Vector Multiplication 
* FILE: main.c
* DESCRIPTION:
*   This file implements the main loop to test the algorithms implementations using
*   the functions from the other files
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "sparse_matrix_multiplication.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <mtx_filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Create CSR matrix from the argument passed to the program
    CSRMatrix csr_matrix;
    read_mtx_to_csr(argv[1], &csr_matrix);

    // Allocate memory and initialize vectors
    double *x = (double *)malloc(csr_matrix.n_cols * sizeof(double));
    double *y = (double *)malloc(csr_matrix.n_rows * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < csr_matrix.n_cols; i++) {
        x[i] = ((double) rand() / (RAND_MAX)); // Random values in range [0, 1]
    }

    #pragma omp parallel for
    for (int i = 0; i < csr_matrix.n_rows; i++) {
        y[i] = 0.0; // Initialize result vector to zero
    }

    // Log statistics and benchmark results
    log_stats(&csr_matrix, argv[1], "output/stats.csv", x, y, 6);

    free_csr_matrix(&csr_matrix);
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
