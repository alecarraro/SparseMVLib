/******************************************************************************
* Parallel Sparse Matrix-Vector Multiplication
* FILE: sparse_matrix_multiplication.c
* DESCRIPTION:
*   This file implements three different approaches for performing sparse matrix-vector
*   multiplication y = y + Ax in parallel, using the CSR format.*
*   The three partitioning strategies implemented are:
*   
*   1. **Standard Row Partitioning (`spmv_classic`)**: 
*      In this method, each thread is assigned a fixed number of rows of the matrix. 
*
*   2. **Relaxed Row Partitioning (`spmv_relaxed`)**:
*      This method assigns rows to threads, but the partitioning is based on the number 
*      of non-zero values (NNZ) rather than rows.
*   
*   3. **Strict NNZ Partitioning (`spmv_strict`)**: 
*      The most granular partitioning approach, which assigns each thread a specific number 
*      of NNZ elements (NNZ) to process, regardless of the rows they belong to. Threads 
*      are assigned to work on specific segments of the non-zero elements, ensuring load balancing 
*      based on actual NNZ operations.
*   
*   Each function measures and prints the execution time for its respective method.
******************************************************************************/

#include <omp.h>
#include <stdlib.h>
#include "utils.h"
#include <stdio.h>

void spmv_classic(const CSRMatrix *matrix, const double *x, double *y, int num_threads) {
    /*
    * Arguments:
    *   matrix      - CSRMatrix struct -> the sparse matrix in CSR format.
    *   x           - Input vector x.
    *   y           - Output vector y to accumulate the results.
    *   num_threads - The number of threads to be used.
    */
    int i, j;
    double start, end;
    
    start = omp_get_wtime();

    // Parallel loop over rows, assigning equal chunks of rows to each thread
    #pragma omp parallel for num_threads(num_threads)
    for (i = 0; i < matrix->n_rows; i++) {
        double sum = 0.0;

        // Multiply row vector with the input vector and accumulate the result
        for (j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
            sum += matrix->values[j] * x[matrix->col_ind[j]];
        }
        y[i] += sum;
    }
    end= omp_get_wtime();
    printf("Classic row partitioning execution time: %f s \n", end-start);
}

// Relaxed partitioning
void spmv_relaxed(const CSRMatrix *matrix, const double *x, double *y, int num_threads) {
    /*
    * Arguments:
    *   matrix      - CSRMatrix struct -> the sparse matrix in CSR format.
    *   x           - Input vector x.
    *   y           - Output vector y to accumulate the results.
    *   num_threads - The number of threads to be used.
    */
    
    double start, end;

    start = omp_get_wtime();

    // Partition rows such that each thread gets roughly the same number of NNZ 
    int *row_partition = (int*)malloc((num_threads + 1) * sizeof(int));
    row_partition[0] = 0;

    int total_nnz = matrix->nnz;
    int nnz_per_thread = total_nnz / num_threads;
    int current_nnz = 0, thread_id = 1;

    // Assign rows to threads based on accumulated NNZ counts
    for (int i = 0; i < matrix->n_rows; i++) {
        current_nnz += matrix->row_ptr[i + 1] - matrix->row_ptr[i];
        if (current_nnz >= nnz_per_thread && thread_id < num_threads) {
            row_partition[thread_id] = i + 1;
            current_nnz = 0;
            thread_id++;
        }
    }
    row_partition[num_threads] = matrix->n_rows; // Last thread goes up to the last row

    // Parallel execution using relaxed row partitioning
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Each thread processes the rows assigned to it based on row_partition
        for (int i = row_partition[tid]; i < row_partition[tid + 1]; i++) {
            double sum = 0.0;

            // Vectorize to compute the sum for the current row
            #pragma omp simd reduction(+:sum)
            for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; j++) {
                sum += matrix->values[j] * x[matrix->col_ind[j]];
            }
            y[i] += sum;
        }
    }

    // Free partioning
    free(row_partition);
    end= omp_get_wtime();
    printf("Relaxed row partitioning execution time: %f s \n", end-start);
}

// Strict partitioning
void spmv_strict(const CSRMatrix *matrix, const double *x, double *y, int num_threads) {
    /*
    * Arguments:
    *   matrix      - CSRMatrix struct -> the sparse matrix in CSR format.
    *   x           - Input vector x.
    *   y           - Output vector y to accumulate the results.
    *   num_threads - The number of threads to be used.
    */
    
    double start, end;
    start = omp_get_wtime();
    
    int total_nnz = matrix->nnz;
    int nnz_per_thread = total_nnz / num_threads;

    // Parallel region with strict non-zero partitioning
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int start_nnz = tid * nnz_per_thread;
        int end_nnz = (tid == num_threads - 1) ? total_nnz : (tid + 1) * nnz_per_thread;

        int current_nnz = 0;
        int row_start = 0;

        // Find the starting row based on the first NNZ 
        for (int i = 0; i < matrix->n_rows; i++) {
            int nnz_in_row = matrix->row_ptr[i + 1] - matrix->row_ptr[i];
            if (current_nnz + nnz_in_row > start_nnz) {
                row_start = i;
                break;
            }
            current_nnz += nnz_in_row;
        }

        int remaining_nnz = end_nnz - start_nnz;
        int row = row_start;
        int col_start = start_nnz - current_nnz;

        // Process the NNZ even in multiple rows
        while (remaining_nnz > 0 && row < matrix->n_rows) {
            int nnz_in_row = matrix->row_ptr[row + 1] - matrix->row_ptr[row];
            int col_end = (remaining_nnz < nnz_in_row - col_start) ? col_start + remaining_nnz : nnz_in_row;

            double sum = 0.0;

            // Vectorize to compute the sum for the current row
            #pragma omp simd reduction(+:sum)
            for (int j = matrix->row_ptr[row] + col_start; j < matrix->row_ptr[row] + col_end; j++) {
                sum += matrix->values[j] * x[matrix->col_ind[j]];
            }

            // Atomic update to avoid race conditions
            #pragma omp atomic
            y[row] += sum;

            remaining_nnz -= (col_end - col_start);
            row++;
            col_start = 0;
        }
    }

    end= omp_get_wtime();
    printf("Strict row partitioning execution time: %f s \n", end-start);
}
