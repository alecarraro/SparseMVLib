/******************************************************************************
* Parallel Sparse Matrix-Vector Multiplication
* FILE: utils.c
* DESCRIPTION:
*   This file implements the necessary utility functions for handling
*   sparse matrices in CSR (Compressed Sparse Row) format tests. These functions include:
*   
*   1. **read_mtx_to_csr**: 
*      Converts a matrix from a Matrix Market (.mtx) file format to CSR format
*   
*   2. **free_csr_matrix**: 
*      Frees the memory allocated .
*   
*   3. **calculate_matrix_stats**: 
*      Calculates various statistics of the CSRMatrix, including total non-zero 
*      elements (NNZ), density, average NNZ per row, and variance of NNZ per row.
*      Returns these statistics in a MatrixStats structure.
*   
*   4. **benchmark_spmv_methods**: 
*      Measures the execution time of different sparse for each algorithm variant
*      and returns the results in an SPMVExecutionTimes structure.
*   
*   5. **write_stats_to_file**: 
*      Writes the calculated matrix statistics and execution times to a specified 
*      output file, facilitating easy logging and analysis of results.
*   
*   6. **log_stats**: 
*      Combines the functionality of calculating statistics and benchmarking SPMV 
*      methods, and then logs the results to a file for a given CSRMatrix.
*   
*   7. **print_entry**: 
*      Prints the value of a specific entry in the CSRMatrix
*   
******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h>
#include <string.h>
#include <omp.h>
#include "sparse_matrix_multiplication.h"

void read_mtx_to_csr(const char *filename, CSRMatrix *csr) {

    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        printf("Unable to open file!\n");
        exit(EXIT_FAILURE);
    }

    // Read the matrix market header & extract useful data
    char line[512];
    do {
        fgets(line, sizeof(line), f);
    } while (line[0] == '%'); // Skip comments

    int M, N, NNZ;
    sscanf(line, "%d %d %d", &M, &N, &NNZ);

    csr->n_rows = M;
    csr->n_cols = N;
    csr->nnz = NNZ;

    // Allocate memory
    csr->row_ptr = (int *)malloc((M + 1) * sizeof(int));
    csr->col_ind = (int *)malloc(NNZ * sizeof(int));
    csr->values = (double *)malloc(NNZ * sizeof(double));

    if (!csr->row_ptr || !csr->col_ind || !csr->values) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Tempo arrays to count the number of NNZ per row
    int *row_counts = (int *)calloc(M, sizeof(int));

    // Pass 1: count the number of non-zeros per row
    int row, col;
    double value;
    for (int i = 0; i < NNZ; i++) {
        fscanf(f, "%d %d %lf", &row, &col, &value);
        row_counts[row - 1]++;
    }

    // Build the row_ptr array
    csr->row_ptr[0] = 0;
    for (int i = 1; i <= M; i++) {
        csr->row_ptr[i] = csr->row_ptr[i - 1] + row_counts[i - 1];
    }

    // Pass 2: fill the col_ind and values arrays
    rewind(f);
    do {
        fgets(line, sizeof(line), f);
    } while (line[0] == '%'); // Skip comments again

    // Arrays to track the insertion position for each row
    int *current_pos = (int *)calloc(M, sizeof(int));
    if (!current_pos) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M; i++) {
        current_pos[i] = csr->row_ptr[i];
    }

    for (int i = 0; i < NNZ; i++) {
        fscanf(f, "%d %d %lf", &row, &col, &value);
        int idx = current_pos[row - 1]++;
        csr->col_ind[idx] = col - 1;
        csr->values[idx] = value;
    }

    // Free temp
    free(row_counts);
    free(current_pos);
    fclose(f);
}

void free_csr_matrix(CSRMatrix *csr){
    free(csr->row_ptr);
    free(csr->col_ind);
    free(csr->values);
}

MatrixStats calculate_matrix_stats(const CSRMatrix *csr) {
    MatrixStats stats;

    // Calculate basic stats
    stats.total_nnz = csr->nnz;
    stats.density = (double)stats.total_nnz / (csr->n_rows * csr->n_cols);
    stats.avg_nnz_per_row = (double)stats.total_nnz / csr->n_rows;

    // Calculate variance of NNZ elements per row
    stats.variance_nnz = 0.0;
    for (int i = 0; i < csr->n_rows; i++) {
        int nnz_in_row = csr->row_ptr[i + 1] - csr->row_ptr[i];
        stats.variance_nnz += (nnz_in_row - stats.avg_nnz_per_row) * (nnz_in_row - stats.avg_nnz_per_row);
    }
    stats.variance_nnz /= csr->n_rows;

    return stats;
}

SPMVExecutionTimes benchmark_spmv_methods(const CSRMatrix *csr, const double *x, int num_threads) {
    SPMVExecutionTimes times;
    double y_classic[csr->n_rows], y_relaxed[csr->n_rows], y_strict[csr->n_rows];

    // Initialize output vectors to zero for testing purpose
    memset(y_classic, 0, csr->n_rows * sizeof(double));
    memset(y_relaxed, 0, csr->n_rows * sizeof(double));
    memset(y_strict, 0, csr->n_rows * sizeof(double));

    double start, end;

    start = omp_get_wtime();
    spmv_classic(csr, x, y_classic, num_threads);
    end = omp_get_wtime();
    times.time_classic = end - start;

    start = omp_get_wtime();
    spmv_relaxed(csr, x, y_relaxed, num_threads);
    end = omp_get_wtime();
    times.time_relaxed = end - start;

    start = omp_get_wtime();
    spmv_strict(csr, x, y_strict, num_threads);
    end = omp_get_wtime();
    times.time_strict = end - start;

    return times;
}

void write_stats_to_file(const char *filename, const char *output_filename, const MatrixStats *stats, const SPMVExecutionTimes *times) {
    FILE *output_file = fopen(output_filename, "a");
    if (output_file == NULL) {
        printf("Unable to open output file!\n");
        return;
    }

    // Write the header line
    fseek(output_file, 0, SEEK_END);
    if (ftell(output_file) == 0) {
        fprintf(output_file, "Matrix Name,Total NNZ,Density,Avg NNZ per Row,Variance NNZ,Time Classic,Time Relaxed,Time Strict\n");
    }

    fprintf(output_file, "%s,%d,%f,%f,%f,%f,%f,%f\n", filename, stats->total_nnz, stats->density,
            stats->avg_nnz_per_row, stats->variance_nnz, 
            times->time_classic, times->time_relaxed, times->time_strict);

    fclose(output_file);
    printf("Matrix statistics and benchmark results logged to %s\n", output_filename);
}

// Combined function to collect stats & save
void log_stats(const CSRMatrix *csr, const char *filename, const char *output_filename, 
               const double *x, double *y, int num_threads) {

    MatrixStats stats = calculate_matrix_stats(csr);

    SPMVExecutionTimes times = benchmark_spmv_methods(csr, x, num_threads);

    write_stats_to_file(filename, output_filename, &stats, &times);
}

void print_entry(CSRMatrix *csr, int row, int col) {

    if (row >= csr->n_rows || col >= csr->n_cols) {
        printf("Invalid entry (%d, %d): Out of bounds.\n", row, col);
        return;
    }
    int start = csr->row_ptr[row];
    int end = csr->row_ptr[row + 1];

    for (int i = start; i < end; i++) {
        if (csr->col_ind[i] == col) {
            printf("Entry at (%d, %d) = %lf\n", row, col, csr->values[i]);
            return;
        }
    }

    printf("Entry at (%d, %d) = 0 (not stored in CSR)\n", row, col);
}


