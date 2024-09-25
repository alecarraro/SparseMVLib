#ifndef UTILS_H
#define UTILS_H

typedef struct {
    int *row_ptr;   // Array to store row pointers
    int *col_ind;   // Array to store column indices
    double *values; // Array to store non-zero values
    int n_rows;   // Number of rows
    int n_cols;   // Number of columns
    int nnz;        // Number of non-zero elements
} CSRMatrix;

// Structure to hold matrix statistics
typedef struct {
    int total_nnz;
    double density;
    double avg_nnz_per_row;
    double variance_nnz;
} MatrixStats;

// Structure to hold execution times
typedef struct {
    double time_classic;
    double time_relaxed;
    double time_strict;
} SPMVExecutionTimes;

void read_mtx_to_csr(const char *filename, CSRMatrix *csr);
void print_entry(CSRMatrix *csr, int row, int col);
void free_csr_matrix(CSRMatrix *csr);
void log_stats(const CSRMatrix *csr, const char *filename, 
                                                const char *output_filename, 
                                                const double *x, double *y, int num_threads);
#endif