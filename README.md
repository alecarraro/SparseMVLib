# Project Title: Parallel Matrix-Vector Multiplication

## Project Overview
In this project I wanted to benchmark the performance of different partioning schemes for parallel sparse matrix-vector multiplication (SPMV) using the the Compressed Sparse Row (CSR) format. The algorithms are written in C and accelerated using OpenMP directives. The three partitioning schemes implemented in this project are:
1. **Fixed Rows-Based Partitioning**  
   This method assigns an equal number of rows to each thread, although an equal number of rows does not always result in an equal workload due to variations in the number of non-zero elements per row.

2. **Relaxed NNZ-Based Partitioning**  
   This method attempts to balance the workload by ensuring that each thread processes a block of matrix rows containing approximately the same number of non-zero elements (NNZ), rather than assigning rows directly.

3. **Strict NNZ-Based Partitioning**  
   In this approach, each thread is assigned exactly the same number of non-zero elements, regardless of how those elements are distributed across rows.

An overview of these three techniques is displayed in the figure below (Gao et Al 2024)[1]:

![partioning](https://github.com/user-attachments/assets/87126d18-3120-48fc-99c3-baad3ef2d42a)

In particular I wanted to investigate how each algorithm performs based on matrices classified using multiple properties:
   - Total number of NNZ elements
   - Matrix density
   - Average number of NNZ elements per row
   -  Variance of NNZ elements per row

## Project Structure

The project is structured into three main C source files in the **`src`** folder:

- **`utils.h` and `utils.c`**:  
  These files define the CSR matrix structure and related data structures, such as matrix properties and execution statistics. Additionally, they implement helper functions such as:
    - Converting matrices from the `.mtx` format to the CSR format.
    - Functions to calculate matrix properties (density, variance, etc.).
    - Functions to save benchmarking statistics and execution times to files.

- **`sparse_matrix_multiplication.c`**:  
  This file contains the implementation of the three matrix multiplication functions, based on the partitioning schemes described above (Fixed Rows, Relaxed NNZ, and Strict NNZ partitioning). It includes OpenMP directives to parallelize the computation across multiple cores.

- **`main.c`**:  
  This file is the main entry point of the program. It orchestrates the process by loading the input matrices, converting them into CSR format, invoking the routines to measure the execution times and saving the logs.

The file **`run_on_dataset.sh`** is a bash script to run the executable for each file in the **`datset`** folder and invoke the **`plot_stats.py`** script to produce an histogram with the execution time of each partioning strategy on the matrices in the dataset.

## Dataset
The matrices used in this project were downloaded from [**SuiteSparse Matrix Collection**](https://sparse.tamu.edu/), a widely used repository of sparse matrices from various scientific and engineering fields. The matrices are provided in the `.mtx` (Matrix Market) format and are converted to CSR format throught the read_mtx_to_csr function.

## How to Run
To compile and run the project, use the following commands:
```bash
make              # Compiles the project
./run_script.sh
```
Alternatively add the **`sparse_matrix_multiplication.h`** header to invoke the matrix vector multiplication functions or **`helper.h`** for the helper functions.

## References
- [1] Gao, J., Ji, W., Chang, F., Han, S., Wei, B., Liu, Z., & Wang, Y. (2023). [A Systematic Survey of General Sparse Matrix-matrix Multiplication](https://doi.org/10.1145/3571157). *ACM Computing Surveys*, 55(12), Article 244. https://doi.org/10.1145/3571157
- [2] Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages. DOI: https://doi.org/10.1145/2049662.2049663
