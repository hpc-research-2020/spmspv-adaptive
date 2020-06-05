// This is used to Test cusparse's csr2csc kernel.

#include <iostream>
#include <string>
#include <float.h>
#include <typeinfo>
#include <limits>
#include <algorithm>
#include <vector>

#include <omp.h>

#include <cusparse.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

//#include <thrust/execution_policy.h>
//#include <thrust/device_vector.h>
//#include <thrust/scan.h>
//#include <thrust/sort.h>

#include "spmspv/csc-spmspv/spmspv.h"
#include "spmspv/class.hpp"
#include "spmspv/mtx.hpp"
#include "spmspv/readsvmdata.hpp"
#include "spmspv/config.h"

#ifndef VALUE_TYPE
#define VALUE_TYPE float
#endif

#ifndef NUM_RUN
#define NUM_RUN 10
#endif
    
#define IS_DOUBLE_ZERO(d)  (abs(d) < DBL_EPSILON)
#define IS_FLOAT_ZERO(d)  (abs(d) < FLT_EPSILON)


void TestCoo2Csr(int m, int mat_nnz,
                 int* coo_row, int* csr_row) {

  int* d_csr_row = NULL;
  int* d_coo_row = NULL;
  
  checkCudaErrors(cudaMalloc((void** )&d_csr_row,  (m + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void** )&d_coo_row, mat_nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_coo_row, coo_row, mat_nnz * sizeof(int),   
                  cudaMemcpyHostToDevice));

  cusparseHandle_t sparse_handle;
  CUSP_CALL(cusparseCreate(&sparse_handle));
  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  checkCudaErrors(cudaDeviceSynchronize());

  cudaEvent_t st, ed;
  float tm = 0;
  cudaEventCreate(&st);
  cudaEventCreate(&ed);
  cudaEventRecord(st, 0);

  CUSP_CALL(cusparseXcoo2csr(sparse_handle, 
            d_coo_row, mat_nnz, m, d_csr_row, 
            CUSPARSE_INDEX_BASE_ZERO));

  cudaEventRecord(ed, 0);
  cudaEventSynchronize(ed);
  //unit: ms.
  cudaEventElapsedTime(&tm, st, ed);
  cudaEventDestroy(st);
  cudaEventDestroy(ed);

  std::cout << "coo2csr time = " << tm <<" ms." << std::endl;

  checkCudaErrors(cudaMemcpy(csr_row, d_csr_row, (m + 1) * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_csr_row));
  checkCudaErrors(cudaFree(d_coo_row));
}  

void testHybridSpMV(int m, int n, int mat_nnz, int csr_nnz, int* csr_row, int* csr_col, VALUE_TYPE* csr_val, int* csc_row, int* csc_col, VALUE_TYPE* csc_val, int* x_key, VALUE_TYPE* x_val, VALUE_TYPE* x_dense, VALUE_TYPE* y_dense){
  int err = 0;
  cudaError_t err_cuda = cudaSuccess;
  
  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
            << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;
  
  int* d_csr_row = NULL;
  int* d_csr_col = NULL;
  VALUE_TYPE* d_csr_val = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_csr_row,  (m+1) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_col,  csr_nnz  * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_val,  csr_nnz  * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
      cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_col, csr_col, csr_nnz * sizeof(int),   
      cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_val, csr_val, csr_nnz * sizeof(VALUE_TYPE),   
      cudaMemcpyHostToDevice));
  
  int csc_nnz = mat_nnz - csr_nnz;
  int* d_csc_row = NULL;
  int* d_csc_col = NULL;
  VALUE_TYPE* d_csc_val = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_csc_col,  (n+1) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csc_row,  csc_nnz  * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csc_val,  csc_nnz  * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_csc_col, csc_col, (n+1) * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csc_row, csc_row, csc_nnz * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csc_val, csc_val, csc_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
  
  
  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
  
  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  int x_nnz = n;//for dense input vector.
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (n) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (n) * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
  
  int num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  int* d_bit_vector = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_bit_vector, num_ints * sizeof(int)));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, csr_nnz);
  err = A.InputCSR(csr_nnz, d_csr_row, d_csr_col, d_csr_val);
  
  err = A.set_vector_type(1);//current vector type is densetype.
  err = A.set_x(d_x); //
  err = A.set_bitvector(d_bit_vector, num_ints);//
  
  VALUE_TYPE* d_y = NULL; //result vec.
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  int* d_y_key = NULL; //result vec.
  VALUE_TYPE* d_y_val = NULL; 
  cudaErrCheck(cudaMalloc((void** )&d_y_key, m * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_y_val, m * sizeof(VALUE_TYPE)));

  VALUE_TYPE alpha_i = 1.0;

  SpmspvTimer timer;
  
  err = A.holaPreprocess();
  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
    err = A.holaspmv(alpha_i, d_y);
  err_cuda = cudaDeviceSynchronize();
  double holaspmv_time = timer.Stop()/ (double)NUM_RUN;
  std::cout << "holaspmv time = " << holaspmv_time << std::endl; 

  A.holaPostprocess();
  A.Destroy();
  
  
  SpmspvHandle<int, unsigned int, VALUE_TYPE> A2(m, n, csc_nnz);
  err = A2.InputCSC(csc_nnz, d_csc_row, d_csc_col, d_csc_val);
  
  err = A2.set_sparse_x(d_x_key, d_x_val, x_nnz);//
  err = A2.set_x(d_x);//
    
  
  A2.allocPreBuffer();
    
  timer.Start();
  int bin_len = A2.computeBinlenVer2();
  double time = timer.Stop();
  std::cout << "DEBUG: compute bin_len time = " << time << "ms." << std::endl;
    
  int y_nnz = 0;
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  timer.Start();
  //err = A.CscBasedNoSortMergeSpmspv_keyval(true, alpha_i, &y_nnz, d_y, d_y_key, d_y_val);
  err = A2.CscBasedNoSortMergeSpmspv(true, alpha_i, &y_nnz, d_y, d_y_key, d_y_val);
  double csc_time = timer.Stop();
  std::cout << "my lb col-spmspv time = " << csc_time << "ms." << std::endl;
  std::cout << "all time = " << csc_time + holaspmv_time << "ms." << std::endl;
   
  cudaError_t err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("cscspmspv() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }
    
  A2.deallocPreBuffer();
  A2.Destroy();
  
  
  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  
  if (d_x)       cudaFree(d_x);
  if (d_x_key)   cudaFree(d_x_key);
  if (d_x_val)   cudaFree(d_x_val);
  
  if (d_y)       cudaFree(d_y);
  if (d_y_key)       cudaFree(d_y_key);
  if (d_y_val)       cudaFree(d_y_val);

  //return err;
}

int main(int argc, char** argv) {
  int m, n, mat_nnz;
  int* coo_row;
  int* coo_col;
  VALUE_TYPE* coo_val;
  
  int* csr_cluster_row;
  int* csr_col;
  VALUE_TYPE* csr_val;
  
  int* csc_row;
  int* csc_cluster_col;
  VALUE_TYPE* csc_val;

  int* csr_row;
  int* csc_col;

  int ret_csr_nnz = 0;
  // report precision of floating-point
  std::cout << "-------------------------" << std::endl;
  char* precision;
  if (sizeof(VALUE_TYPE) == 4) {
      precision = "32-bit Single Precision (float)";
  } else if (sizeof(VALUE_TYPE) == 8) {
      precision = "64-bit Double Precision (double)";
  } else {
      std::cout << "Wrong precision. Program exit!" << std::endl;
      return 0;
  }
  std::cout << "PRECISION = " << precision << std::endl;
  std::cout << "-------------------------" << std::endl;

  char* file_name;
  if (argc == 2) {
    file_name = argv[1];
  } else {
    std::cout << "Usage: matrix_file_name" << std::endl;
    exit(1);
  }
  std::cout << "-------" << file_name << "---------" << std::endl;

  MTX<VALUE_TYPE> mtx;
  fileToMtxCoo<VALUE_TYPE>(file_name, &mtx, true); 
  m = mtx.rows;
  n = mtx.cols;
  mat_nnz = mtx.nnz;

  printf("m=%d, n=%d, nnz=%d\n", m, n, mat_nnz);
  
  coo_row = mtx.row;
  coo_col = mtx.col;
  coo_val = mtx.data;
  
  COO2Hybrid<VALUE_TYPE>(1, m, n, mat_nnz, coo_row, coo_col, coo_val, &ret_csr_nnz);
  
  std::cout << "ret csr_nnz = " << ret_csr_nnz << std::endl;

  csr_cluster_row = coo_row;
  csr_col = coo_col;
  csr_val = coo_val;

  csc_row = &coo_row[ret_csr_nnz];
  csc_cluster_col = &coo_col[ret_csr_nnz];
  csc_val = &coo_val[ret_csr_nnz];
  
  csr_row = (int* )malloc((m + 1) * sizeof(int));
  CHECK_MALLOC(csr_row);
  
  csc_col = (int* )malloc((n + 1) * sizeof(int));
  CHECK_MALLOC(csc_col);
#if 1  
  TestCoo2Csr(m, ret_csr_nnz, csr_cluster_row, csr_row);
  TestCoo2Csr(n, mat_nnz - ret_csr_nnz, csc_cluster_col, csc_col);
#endif
  int* x_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_key);
  VALUE_TYPE* x_val = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_val);
  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);
  for(int i=0; i<n; i++) {
    x_dense[i] = (VALUE_TYPE)i;
    x_key[i] = i;
    x_val[i] = (VALUE_TYPE)i;
  }
  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(y_dense);
  
  
  testHybridSpMV(m, n, mat_nnz, ret_csr_nnz, csr_row, csr_col, csr_val, csc_row, csc_col, csc_val, 
      x_key, x_val, x_dense, y_dense);
 
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
  
  if(csr_row) free(csr_row);
  if(csc_col) free(csc_col);
  
  if (x_key)      free(x_key);
  if (x_val)      free(x_val);
  if (x_dense)      free(x_dense);
  if (y_dense)      free(y_dense);
  return 0;
}
