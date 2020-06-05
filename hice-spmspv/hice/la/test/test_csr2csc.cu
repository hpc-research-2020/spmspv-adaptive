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

#define RIGHT

template<typename T>
int readSVMToCoo(char* file_name, SvmData<T>* svmdata);

template<typename T>
int freeSVMToCoo(SvmData<T>* svmdata);

template<typename T>
int ConvertSVMDataToCSR(SvmData<T> svmdata, 
                        int* csr_row, int* csr_col, T* csr_val, 
                        int numVects, int dimVects, int numNonZeros);

void SparseVec2DenseVec(VALUE_TYPE* y_dense, int len, 
                        /*SparseVec* y_sparse, */
                        int* y_sparse_key, VALUE_TYPE* y_sparse_val,
                        int y_nnz) {
  memset(y_dense, 0, len * sizeof(VALUE_TYPE));
  for (int i = 0; i < y_nnz; i++) {
    //y_dense[y_sparse[i].key] = y_sparse[i].val;
    y_dense[y_sparse_key[i]] = y_sparse_val[i];
  }
}

void Vec2File(VALUE_TYPE* y, int len, const char* file_name) {

  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout << "open file " << file_name << " failed." << std::endl;
      exit(1);
    }
   
  //std::cout << "result:" << std::endl;
  for (size_t i = 0; i < len; i++) {
    fout << y[i] << std::endl;
  }
  fout.close();
}

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

void TestCsr2Csc(int m, int n, int mat_nnz,
                 int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                 int* csc_row, int* csc_col, VALUE_TYPE* csc_val) {

  int* d_csr_row = NULL;
  int* d_csr_col = NULL;
  VALUE_TYPE* d_csr_val = NULL;

  int* d_csc_row = NULL;
  int* d_csc_col = NULL;
  VALUE_TYPE* d_csc_val = NULL;

  checkCudaErrors(cudaMalloc((void **)&d_csr_row,  (m + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csr_col,  mat_nnz  * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csr_val,  mat_nnz  * sizeof(VALUE_TYPE)));

  checkCudaErrors(cudaMalloc((void **)&d_csc_row, mat_nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csc_val, mat_nnz  * sizeof(VALUE_TYPE)));
  checkCudaErrors(cudaMalloc((void **)&d_csc_col, (n+1) * sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
                  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int),   
                  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE), 
                  cudaMemcpyHostToDevice));

  cusparseHandle_t sparse_handle;
  CUSP_CALL(cusparseCreate(&sparse_handle));
  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  checkCudaErrors(cudaDeviceSynchronize());

  //warmup
  #ifdef DOUBLE
      CUSP_CALL(cusparseDcsr2csc(sparse_handle, 
              m, n, mat_nnz, 
              d_csr_val, d_csr_row, d_csr_col, 
              d_csc_val, d_csc_row, d_csc_col, 
              CUSPARSE_ACTION_NUMERIC, 
              CUSPARSE_INDEX_BASE_ZERO));
  #else
      CUSP_CALL(cusparseScsr2csc(sparse_handle, 
              m, n, mat_nnz, 
              d_csr_val, d_csr_row, d_csr_col, 
              d_csc_val, d_csc_row, d_csc_col, 
              CUSPARSE_ACTION_NUMERIC, 
              CUSPARSE_INDEX_BASE_ZERO));
  #endif

  //perf test
  cudaEvent_t st, ed;
  float tm = 0;
  cudaEventCreate(&st);
  cudaEventCreate(&ed);
  cudaEventRecord(st, 0);

  for (int i = 0; i < 10; i++){
#ifdef DOUBLE
   CUSP_CALL(cusparseDcsr2csc(sparse_handle, 
            m, n, mat_nnz, 
            d_csr_val, d_csr_row, d_csr_col, 
            d_csc_val, d_csc_row, d_csc_col, 
            CUSPARSE_ACTION_NUMERIC, 
            CUSPARSE_INDEX_BASE_ZERO));
#else
   CUSP_CALL(cusparseScsr2csc(sparse_handle, 
            m, n, mat_nnz, 
            d_csr_val, d_csr_row, d_csr_col, 
            d_csc_val, d_csc_row, d_csc_col, 
            CUSPARSE_ACTION_NUMERIC, 
            CUSPARSE_INDEX_BASE_ZERO));
  #endif
  }

  cudaEventRecord(ed, 0);
  cudaEventSynchronize(ed);
  //unit: ms.
  cudaEventElapsedTime(&tm, st, ed);
  cudaEventDestroy(st);
  cudaEventDestroy(ed);

  //std::cout << "csr2csc time = " << tm/NUM_RUN <<" ms." << std::endl;
  std::cout << "csr2csc time = " << tm/10 <<" ms." << std::endl;
  

  checkCudaErrors(cudaMemcpy(csc_row, d_csc_row, mat_nnz * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csc_col, d_csc_col, (n + 1) * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(csc_val, d_csc_val, mat_nnz * sizeof(VALUE_TYPE),
                  cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_csr_row));
  checkCudaErrors(cudaFree(d_csr_col));
  checkCudaErrors(cudaFree(d_csr_val));
  checkCudaErrors(cudaFree(d_csc_row));
  checkCudaErrors(cudaFree(d_csc_col));
  checkCudaErrors(cudaFree(d_csc_val));
}

void TestConversion(int m, int n, int mat_nnz,
                    int* csr_row, int* csr_col, VALUE_TYPE* csr_val) {

  //csr2csc
  int* csc_row = NULL;
  int* csc_col = NULL;
  VALUE_TYPE* csc_val = NULL;
  
  csc_row = (int*)malloc(mat_nnz * sizeof(int));
  csc_col = (int*)malloc((n+1) * sizeof(int));
  csc_val = (VALUE_TYPE*)malloc(mat_nnz * sizeof(VALUE_TYPE));
  CHECK_MALLOC(csc_row);
  CHECK_MALLOC(csc_col);
  CHECK_MALLOC(csc_val);

  TestCsr2Csc(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                    csc_row, csc_col, csc_val);

  if (csc_row) free(csc_row);
  if (csc_col) free(csc_col);
  if (csc_val) free(csc_val);
}


//TODO: format conversion need to optimize.
int main(int argc, char** argv) {
  int m, n, mat_nnz;
  int* csr_row;
  int* csr_col;
  VALUE_TYPE* csr_val;

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

#ifndef READ_SVM_DATA
  MTX<VALUE_TYPE> mtx;
  fileToMtxCoo<VALUE_TYPE>(file_name, &mtx, true); 
  m = mtx.rows;
  n = mtx.cols;
  mat_nnz = mtx.nnz;

  //coo2csr: attention memory alloc and free.
  csr_row = (int* )malloc((m + 1) * sizeof(int));
  CHECK_MALLOC(csr_row);
  
  TestCoo2Csr(m, mat_nnz, mtx.row, csr_row);
  csr_col = mtx.col;
  csr_val = mtx.data;
#else
  SvmData<VALUE_TYPE> svm_data;
  readSVMToCoo<VALUE_TYPE>(file_name, &svm_data);

  m = svm_data.numVects;
  n = svm_data.dimVects;
  mat_nnz = svm_data.numNonZeros;
  //coo2csr: attention memory alloc and free.
  csr_row = (int* )malloc((m + 1) * sizeof(int));
  CHECK_MALLOC(csr_row);
  csr_col = (int* )malloc(mat_nnz * sizeof(int));
  CHECK_MALLOC(csr_col);
  csr_val = (VALUE_TYPE* )malloc(mat_nnz * sizeof(VALUE_TYPE));
  CHECK_MALLOC(csr_val);

  ConvertSVMDataToCSR(svm_data, csr_row, csr_col, csr_val, 
                      m, n, mat_nnz);
  freeSVMToCoo(&svm_data);
#endif

  // easy for test correctness.
  for (int i = 0; i < mat_nnz; i++) {
  	csr_val[i] = 1.0;
  }

  
  int y_nnz = 0;
  VALUE_TYPE alpha = 1;
  TestConversion(m, n, mat_nnz, csr_row, csr_col, csr_val);

#ifndef READ_SVM_DATA
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#endif

  return 0;
}
