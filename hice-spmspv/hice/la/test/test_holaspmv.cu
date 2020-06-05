#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <algorithm>
#include <vector>
#include <tuple>

#include <cusparse.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>



#include <omp.h>

#include "spmspv/config.h"
#include "spmspv/csc-spmspv/spmspv.h"
#include "spmspv/class.hpp"
#include "spmspv/mtx.hpp"
#include "spmspv/readsvmdata.hpp"
#include "../include/spmspv/csc-spmspv/detail/util.h"

//#include "../include/naivespmv.h"
//#include "../include/spmspv/holaspmv.h"

//added by
//#define READ_SVM_DATA

//#define TEST_SPMSPV
//#define TEST_SERIALSPMV
//#define TEST_NAIVESPMV
//#define TEST_HOLASPMV

//#define TEST_GEMV

#ifdef TEST_GEMV
#include <cublas_v2.h>

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS Error:  %s %d\n", file, line);
  }
}

#endif

//HolaMode holaMode = HolaMode::Default;
//unsigned int padding = 0; // 0; // 1024;

template<typename T>
void CheckVector(T* cpu, T* gpu, int len) {
	int flag = 1;
	for(int i = 0; i < len; i++) {
		if(cpu[i] != gpu[i]) {
			std::cout << "Err at " << i << ", cpu[i] = " << cpu[i] <<", gpu[i] = " << gpu[i] << std::endl;
			flag = 0;
		}
	}
	if(flag == 1) 
		std::cout <<"RESULT OK" <<std::endl;
}

void TestCoo2Csr(int m, int mat_nnz,
  int* coo_row, int* csr_row) {

int* d_csr_row = NULL;
int* d_coo_row = NULL;

cudaErrCheck(cudaMalloc((void** )&d_csr_row,  (m + 1) * sizeof(int)));
cudaErrCheck(cudaMalloc((void** )&d_coo_row, mat_nnz * sizeof(int)));

cudaErrCheck(cudaMemcpy(d_coo_row, coo_row, mat_nnz * sizeof(int),   
   cudaMemcpyHostToDevice));

cusparseHandle_t sparse_handle;
CUSP_CALL(cusparseCreate(&sparse_handle));
cusparseMatDescr_t descr = 0;
CUSP_CALL(cusparseCreateMatDescr(&descr));
cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

cudaErrCheck(cudaDeviceSynchronize());

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

cudaErrCheck(cudaMemcpy(csr_row, d_csr_row, (m + 1) * sizeof(int),   
  cudaMemcpyDeviceToHost));

cudaErrCheck(cudaDeviceSynchronize());

cudaErrCheck(cudaFree(d_csr_row));
cudaErrCheck(cudaFree(d_coo_row));
}  

#ifdef TEST_GEMV
int TestGemv(int m, int n, VALUE_TYPE* mat, VALUE_TYPE* x, VALUE_TYPE* y){
  int iter = 10;

  VALUE_TYPE* d_mat = NULL;
  VALUE_TYPE* d_x = NULL;
  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_mat,  m * n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMalloc((void **)&d_x,  n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMalloc((void **)&d_y,  m * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_mat, mat, m * n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));  
  cudaErrCheck(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));  
  
  const VALUE_TYPE alpha = 1.0f;
  const VALUE_TYPE beta  = 0.0f;
  cublasHandle_t handle;
  //cublasStatus_t
  cublasErrCheck(cublasCreate(&handle));

  //Perform warmup operation with cublas
  cublasErrCheck(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_mat, m, d_x, 1, &beta, d_y, 1));
  
  SpmspvTimer timer;
  timer.Start();
  for(int i=0; i<iter; i++){
    cublasErrCheck(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_mat, m, d_x, 1, &beta, d_y, 1));
  }
  double time = timer.Stop();
  time = time/iter;
  std::cout << "cublas gemv time = " << time << "ms." << std::endl;

  cudaErrCheck(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));    

  cublasErrCheck(cublasDestroy(handle));
}
#endif

int TestNaivespmv(int m, int n, int mat_nnz, int* csr_row, int* csr_col, 
            VALUE_TYPE* csr_val, VALUE_TYPE* x_dense, 
            VALUE_TYPE* y_dense, VALUE_TYPE alpha) {
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
  << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  double gflop = SpmvGetFlops<int>(mat_nnz);

  int* d_csr_row = NULL;
  int* d_csr_col = NULL;
  VALUE_TYPE* d_csr_val = NULL;

  cudaErrCheck(cudaMalloc((void **)&d_csr_row,  (m+1) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_col,  mat_nnz  * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_val,  mat_nnz  * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int),   
            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE),   
            cudaMemcpyHostToDevice));
  
  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
                            cudaMemcpyHostToDevice));
  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  
  err = A.set_vector_type(1);//current vector type is dense type.
  err = A.set_x(d_x); 
  
  SpmspvTimer timer;
  timer.Start();
  err = A.naivespmv(alpha, d_y);
  double pre_time = timer.Stop();
  std::cout << "naive spmv preprocess time = " << pre_time << "ms." << std::endl;

  timer.Start();
  for(int i=0; i<10; i++){
    err = A.naivespmv(alpha, d_y);
  }
  double spmv_time = timer.Stop()/10;
  std::cout << "naive spmv time = " << spmv_time << "ms." << std::endl;
  
  cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
  
  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_y)       cudaFree(d_y);
  
  A.Destroy();
return err;
}


int TestHolaspmv(int m, int n, int mat_nnz, int* csr_row, int* csr_col, 
            VALUE_TYPE* csr_val, VALUE_TYPE* x_dense, 
            VALUE_TYPE* y_dense, VALUE_TYPE alpha) {
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
  << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  double gflop = SpmvGetFlops<int>(mat_nnz);

  int* d_csr_row = NULL;
  int* d_csr_col = NULL;
  VALUE_TYPE* d_csr_val = NULL;

  cudaErrCheck(cudaMalloc((void **)&d_csr_row,  (m+1) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_col,  mat_nnz  * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_val,  mat_nnz  * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int),   
            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE),   
            cudaMemcpyHostToDevice));
  
  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
                            cudaMemcpyHostToDevice));
  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  
  err = A.set_vector_type(1);//current vector type is dense type.
  err = A.set_x(d_x); 
  
  SpmspvTimer timer;
  timer.Start();
  err = A.holaPreprocess();
  double pre_time = timer.Stop();
  std::cout << "hola spmv preprocess time = " << pre_time << "ms." << std::endl;

  timer.Start();
  for(int i=0; i<10; i++){
    err = A.holaspmv(alpha, d_y);
  }
  double spmv_time = timer.Stop()/10;
  std::cout << "hola spmv time = " << spmv_time << "ms." << std::endl;
  
  cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
  
  timer.Start();
  err = A.holaPostprocess();
  double post_time = timer.Stop();
  std::cout << "hola spmv postprocess time = " << post_time << "ms." << std::endl;

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_y)       cudaFree(d_y);
  
  A.Destroy();
return err;
}

int TestNaivespmspv(int m, int n, int mat_nnz, int* csr_row, int* csr_col, 
                VALUE_TYPE* csr_val, int x_nnz, int* x_key, VALUE_TYPE* x_val, 
                VALUE_TYPE* x_dense, VALUE_TYPE* y_dense, VALUE_TYPE alpha) {
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
            << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  double gflop = SpmvGetFlops<int>(mat_nnz);

  int* d_csr_row = NULL;
  int* d_csr_col = NULL;
  VALUE_TYPE* d_csr_val = NULL;

  cudaErrCheck(cudaMalloc((void **)&d_csr_row,  (m+1) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_col,  mat_nnz  * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_val,  mat_nnz  * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
               cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int),   
              cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
                            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE),   
                            cudaMemcpyHostToDevice));

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  // cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
  //                   cudaMemcpyHostToDevice));

  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  int num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  int* d_bit_vector = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_bit_vector, num_ints * sizeof(int)));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);

  err = A.set_vector_type(0);//current vector type is dense type.
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 
  err = A.set_bitvector(d_bit_vector, num_ints);//
  
  //A.allocPreBuffer();
  
  SpmspvTimer timer;

  timer.Start();
  A.sparse2bitarray();
  double time1 = timer.Stop();
  std::cout << " sparse2bitarray time = " << time1 << "ms." << std::endl;
  
  timer.Start();
  err = A.naivespmspv(alpha, d_y);
  double time = timer.Stop();
  std::cout << "naive spmspv preprocess time = " << time << "ms." << std::endl;

  timer.Start();
  for(int i=0; i<10; i++){
    err = A.naivespmspv(alpha, d_y);
  }
  time = timer.Stop()/10;
  std::cout << "naive spmspv time = " << time << "ms." << std::endl;

  cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_y)       cudaFree(d_y);

  //A.deallocPreBuffer();
  A.Destroy();
return err;
}


int TestHolaspmspv(int m, int n, int mat_nnz, int* csr_row, int* csr_col, 
                VALUE_TYPE* csr_val, int x_nnz, int* x_key, VALUE_TYPE* x_val, 
                VALUE_TYPE* x_dense, VALUE_TYPE* y_dense, VALUE_TYPE alpha) {
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
            << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  double gflop = SpmvGetFlops<int>(mat_nnz);

  int* d_csr_row = NULL;
  int* d_csr_col = NULL;
  VALUE_TYPE* d_csr_val = NULL;

  cudaErrCheck(cudaMalloc((void **)&d_csr_row,  (m+1) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_col,  mat_nnz  * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&d_csr_val,  mat_nnz  * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
               cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int),   
              cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
                            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE),   
                            cudaMemcpyHostToDevice));

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  // cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
  //                   cudaMemcpyHostToDevice));

  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  int num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  int* d_bit_vector = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_bit_vector, num_ints * sizeof(int)));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);

  err = A.set_vector_type(0);//current vector type is dense type.
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 
  err = A.set_bitvector(d_bit_vector, num_ints);//
  
  //A.allocPreBuffer();
  
  SpmspvTimer timer;

  timer.Start();
  A.sparse2bitarray();
  double time1 = timer.Stop();
  std::cout << " sparse2bitarray time = " << time1 << "ms." << std::endl;
  
  timer.Start();
  err = A.holaPreprocess();
  double time = timer.Stop();
  std::cout << "hola spmspv preprocess time = " << time << "ms." << std::endl;

  timer.Start();
  for(int i=0; i<10; i++){
    err = A.holaspmspv(alpha, d_y);
  }
  time = timer.Stop()/10;
  std::cout << "hola spmspv time = " << time << "ms." << std::endl;

  cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 

  timer.Start();
  err = A.holaPostprocess();
  time = timer.Stop();
  std::cout << "hola spmspv postprocess time = " << time << "ms." << std::endl;

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_y)       cudaFree(d_y);

  //A.deallocPreBuffer();
  A.Destroy();
return err;
}


void serialspmv(int m, int n, int mat_nnz,
                int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                VALUE_TYPE* x_dense, VALUE_TYPE* y_ref, VALUE_TYPE alpha) {
  for (int i = 0; i < m; i++) {
    VALUE_TYPE sum = 0;
    for (int j = csr_row[i]; j < csr_row[i+1]; j++)
      sum += x_dense[csr_col[j]] * csr_val[j] * alpha;
    y_ref[i] = sum;
  }
}

int main(int argc, char** argv) {
  int m, n, mat_nnz;
  int* csr_row;
  int* csr_col;
  VALUE_TYPE* csr_val;

  // report precision of floating-point
  std::cout << "-------------------------" << std::endl;
  std::string precision;
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
  //printMtx2File(&mtx);

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

  std::cout << " m = " << m << ", n = " << n << ", nnz=" << mat_nnz << std::endl;

#ifdef TEST_GEMV

#ifndef READ_SVM_DATA
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#endif

  VALUE_TYPE* mat = (VALUE_TYPE* )malloc(n * m * sizeof(VALUE_TYPE)); CHECK_MALLOC(mat);
  VALUE_TYPE* x = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));       CHECK_MALLOC(x);
  VALUE_TYPE* y = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));       CHECK_MALLOC(y);

  for (int i = 0; i < m * n; i++)  mat[i] = 1.0;
  for(int i = 0; i < n; i++)       x[i] = i;

  std::cout << "begin test gemv: " << std::endl;
  TestGemv(m, n, mat, x, y);
  std::cout << "end test gemv: " << std::endl;

  if (mat)  free(mat);
  if (x)    free(x);
  if (y)    free(y);

#else
  // SparseVec* x_sparse = (SparseVec* )malloc(n * sizeof(SparseVec));
  // CHECK_MALLOC(x_sparse);
  int* x_sparse_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_sparse_key);
  VALUE_TYPE* x_sparse_val = (VALUE_TYPE* )malloc(n * 
                            sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_sparse_val);
  
  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * 
    sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);

  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  VALUE_TYPE* hres =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));

  CHECK_MALLOC(y_dense);
  CHECK_MALLOC(hres);

#ifdef READ_SVM_DATA
  //read sparse vector from file
  char* suffix = ".out";
  char svm_file_name[35];
  sprintf(svm_file_name, "%s%s", file_name, suffix);
  std::cout << "svm_file_name = " << svm_file_name << std::endl;

  int iter = 0;
  getNumofRows(svm_file_name, &iter);
  std::cout << "svm iter = " << iter << std::endl;

  int* ind = (int*)malloc(iter*sizeof(int));
  CHECK_MALLOC(ind);

  readSparseXInxFromFile(svm_file_name, iter, ind);

  int new_iter = 0;
  removeRepetition(iter, ind, &new_iter, ind);
  std::cout <<  "iter = " << iter << " ,new_iter = " 
            << new_iter << std::endl;

  std::cout << "-------------------------------" << std::endl;

  //run the iterations of spmspv
  //for (int i = 0; i < new_iter; i++) {
  for (int i = 0; i < new_iter; i++) {
    //the number of the nonzeroes of the ind[i]-th row.
    //i = 524;
    int x_nnz = csr_row[ind[i]+1] - csr_row[ind[i]];
    printf("ind[%d] = %d, x_nnz = %d\n", i, ind[i], x_nnz);

    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));

    extractSparseXfromMat<int, VALUE_TYPE>(ind[i], csr_row, csr_col, csr_val, 
                      m, n, mat_nnz, x_sparse_key, x_sparse_val);
#else
  int iter = 1;
  //int xnnz_vec[1]={991};
  //float x_sparsity[4] = {0.1, 0.5, 0.9, 1};
  float x_sparsity[1] = {1};
  for (int i = 0; i < iter; i++) {
    //int x_nnz = xnnz_vec[i];
    //int x_nnz = n;
    int x_nnz = (int)(n * x_sparsity[i]);
    if(x_nnz >= n)  x_nnz = n;
    //if(x_nnz > n) break;
    printf("sparsity = %f, x_nnz = %d\n", x_sparsity[i], x_nnz);
    
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));
    memset(x_dense, 0, n * sizeof(VALUE_TYPE));

#ifdef SPARSE_X_RANDOM
    srand(time(NULL));
    generate_random_sparse_vector(n, x_nnz, x_sparse_key, x_sparse_val);
#else
    for (int i = 0; i < x_nnz; i++) {
      x_sparse_key[i] = i;
      x_sparse_val[i] = 1;
      x_dense[i] = 1;
    }
#endif

#endif //end of different methods to read sparse x.
    
    VALUE_TYPE alpha = 1.0;
    memset(y_dense, 0, m * sizeof(VALUE_TYPE));//
    serialspmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, y_dense, alpha);
    
    memset(hres, 0, m * sizeof(VALUE_TYPE));//
    TestNaivespmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, hres, alpha);
    std::cout << std::endl;
    CheckVector<VALUE_TYPE>(y_dense, hres, m);
    
    memset(hres, 0, m * sizeof(VALUE_TYPE));//
    TestNaivespmspv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_nnz, x_sparse_key, x_sparse_val, x_dense, hres, alpha);
    std::cout << std::endl;
    CheckVector<VALUE_TYPE>(y_dense, hres, m);

    memset(hres, 0, m * sizeof(VALUE_TYPE));//
    TestHolaspmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, hres, alpha);
    std::cout << std::endl;
    CheckVector<VALUE_TYPE>(y_dense, hres, m);
    
    memset(hres, 0, m * sizeof(VALUE_TYPE));//
    TestHolaspmspv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_nnz, x_sparse_key, x_sparse_val, x_dense, hres, alpha);
    std::cout << std::endl;
    CheckVector<VALUE_TYPE>(y_dense, hres, m);

  }//end of iteration.

  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);
  if (x_dense) free(x_dense);
  
  if (y_dense)         free(y_dense);
  if (hres)    free(hres);

#ifndef READ_SVM_DATA
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
  if (ind)       free(ind);
#endif

#endif

  return 0;
}



