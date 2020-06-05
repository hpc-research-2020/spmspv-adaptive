// This is used to Test my Spmspv performance in applications. 

#include <iostream>
#include <string>
#include <float.h>
#include <typeinfo>
#include <limits>
#include <algorithm>
#include <vector>
#include <random>

#include <omp.h>

#include <cusparse.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

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

template<typename T>
int readSVMToCoo(char* file_name, SvmData<T>* svmdata);

template<typename T>
int freeSVMToCoo(SvmData<T>* svmdata);

template<typename T>
int ConvertSVMDataToCSR(SvmData<T> svmdata, 
                        int* csr_row, int* csr_col, T* csr_val, 
                        int numVects, int dimVects, int numNonZeros);

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

  std::cout << "csr2coo time = " << tm <<" ms." << std::endl;

  cudaErrCheck(cudaMemcpy(csr_row, d_csr_row, (m + 1) * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  
  cudaErrCheck(cudaDeviceSynchronize());

  cudaErrCheck(cudaFree(d_csr_row));
  cudaErrCheck(cudaFree(d_coo_row));
}  

int TestSparse2Dense(int x_nnz, int* x_key, VALUE_TYPE* x_val, 
                    int m, int n, int nnz, VALUE_TYPE* x_dense){
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
            << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  // double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  // double gflop = SpmvGetFlops<int>(mat_nnz);
  std::cout << "DEBUG: n = " << n << std::endl;
  std::cout << "DEBUG: xnnz = " << x_nnz << std::endl;

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
                            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE),   
                            cudaMemcpyHostToDevice));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, nnz);


  //current vector type is sparse type.
  err = A.set_vector_type(0);
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 
  
  A.allocPreBuffer();

  SpmspvTimer s2d_timer;
  s2d_timer.Start();
  //sparse vect to dense vec.
  err = A.sparse2dense();
  double time = s2d_timer.Stop();

  std::cout << "sparse2dense time = " <<  time << " ms." << std::endl;
  
  cudaErrCheck(cudaFree(d_x));        
  cudaErrCheck(cudaFree(d_x_key)); 
  cudaErrCheck(cudaFree(d_x_val)); 
  
  A.deallocPreBuffer();
  A.Destroy();

return err;      
}


int TestDense2Sparse(int x_nnz, int* x_key, VALUE_TYPE* x_val, 
  int m, int n, int nnz, VALUE_TYPE* x_dense){
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
  << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  // double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  // double gflop = SpmvGetFlops<int>(mat_nnz);
  std::cout << "DEBUG: n = " << n << std::endl;
  std::cout << "DEBUG: xnnz = " << x_nnz << std::endl;

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x, x_dense,  n * sizeof(VALUE_TYPE),   
  cudaMemcpyHostToDevice));


  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  //TODO: check d_x_key and d_x_val 需要初始化为0吗？

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, nnz);


  //only assignment. Don not have memory alloc.
  //current vector type is dense type.
  err = A.set_vector_type(1);
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 

  A.allocPreBuffer();

  SpmspvTimer d2s_timer;
  d2s_timer.Start();
  //dense vect to sparse vec.
  err = A.dense2sparse();
  double time = d2s_timer.Stop();

  std::cout << "dense2sparse time = " << time << " ms." << std::endl;

  cudaErrCheck(cudaFree(d_x));        
  cudaErrCheck(cudaFree(d_x_key)); 
  cudaErrCheck(cudaFree(d_x_val));     

  A.deallocPreBuffer();  
  A.Destroy();

return err;
}

int Testsparse2bitarray(int x_nnz, int* x_key, VALUE_TYPE* x_val, 
  int m, int n, int nnz, VALUE_TYPE* x_dense){
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
  << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  // double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  // double gflop = SpmvGetFlops<int>(mat_nnz);
  std::cout << "DEBUG: n = " << n << std::endl;
  std::cout << "DEBUG: xnnz = " << x_nnz << std::endl;

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE),   
            cudaMemcpyHostToDevice));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, nnz);

  //current vector type is sparse type.
  err = A.set_vector_type(0);
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 

  //alloc for bitarray in this kernel. 
  A.allocPreBuffer();

  SpmspvTimer s2b_timer;
  s2b_timer.Start();
  //sparse vect to dense vec.
  err = A.sparse2bitarray();
  double time = s2b_timer.Stop();

  std::cout << "sparse2bitarray time = " <<  time << " ms." << std::endl;

  cudaErrCheck(cudaFree(d_x));        
  cudaErrCheck(cudaFree(d_x_key)); 
  cudaErrCheck(cudaFree(d_x_val)); 

  A.deallocPreBuffer();
  A.Destroy();

return err;      
}

int Testdense2bitarray(int x_nnz, int* x_key, VALUE_TYPE* x_val, 
  int m, int n, int nnz, VALUE_TYPE* x_dense){
  int err = 0;
  //cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
  << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  // double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  // double gflop = SpmvGetFlops<int>(mat_nnz);
  std::cout << "DEBUG: n = " << n << std::endl;
  std::cout << "DEBUG: xnnz = " << x_nnz << std::endl;

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x, x_dense,  n * sizeof(VALUE_TYPE),   
  cudaMemcpyHostToDevice));

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  //TODO: check d_x_key and d_x_val

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, nnz);

  //only assignment. Don not have memory alloc.
  //current vector type is dense type.
  err = A.set_vector_type(1);
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 

  A.allocPreBuffer();

  SpmspvTimer d2b_timer;
  d2b_timer.Start();
  //dense vect to sparse vec.
  err = A.dense2bitarray();
  double time = d2b_timer.Stop();

  std::cout << "dense2bitarray time = " << time << " ms." << std::endl;

  cudaErrCheck(cudaFree(d_x));        
  cudaErrCheck(cudaFree(d_x_key)); 
  cudaErrCheck(cudaFree(d_x_val));     

  A.deallocPreBuffer();  
  A.Destroy();

return err;
}


int SelectSpMV(int m, int n, int mat_nnz,
  int* csr_row, int* csr_col, 
  VALUE_TYPE* csr_val,
  VALUE_TYPE* x_dense, 
  VALUE_TYPE* y_dense, 
  VALUE_TYPE alpha) {
  int err = 0;
  cudaError_t err_cuda = cudaSuccess;

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

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
                  cudaMemcpyHostToDevice));
  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.

  err = A.set_vector_type(1);//current vector type is sparse type.
  err = A.setX(d_x); 

  A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
  A.warmup();

  SpmspvTimer timer;
  timer.Start();
  err = A.CSR5Preprocess();
  std::cout << "CSR->CSR5 time = " << timer.Stop() 
      << " ms." << std::endl;

  //check correctness by running 1 time
  err = A.csr5spmv(alpha, d_y);
  cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 

  // warm up by running 50 times
  if (NUM_RUN) {
  for (int i = 0; i < 50; i++)
  err = A.csr5spmv(alpha, d_y);
  }
  err_cuda = cudaDeviceSynchronize();

  // test 
  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
  err = A.csr5spmv(alpha, d_y);
  err_cuda = cudaDeviceSynchronize();
  double time = timer.Stop() / (double)NUM_RUN;

  timer.Start();
  A.CSR5Postprocess();
  std::cout << "CSR5->CSR time = " << timer.Stop() 
      << " ms." << std::endl;

  if (NUM_RUN)
  std::cout << "CSR5-based SpMV time = " << time
    << " ms. Bandwidth = " << gb/(1.0e+6 * time)
    << " GB/s. GFlops = " << gflop/(1.0e+6 * time)  
    << " GFlops." << std::endl;

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_y)       cudaFree(d_y);

  A.Destroy();
return err;
}

int TestCscNoSortOrBucketSpmspv(int m, int n, int mat_nnz,
                    int* csr_row, int* csr_col, 
                    VALUE_TYPE* csr_val,
                    int x_nnz, 
                    int* x_key, VALUE_TYPE* x_val,
                    VALUE_TYPE* x_dense, 
                    int* y_nnz,
                    int* y_inx, VALUE_TYPE* y_val,
                    VALUE_TYPE* y_dense, 
                    VALUE_TYPE alpha) {
  /****step-1: format conversion****/
  int err = 0;
  cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
            << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  // double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  // double gflop = SpmvGetFlops<int>(mat_nnz);
  //std::cout << "DEBUG: xnnz = " << x_nnz << std::endl;
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

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

  cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
                            cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE),   
                            cudaMemcpyHostToDevice));
  //result vec.
  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  err = A.ToCSC();

  err = A.set_vector_type(0);//current vector type is sparse type.
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  err = A.set_x(d_x); 
  
  /*********select spmv.***********/
  SpmspvTimer timer;

  err = A.holaPreprocess();
  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
    err = A.holaspmv(alpha, d_y);
  err_cuda = cudaDeviceSynchronize();
  double holaspmv_time = timer.Stop()/ (double)NUM_RUN;
  
  err = A.CSR5Preprocess();
  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
      err = A.csr5spmv(alpha, d_y);
  err_cuda = cudaDeviceSynchronize();
  double csr5spmv_time = timer.Stop() / (double)NUM_RUN;
  
  if (csr5spmv_time /holaspmv_time > 1.2) {
    A.set_spmv_type(0);//
    A.CSR5Postprocess();
  }else{
    A.set_spmv_type(1);//
    A.holaPostprocess();
  }
  
  A.allocPreBuffer();
  //generate x: iterate begin. 
  
  timer.Start();
  int bin_len = A.computeBinlen();
  double time = timer.Stop();
  std::cout << "DEBUG: compute bin_len time = " << time << "ms." << std::endl;
  std::cout << "DEBUG: TestCscNoSortOrBucketSpmspv: bin_len = " << bin_len << std::endl;

  int GM1 = A.computeGM1();
  std::cout << "DEBUG: TestCscNoSortOrBucketSpmspv: GM1 = " << GM1 << std::endl;

  int GM2 =  A.computeGM2();
  std::cout << "DEBUG: TestCscNoSortOrBucketSpmspv: GM2 = " << GM2 << std::endl;
  
  int GM3 = A.computeGM3();
  std::cout << "DEBUG: TestCscNoSortOrBucketSpmspv: GM3 = " << GM3 << std::endl;
  
  timer.Start();
  err = A.CscBasedNoSortSpmspv(alpha, y_nnz, d_y);
  time = timer.Stop();
  std::cout << "csc spmspv time = " << time << "ms." << std::endl;
  
  //cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));  
 
  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_x_key)   cudaFree(d_x_key);
  if (d_x_val)   cudaFree(d_x_val);
  if (d_y)       cudaFree(d_y);
  A.deallocPreBuffer();
  A.Destroy();
  return err;
}

bool key_present(int dst, const int* key, 
        const int start, const int end) {
  bool retval = false;
  for (int i = start; i < end; i++) {
    if (key[i] == dst) {
      retval = true;
      break;
    }
  }
  return retval;
}
template <typename vT=int>
void generate_random_sparse_vector(int n, int nnz, int* key, vT* value) {
  if (nnz > n) { 
    nnz = n;
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, n);

  for (int j = 0; j < nnz; j++) {
    int dst;
    do {
      dst = dist(gen);
    } while(key_present(dst, key, 0, j));

    key[j] = dst;
    value[j] = 1;
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
  //SparseVec* y_sparse = (SparseVec* )malloc(m * sizeof(SparseVec));
  int* y_sparse_key =  (int* )malloc(m * sizeof(int));
  VALUE_TYPE* y_sparse_val =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));

  CHECK_MALLOC(y_dense);
  //CHECK_MALLOC(y_sparse);
  CHECK_MALLOC(y_sparse_key);
  CHECK_MALLOC(y_sparse_val);

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
 
  int iter = 11;
  int xnnz_vec[11]={1,1,3851,814949,2166177,511191,13677,2306,272,34,4};
  
   
  for (int i = 0; i < iter; i++) {
    int x_nnz = xnnz_vec[i];
    if(x_nnz >= n)  x_nnz = n;
    //if(x_nnz > n) break;
    printf("x_nnz = %d\n", x_nnz);
    
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
   
#ifdef TEST_S2D_D2S
    TestSparse2Dense(x_nnz, x_sparse_key, x_sparse_val, 
                     m, n, mat_nnz, x_dense);

    TestDense2Sparse(x_nnz, x_sparse_key, x_sparse_val, 
                     m, n, mat_nnz, x_dense); 

    Testsparse2bitarray(x_nnz, x_sparse_key, x_sparse_val, 
                     m, n, mat_nnz, x_dense);

    Testdense2bitarray(x_nnz, x_sparse_key, x_sparse_val, 
                     m, n, mat_nnz, x_dense);                     
#endif

#ifdef SMSV_CUDA
    VALUE_TYPE alpha = 1.0;
    memset(y_dense, 0, m * sizeof(VALUE_TYPE));//
    SelectSpMV(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, y_dense, alpha);
    std::cout << std::endl; 

    int cu_y_nnz = 0;
    memset(y_dense, 0, m * sizeof(VALUE_TYPE));//
    TestCscNoSortOrBucketSpmspv(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                              x_nnz, x_sparse_key, x_sparse_val, 
                              x_dense,
                              &cu_y_nnz,
                              y_sparse_key, y_sparse_val, y_dense, alpha);
    std::cout << std::endl;                              
#endif
  }//end of iteration.

  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);

  if (x_dense) free(x_dense);
  
  //if (y_sparse)    free(y_sparse);
  if (y_dense)         free(y_dense);
  if (y_sparse_key)    free(y_sparse_key);
  if (y_sparse_val)    free(y_sparse_val);

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

  return 0;
}
