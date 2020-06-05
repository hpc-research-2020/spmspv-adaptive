// This is used to TestCscSpmspv: my work.

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

//#define READ_SVM_DATA 1

//#define SPARSE_X_RANDOM

//#define CORRECT
#define MY_SPMSPV 1

#ifndef VALUE_TYPE
#define VALUE_TYPE float
#endif

#ifndef NUM_RUN
#define NUM_RUN 1
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

  std::cout << "csr2coo time = " << tm <<" ms." << std::endl;

  checkCudaErrors(cudaMemcpy(csr_row, d_csr_row, (m + 1) * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_csr_row));
  checkCudaErrors(cudaFree(d_coo_row));
}  

int TestCscSpmspv(int m, int n, int mat_nnz,
                    int* csr_row, int* csr_col, 
                    VALUE_TYPE* csr_val,
                    int x_nnz, /*SparseVec* x,*/
                    int* x_key, VALUE_TYPE* x_val,
                    VALUE_TYPE* x_dense,
                    int* y_nnz, /*SparseVec* y,*/
                    int* y_inx, VALUE_TYPE* y_val,
                    VALUE_TYPE* y_dense,
                    VALUE_TYPE* y_sort_dense, 
                    VALUE_TYPE alpha/*,
                    std::ofstream fout*/) {
  /****step-1: format conversion****/
  int err = 0;
  cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::ofstream fout;
  fout.open("./column-major-spmspv-perf.out", std::ofstream::app);
  if(!fout){
    std::cout << "file can't open" << std::endl;
    exit(1);
  }                      
  // std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
  //           << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

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

#if 1
  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  err = A.ToCSC();

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

  VALUE_TYPE* d_y = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_y,  m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE)));

  int* d_y_key_output = NULL;
  VALUE_TYPE* d_y_val_output = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y_key_output, (m) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_y_val_output, (m) * sizeof(VALUE_TYPE)));

  err = A.set_vector_type(0);//current vector type is sparse type.
  err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
  //std::cout << "DEBUG: set_sparse_x err = " << err << std::endl; 
  err = A.set_x(d_x); 

  A.allocPreBuffer();

  long long int bin_len = A.computeBinlen();
  std::cout << "computebin_len = " << bin_len << std::endl;
  fout << "binlen/nnz = " << bin_len/(float)mat_nnz << std::endl;
  //int group_num = 2;
  //err = A.ToBcsc(group_num)；

#if 1

  SpmspvTimer timer;
  double time0 = 0, time1 = 0, time2 = 0, time3 = 0, time4 = 0, time5 = 0, time6 = 0;
#if 1
  timer.Start();
  for (int i = 0; i < 1/*NUM_RUN*/; i += 1) {
    //A.CscBasedSortSpmspv(alpha, y_nnz, d_y_key, d_y_val, 0);
    A.CscBasedSortNaiveSpmspv(alpha, y_nnz, d_y_key_output, d_y_val_output, 0);
  }
  time0 = timer.Stop()/1 /*NUM_RUN*/;
  std::cout << "sort-based naive spmspv time = " << time0 << " ms." << std::endl << std::endl;
  fout << time0 << " ";
  printf("y_nnz = %d\n", *y_nnz);
#ifdef CORRECT
  A.sparse2dense_outer(*y_nnz, d_y_key_output, d_y_val_output, m, d_y);
  cudaMemcpy(y_sort_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
#endif

#endif

#if 1
  timer.Start();
  for (int i = 0; i < 1/*NUM_RUN*/; i += 1) {
    //A.CscBasedSortSpmspv(alpha, y_nnz, d_y_key, d_y_val, 0);
    A.CscBasedSortMergeSpmspv(false, alpha, y_nnz, d_y, d_y_key_output, d_y_val_output);
  }
  time1 = timer.Stop()/1 /*NUM_RUN*/;
  std::cout << "sort-based merge spmspv time = " << time1 << " ms." << std::endl << std::endl;
  fout << time1 << " ";
  printf("y_nnz = %d\n", *y_nnz);
  #ifdef CORRECT
  A.sparse2dense_outer(*y_nnz, d_y_key_output, d_y_val_output, m, d_y);
  cudaMemcpy(y_sort_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
  #endif
#endif

#if 1
  timer.Start();
  for (int i = 0; i < 1/*NUM_RUN*/; i += 1) {
    //A.CscBasedSortSpmspv(alpha, y_nnz, d_y_key, d_y_val, 0);
    A.CscBasedSortMySpmspv(false, alpha, y_nnz, d_y, d_y_key_output, d_y_val_output);
  }
  time2 = timer.Stop()/1 /*NUM_RUN*/;
  std::cout << "sort-based my spmspv time = " << time2 << " ms." << std::endl << std::endl;
  fout << time2 << " ";
  printf("y_nnz = %d\n", *y_nnz);
  #ifdef CORRECT
  A.sparse2dense_outer(*y_nnz, d_y_key_output, d_y_val_output, m, d_y);
  cudaMemcpy(y_sort_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
  #endif
#endif

  //std::cout << "sort speedup(naive/my) = " << time0/time2 << std::endl;
  //std::cout << "sort speedup(naive/merge) = " << time0/time1 << std::endl;
  //std::cout << "sort speedup(merge/my) = " << time1/time2 << std::endl << std::endl;

  printf("\n\n");

//     //for(int buckets = 32; buckets <= 8192; buckets *= 2){
//     for (int buckets = 4; buckets <= 256; buckets *= 2) {
//     //for(int buckets = 4; buckets <= 4; buckets *= 2){
//       err = A.CscBasedBucketSpmspv(alpha, y_nnz, y_dense, buckets);
//     }


#if 1
  time3 = 0.0;
  for (int i = 0; i < 1/*NUM_RUN*/; i += 1) {
    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    timer.Start();
    A.CscBasedNoSortNaiveSpmspv_keyval(alpha, y_nnz, d_y, d_y_key_output, d_y_val_output);
    time3 += timer.Stop();
  }
  time3 /= 1 /*NUM_RUN*/;
  std::cout << "naive spmspv time (sparse output) = " << time3 << " ms." << std::endl << std::endl;
  fout << time3 << " ";
  //cudaMemcpy(y_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost); 
#endif

#if 1
  time4 = 0.0;
  for (int i = 0; i < 1/*NUM_RUN*/; i += 1) {
    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    timer.Start();
    A.CscBasedNoSortMergeSpmspv_keyval(false, alpha, y_nnz, d_y, d_y_key_output, d_y_val_output);
    time4 += timer.Stop();
  }
  time4 /= 1 /*NUM_RUN*/;
  std::cout << "merge spmspv time (sparse output) = " << time4 << " ms." << std::endl << std::endl;
  fout << time4 << " ";
  //cudaMemcpy(y_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost); 
#endif

//std::cout << "merge speedup (sort/no-sort) = " << time1/time4 << std::endl << std::endl;

// #if 0
//   time5 = 0.0;
//   for (int i = 0; i < 1 /*NUM_RUN*/; i += 1) {
//     cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
//     timer.Start();
//     //A.CscBasedMyNoSortSpmspv(false, alpha, y_nnz, d_y);
//     A.CscBasedMyNoSortSpmspv(false, alpha, y_nnz, d_y, d_y_key_output, d_y_val_output);
//     //A.dense2sparse_outer(m, d_y, y_nnz, d_y_key_output, d_y_val_output);
//     time5 += timer.Stop();
//   }
//   time5 /= 1/*NUM_RUN*/;
//   std::cout << "my spmspv time (dense output) " << time5 << " ms." << std::endl << std::endl;
// #endif

#if 1
  time6 = 0.0;
  for (int i = 0; i < 1 /*NUM_RUN*/; i += 1) {
    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    timer.Start();
    //A.CscBasedNoSortSpmspv(false, alpha, y_nnz, d_y);
    //A.CscBasedNoSortSpmspv(true, alpha, y_nnz, d_y);
    A.CscBasedMyNoSortSpmspv_keyval(false, alpha, y_nnz, d_y, d_y_key_output, d_y_val_output);
    //A.dense2sparse_outer(m, d_y, y_nnz, d_y_key_output, d_y_val_output);
    time6 += timer.Stop();
  }
  time6 /= 1/*NUM_RUN*/;
  std::cout << "my spmspv time (sparse output) " << time6 << " ms." << std::endl << std::endl;
  fout << time6 << std::endl << std::endl;
  //std::cout << "no sort speedup(naive/my) = " << time3/time6 << std::endl;
  //std::cout << "no sort speedup(naive/merge) = " << time3/time4 << std::endl;
  //std::cout << "no sort speedup(merge/my) = " << time4/time6 << std::endl << std::endl;

  cudaMemcpy(y_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
#endif 

#endif

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));

  if (d_x)       cudaErrCheck(cudaFree(d_x));
  if (d_x_key)   cudaErrCheck(cudaFree(d_x_key));
  if (d_x_val)   cudaErrCheck(cudaFree(d_x_val));

  if (d_y)              cudaErrCheck(cudaFree(d_y)); 
  if(d_y_key_output)    cudaErrCheck(cudaFree(d_y_key_output)); 
  if(d_y_val_output)    cudaErrCheck(cudaFree(d_y_val_output)); 

  A.deallocPreBuffer();
  A.Destroy();
#endif
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
template <typename vT=float>
void generate_random_sparse_vector(int n, int nnz, int* key, vT* value) {
  if (nnz > n) { 
    nnz = n;
  }
  //TODO: this method can be blocked!!!
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_int_distribution<> dist(0, n-1);//[0,n-1]

  // for (int j = 0; j < nnz; j++) {
  //   int dst;
  //   do {
  //     dst = dist(gen);
  //     std::cout << "dst = " << dst << std::endl;
  //   } while(key_present(dst, key, 0, j));

  //   key[j] = dst;
  //   value[j] = 1;
  // }
  for(int j = 0; j < nnz; j++){
    key[j] = j + 4;
    assert((j + 4) < n);
    value[j] = 1.0;
  }
}

#ifdef CORRECT
template<typename T>
void CheckVector(T* cpu, T* gpu, int len) {
	int flag = 1;
	for(int i = 0; i < len; i++) {
		if(cpu[i] != gpu[i]) {
			std::cout << "Err at " << i << ", cpu[i] = " << cpu[i] <<", gpu[i] = " << gpu[i] << std::endl;
      flag = 0;
      return ;
		}
	}
	if(flag == 1) 
		std::cout <<"RESULT OK" <<std::endl;
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
#endif

//TODO: format conversion need to optimize.
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
  fileToMtxCoo<VALUE_TYPE>(file_name, &mtx, true); //何时释放mtx的内存？
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

  int* x_sparse_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_sparse_key);
  VALUE_TYPE* x_sparse_val = (VALUE_TYPE* )malloc(n * 
                            sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_sparse_val);
  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * 
    sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);

  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  int* y_sparse_key =  (int* )malloc(m * sizeof(int));
  VALUE_TYPE* y_sparse_val =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  VALUE_TYPE* y_sort_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));

  CHECK_MALLOC(y_dense);
  CHECK_MALLOC(y_sort_dense);
  CHECK_MALLOC(y_sparse_key);
  CHECK_MALLOC(y_sparse_val);
 
#ifdef CORRECT
  //for serial spmv.
  VALUE_TYPE* hres =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(hres);
#endif

  // int xnnz_vec[14] = {500,1000,1500,2000,2500,3000,3500,4000,4500,
  //                     5000,5500,6000,6500,7000};  
  //int xnnz_vec[1] = {3400000};  
  //int xnnz_vec[1] = {3566908}; 
  //int xnnz_vec[7] = {1,2,3,4,5,6,7};   
  //float x_sparsity[4] = {0.001, 0.01, 0.1, 0.2};  
  float x_sparsity[7] = {0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2};  
  //float x_sparsity[1] = {0.1};
  
  std::ofstream fout;
  fout.open("./column-major-spmspv-perf.out", std::ofstream::app);
  if(!fout){
    std::cout << "file can't open" << std::endl;
    exit(1);
  }
  fout << file_name << std::endl;

  //for (int i = 0; i < 4; i++) {
  for (int i = 0; i < 6; i++) {
    //nt x_nnz = xnnz_vec[i];
    //std::cout << "i begin-1" << std::endl;
    int x_nnz = (int)(x_sparsity[i] * n);
    if(x_nnz >= n)  x_nnz = n;
    if(x_nnz <= 0) x_nnz = 1;
    printf("sparisty = %f, x_nnz = %d\n", x_sparsity[i], x_nnz);
    fout << "sparisty = " << x_sparsity[i] << ", xnnz = " << x_nnz << std::endl;
    //printf("x_nnz = %d\n", x_nnz);
    
    //std::cout << "i begin-2" << std::endl;
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));
    memset(x_dense, 0, n * sizeof(VALUE_TYPE));

    //std::cout << "i begin-3" << std::endl;
#ifdef SPARSE_X_RANDOM
    srand(time(NULL));
    generate_random_sparse_vector(n, x_nnz, x_sparse_key, x_sparse_val);
    std::sort(x_sparse_key, x_sparse_key + x_nnz);
    for(int i = 0; i < x_nnz; i++){
      x_dense[x_sparse_key[i]] = 1;
    }
#else
    for (int i = 0; i < x_nnz; i++) {
      //x_sparse[i].key = i;
      //x_sparse[i].val = 1.;
      x_sparse_key[i] = i;
      x_sparse_val[i] = 1;
      x_dense[i] = 1;
    }
#endif

  //std::cout << "i begin-4" << std::endl;
#ifdef CORRECT
    //serial spmv computation.
    memset(hres, 0, m * sizeof(VALUE_TYPE));//
    serialspmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, hres, 1.0);
#endif

    VALUE_TYPE alpha = 1.0;
    int cu_y_nnz = 0;
    memset(y_dense, 0, m * sizeof(VALUE_TYPE));
    memset(y_sort_dense, 0, m * sizeof(VALUE_TYPE));
    TestCscSpmspv(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                              x_nnz, x_sparse_key, x_sparse_val, 
                              x_dense,
                              &cu_y_nnz, 
                              y_sparse_key, y_sparse_val, y_dense, y_sort_dense, alpha/*, fout*/);
    //printf("finished\n");
#ifdef CORRECT
  std::cout << "******SORT-SPMSPV-CORRECTNESS:******" << std::endl;
  CheckVector<VALUE_TYPE>(hres, y_sort_dense, m);
  // std::cout << "***NO-SORT-SPMSPV-CORRECTNESS:*** " << std::endl;
  // CheckVector<VALUE_TYPE>(hres, y_dense, m);
#endif
  std::cout << std::endl << std::endl;

  //std::cout << "i finished" << std::endl;
  }//end of iteration.

  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);
  if (x_dense)      free(x_dense);
  
  if (y_dense)         free(y_dense);
  if (y_sort_dense)    free(y_sort_dense);
  if (y_sparse_key)    free(y_sparse_key);
  if (y_sparse_val)    free(y_sparse_val);

#ifdef CORRECT
  if(hres) free(hres);
#endif

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
