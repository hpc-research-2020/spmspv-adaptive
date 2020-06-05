// This is used to TestCsr5Spmspv

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

// #ifdef CSR5_SPMSPV
// #include "spmspv/csr5-spmspv/anonymouslib_cuda.h"
// #endif

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

#ifdef CSR5_SPMSPV
int TestCsr5Spmspv(int m, int n, int mat_nnz,
                  int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                  int x_nnz, int* x_key, VALUE_TYPE* x_dense_val, 
                  VALUE_TYPE* y, VALUE_TYPE alpha)
{
    int err = 0;
    cudaError_t err_cuda = cudaSuccess;

    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    std::cout << "Device [" <<  device_id << "] " << deviceProp.name 
         << ", " << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " 
         << std::endl;

    double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
    double gflop = SpmvGetFlops<int>(mat_nnz);

    // Define pointers of matrix A, vector x and y
    int* d_csr_row = NULL;
    int* d_csr_col = NULL;
    VALUE_TYPE* d_csr_val = NULL;
    VALUE_TYPE* d_x = NULL;
    VALUE_TYPE* d_y = NULL;
    int* d_x_key = NULL;

    // Matrix A
    cudaErrCheck(cudaMalloc((void** )&d_csr_row, (m + 1) * sizeof(int)));
    cudaErrCheck(cudaMalloc((void** )&d_csr_col, mat_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void** )&d_csr_val, 
    			 mat_nnz * sizeof(VALUE_TYPE)));
    //Added by 
    cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
    // Vector x
    cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));

    cudaErrCheck(cudaMemcpy(d_csr_row, csr_row, (m+1) * sizeof(int),   
            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int),   
            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_val, csr_val, 
    			mat_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    //Added by 
    cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x, x_dense_val, n * sizeof(VALUE_TYPE), 
                            cudaMemcpyHostToDevice));

    // Vector y
    cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
    cudaErrCheck(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));

    SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
    err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
    //err = A.setX(d_x); // you only need to do it once!
    err = A.set_bitvector_x(d_x_key, d_x, x_nnz);
    //cout << "setX err = " << err << endl;

    A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

    // warmup device
    A.warmup();

    //csr->csr5 time.
    SpmspvTimer asCSR5_timer;
    asCSR5_timer.Start();
    err = A.CSR5Preprocess();
    std::cout << "CSR->CSR5 time = " << asCSR5_timer.Stop() 
    		<< " ms." << std::endl;
    //cout << "asCSR5 err = " << err << endl;

    // check correctness by running 1 time
    err = A.csr5spmspv(alpha, d_y);
    //cout << "spmv err = " << err << endl;
    cudaErrCheck(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), 
                            cudaMemcpyDeviceToHost));
    
    // warm up by running 50 times
    if (NUM_RUN) {
        for (int i = 0; i < 50; i++)
            err = A.csr5spmspv(alpha, d_y);
    }
    err_cuda = cudaDeviceSynchronize();

    //time spmv by running NUM_RUN times
    SpmspvTimer timer;
    timer.Start();
    for (int i = 0; i < NUM_RUN; i++)
        err = A.csr5spmspv(alpha, d_y);
    err_cuda = cudaDeviceSynchronize();
    double time = timer.Stop() / (double)NUM_RUN;

    if (NUM_RUN)
        std::cout << "CSR5-based Spmspv time = " << time
             << " ms. Bandwidth = " << gb/(1.0e+6 * time)
             << " GB/s. GFlops = " << gflop/(1.0e+6 * time)  
             << " GFlops." << std::endl;
    
    //将格式在转换为csr格式。
    A.CSR5Postprocess();
    
    A.Destroy();
    cudaErrCheck(cudaFree(d_csr_row));
    cudaErrCheck(cudaFree(d_csr_col));
    cudaErrCheck(cudaFree(d_csr_val));
    cudaErrCheck(cudaFree(d_x));
    cudaErrCheck(cudaFree(d_y));
    cudaErrCheck(cudaFree(d_x_key));
    return err;
}
#endif

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

//randomly produce sparse vector
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
  
  //read data from file: svm data or matrix market data.
#ifndef READ_SVM_DATA
  MTX<VALUE_TYPE> mtx;
  //row is in order, and coluln ids in one row is also in order
  //when to free mtx?
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

  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);
  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(y_dense);

  int* x_sparse_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_sparse_key);
  VALUE_TYPE* x_sparse_val = (VALUE_TYPE* )malloc(n * 
                                sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_sparse_val);

  int xnnz_vec[63] = {1000,3000,5000,7000,9000,11000,13000,16000,19000,
                      22000,25000,28000,31000,34000,37000,40000,50000,
                      64000,98000,123000,166000,240000,260000,280000,
                      315000,340000,380000,420000,460000,484000,540000,
                      580000,616000,650000,700000,735000,853000,955000,
                      1129000,1213000,1366000,1538000,1729000,1902000,
                      2061000,2252000,2403000,2563000,2740000,2899000,
                      3069000,3200000,3400000,3600000,3800000,4000000,
                      4200000,4400000,4600000,4800000,5000000,5200000,
                      5363260};
  // int xnnz_vec[61] = {1000,3000,5000,7000,9000,11000,13000,16000,19000,
  //                     22000,25000,28000,31000,34000,37000,40000,50000,
  //                     64000,98000,123000,166000, 240000,260000,280000,
  //                     315000,340000,380000,420000,460000,484000,540000,
  //                     580000,616000,650000,700000,735000,853000,955000,
  //                     1129000,1213000,1366000,1538000,1729000,1902000,
  //                     2061000,2252000,2403000,2563000,2740000,2899000,
  //                     3069000,3200000,3400000,3600000,3800000,4000000,
  //                     4200000,4400000,4600000,4800000,4847571};
  // //amazon and coAuthorsDBLP
  // int xnnz_vec[29] = {1000,3000,5000,7000,9000,11000,13000,16000,19000,
  //                     22000, 25000,28000,31000,34000,37000,40000,50000,
  //                     64000,98000,123000,166000,240000,260000,280000,
  //                     299067,315000,340000,380000,400727};
  // //delaunay_n13
  // int xnnz_vec[17]={500,1000,1500,2000,2500,3000,3500,4000,4500,5000,
  //                   5500,6000,6500,7000,7500,8000,8192};
  // data mining datasets.                  
  // int xnnz_vec[19]={1000,3000,5000,7000,9000,11000,13000,16000,19000,
  //                   22000,25000,28000,31000,34000,37000,40000,50000,
  //                   64000,98000};                 
   // int xnnz_vec[22] = {600000,6500000,7000000,7500000,8000000,8500000,
   //                    9000000,9500000,10500000,11000000,11500000,12000000,
   //                    12500000,13000000,13500000,14000000,14500000,
   //                    15000000,15500000,16000000,16500000,17000000};
  //int xnnz_vec[3] = {5400000,5558326,6000000};
  for (int i = 0; i < 63; i++) {
  	int x_nnz = xnnz_vec[i];
    if(x_nnz >= n)  x_nnz = n;
    printf("x_nnz = %d\n", x_nnz);
    memset(x_dense, 0, n * sizeof(VALUE_TYPE));
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));
    memset(y_dense, 0, m * sizeof(VALUE_TYPE));
    VALUE_TYPE alpha = 1.0;
    //assign input vector.
#ifdef SPARSE_X_RANDOM
  	srand(time(NULL));
  	generate_random_sparse_vector(n, x_nnz, x_sparse_key, x_sparse_val);
  	for (int i = 0; i < x_nnz; i++) {
    	x_dense[i] = 1;
  	}
#else
    //produce sparse vector in the order.
    for (int i = 0; i < x_nnz; i++) {
      x_sparse_key[i] = i;
      x_sparse_val[i] = 1;
      x_dense[i] = 1;
    }
#endif
    std::cout << "-------------------------------" << std::endl;

#ifdef CSR5_SPMSPV
    //memset(y_dense, 0, m * sizeof(VALUE_TYPE));
    std::cout << "begin TestCsr5Spmspv()" << std::endl;
    TestCsr5Spmspv(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                     x_nnz, x_sparse_key, x_dense, y_dense, alpha);
    //Vec2File(y_dense, m, "../../res/csr5_smsv.vec");
#endif
    std::cout << "-------------------------------"  << std::endl;
  }// end of for.
  
  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);
#ifndef READ_SVM_DATA
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#endif
  if (x_dense)   free(x_dense);
  if (y_dense)   free(y_dense);
  return 0;
}
