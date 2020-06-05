// This is the test source file of spmspv.

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

// #ifdef CSR5
// #include "CSR5_cuda/anonymouslib_cuda.h"
// #endif

// #ifdef MERGE_SPMV
// using namespace cub;
// CachingDeviceAllocator  g_allocator(true);
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

void TransferData2Device(int* csr_row, int* csr_col, 
                         VALUE_TYPE* csr_val,VALUE_TYPE* x,
                         int* d_row, int* d_col, 
                         VALUE_TYPE* d_val,VALUE_TYPE* d_x, 
                         int m, int n, int mat_nnz) {

  checkCudaErrors(cudaMemcpy(d_row, csr_row, (m + 1) * sizeof(int),   
                   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_col, csr_col, mat_nnz * sizeof(int),   
                   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_val, csr_val, mat_nnz * sizeof(VALUE_TYPE),   
                   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE),   
                   cudaMemcpyHostToDevice));
}
    
void TransferVec2Host(VALUE_TYPE* y, VALUE_TYPE* d_y, int len) {
    
  checkCudaErrors(cudaMemcpy(y, d_y, len * sizeof(VALUE_TYPE), 
                  cudaMemcpyDeviceToHost));
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

  cudaEvent_t st, ed;
  float tm = 0;
  cudaEventCreate(&st);
  cudaEventCreate(&ed);
  cudaEventRecord(st, 0);

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
  cudaEventRecord(ed, 0);
  cudaEventSynchronize(ed);
  //unit: ms.
  cudaEventElapsedTime(&tm, st, ed);
  cudaEventDestroy(st);
  cudaEventDestroy(ed);

  std::cout << "csr2csc time = " << tm <<" ms." << std::endl;

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

void TestSerialSpmv(int m, int n, int mat_nnz,
                   int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                   VALUE_TYPE* x_dense, VALUE_TYPE* y_ref, VALUE_TYPE alpha,
                   double gb, double gflop) {
  SpmspvTimer ref_timer;
  ref_timer.Start();

  int ref_iter = 1;
  for (int iter = 0; iter < ref_iter; iter++) {
    for (int i = 0; i < m; i++) {
      VALUE_TYPE sum = 0;
      for (int j = csr_row[i]; j < csr_row[i+1]; j++)
        sum += x_dense[csr_col[j]] * csr_val[j] * alpha;
      y_ref[i] = sum;
    }
  }
  double ref_time = ref_timer.Stop() / (double)ref_iter;
  std::cout << "cpu sequential time = " << ref_time
           << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
           << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  
           << " GFlops." << std::endl << std::endl;
}

int TestCusparseCsrSpmv(int m, int n, int mat_nnz,
                        int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                        VALUE_TYPE* x, VALUE_TYPE* y, VALUE_TYPE alpha, 
                        double gb, double gflop) {  
  cusparseHandle_t sparse_handle;
  VALUE_TYPE beta = 0.0;
 
  int* d_row = NULL;
  int* d_col = NULL;
  VALUE_TYPE* d_val = NULL;
  VALUE_TYPE* d_x = NULL;
  VALUE_TYPE* d_y = NULL;

  checkCudaErrors(cudaMalloc((void** )&d_row, (m + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void** )&d_col, mat_nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void** )&d_val,  mat_nnz * sizeof(VALUE_TYPE)));
  
  checkCudaErrors(cudaMalloc((void** )&d_x,  n * sizeof(VALUE_TYPE)));
  checkCudaErrors(cudaMalloc((void** )&d_y,  m * sizeof(VALUE_TYPE)));

  TransferData2Device(csr_row, csr_col, csr_val, x, 
                          d_row, d_col, d_val, d_x, m, n, mat_nnz);
 
  CUSP_CALL(cusparseCreate(&sparse_handle));
  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  checkCudaErrors(cudaDeviceSynchronize());
  
  cudaEvent_t st, ed;
  float tm = 0, tm_tmp;
  for (int _ = 0; _ < NUM_RUN; _++) {
    tm_tmp = 0;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st, 0); 
#ifdef DOUBLE
    CUSP_CALL(cusparseDcsrmv(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             m, n, mat_nnz, &alpha, descr, d_val,
                             d_row, d_col, d_x, &beta, d_y));
#else
    CUSP_CALL(cusparseScsrmv(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             m, n, mat_nnz, &alpha, descr, d_val,
                             d_row, d_col, d_x, &beta, d_y));
#endif
    cudaEventRecord(ed, 0);
    cudaEventSynchronize(ed);
    cudaEventElapsedTime(&tm_tmp, st, ed);
    cudaEventDestroy(st);
    cudaEventDestroy(ed);
    tm += tm_tmp;
  }

  double cusparse_time = tm /NUM_RUN;
  std::cout << "cusparse_csrmv time = " << cusparse_time
       << " ms. Bandwidth = " << gb/(1.0e+6 * cusparse_time)
       << " GB/s. GFlops = " << gflop/(1.0e+6 * cusparse_time)  
       << " GFlops." << std::endl;

  TransferVec2Host(y, d_y, m);
  
  checkCudaErrors(cudaFree(d_row));
  checkCudaErrors(cudaFree(d_col));
  checkCudaErrors(cudaFree(d_val));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

return 0;
}

//#ifdef CUSPARSE_HYBRID_SPMV
int TestCusparseHybridSpmv(int m, int n, int mat_nnz,
                        int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                        VALUE_TYPE* x, VALUE_TYPE* y, VALUE_TYPE alpha, 
                        double gb, double gflop) {  
  VALUE_TYPE beta = 0.0;
 
  int* d_row = NULL;
  int* d_col = NULL;
  VALUE_TYPE* d_val = NULL;
  VALUE_TYPE* d_x = NULL;
  VALUE_TYPE* d_y = NULL;

  checkCudaErrors(cudaMalloc((void** )&d_row, (m + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void** )&d_col, mat_nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void** )&d_val,  mat_nnz * sizeof(VALUE_TYPE)));
  
  checkCudaErrors(cudaMalloc((void** )&d_x,  n * sizeof(VALUE_TYPE)));
  checkCudaErrors(cudaMalloc((void** )&d_y,  m * sizeof(VALUE_TYPE)));

  TransferData2Device(csr_row, csr_col, csr_val, x, 
                      d_row, d_col, d_val, d_x, m, n, mat_nnz);
  
  cusparseHandle_t sparse_handle;
  CUSP_CALL(cusparseCreate(&sparse_handle));
  cusparseMatDescr_t mat_desc;
  CUSP_CALL(cusparseCreateMatDescr(&mat_desc));
  cusparseSetMatType(mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(mat_desc, CUSPARSE_INDEX_BASE_ZERO);

   // Construct Hyb matrix
  cusparseHybMat_t hyb_desc;
  CUSP_CALL(cusparseCreateHybMat(&hyb_desc));

  checkCudaErrors(cudaDeviceSynchronize());

  float setup_ms = 0.0;
  SpmspvTimer timer;
  timer.Start();
#ifdef DOUBLE
    CUSP_CALL(cusparseDcsr2hyb(
        sparse_handle,
        m, n,
        /*params.num_rows, params.num_cols,*/
        mat_desc,
        d_val, d_row, d_col,
        /*params.d_values, params.d_row_end_offsets, params.d_column_indices,*/
        hyb_desc,
        0,
        CUSPARSE_HYB_PARTITION_AUTO));
#else
    CUSP_CALL(cusparseScsr2hyb(
        sparse_handle,
        m, n,
        /*params.num_rows, params.num_cols,*/
        mat_desc,
        d_val, d_row, d_col,
        /*params.d_values, params.d_row_end_offsets, params.d_column_indices,*/
        hyb_desc,
        0,
        CUSPARSE_HYB_PARTITION_AUTO));
#endif

    cudaDeviceSynchronize();
    setup_ms = timer.Stop();
    std::cout << "setup_ms = " << setup_ms << std::endl;

    // Reset input/output vector y
    // CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
#ifdef DOUBLE
     CUSP_CALL(cusparseDhybmv(
        sparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc,
        hyb_desc,
        d_x, &beta, d_y));
#else
    CUSP_CALL(cusparseShybmv(
        sparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc,
        hyb_desc,
        d_x, &beta, d_y));
#endif
    // Timing
    float elapsed_ms = 0.0;
    timer.Start();
    for (int it = 0; it < NUM_RUN; ++it) {
#ifdef DOUBLE
      CUSP_CALL(cusparseDhybmv(
            sparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_desc,
            hyb_desc,
            d_x, &beta, d_y));
#else
      CUSP_CALL(cusparseShybmv(
          sparse_handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha, mat_desc,
          hyb_desc,
          d_x, &beta, d_y));
#endif
    }

    elapsed_ms = timer.Stop()/NUM_RUN;
    std::cout << "cusparse_hybrid_mv time = " << elapsed_ms
       << " ms. Bandwidth = " << gb/(1.0e+6 *elapsed_ms)
       << " GB/s. GFlops = " << gflop/(1.0e+6 * elapsed_ms)  
       << " GFlops." << std::endl;

    // Cleanup
  CUSP_CALL(cusparseDestroyHybMat(hyb_desc));
  CUSP_CALL(cusparseDestroyMatDescr(mat_desc));

  TransferVec2Host(y, d_y, m);
  
  checkCudaErrors(cudaFree(d_row));
  checkCudaErrors(cudaFree(d_col));
  checkCudaErrors(cudaFree(d_val));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
return 0;
}
//#endif

//#ifdef CSR5
int TestCsr5Spmv(int m, int n, int mat_nnz,
                 int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                 VALUE_TYPE* x, VALUE_TYPE* y, VALUE_TYPE alpha) {
  int err = 0;
  cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " 
            << deviceProp.name << ", " << " @ " 
            << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

  double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  double gflop = SpmvGetFlops<int>(mat_nnz);

  // Define pointers of matrix A, vector x and y
  int* d_csr_row;
  int* d_csr_col;
  VALUE_TYPE* d_csr_val;
  VALUE_TYPE* d_x;
  VALUE_TYPE* d_y;

  // Matrix A
  checkCudaErrors(cudaMalloc((void **)&d_csr_row, (m + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csr_col, mat_nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csr_val, mat_nnz * sizeof(VALUE_TYPE)));

  checkCudaErrors(cudaMemcpy(d_csr_row, csr_row, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

  // Vector x
  checkCudaErrors(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
  checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE), 
                  cudaMemcpyHostToDevice));

  // Vector y
  checkCudaErrors(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(mat_nnz, m, n);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  //std::cout << "inputCSR err = " << err << std::endl;

  err = A.set_x(d_x); // you only need to do it once!
  //std::cout << "setX err = " << err << std::endl;

  A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
  // warmup device
  A.warmup();

  SpmspvTimer asCSR5_timer;
  asCSR5_timer.Start();
  err = A.CSR5Preprocess();
  std::cout << "CSR->CSR5 time = " << asCSR5_timer.Stop() 
                << " ms." << std::endl;
  //std::cout << "asCSR5 err = " << err << std::endl;

  //check correctness by running 1 time
  err = A.spmv(alpha, d_y);
  //std::cout << "spmv err = " << err << std::endl;
  checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), 
                 cudaMemcpyDeviceToHost));

  // warm up by running 50 times
  if (NUM_RUN) {
      for (int i = 0; i < 50; i++)
          err = A.spmv(alpha, d_y);
  }

  err_cuda = cudaDeviceSynchronize();

  SpmspvTimer CSR5Spmv_timer;
  CSR5Spmv_timer.Start();

  // time spmv by running NUM_RUN times
  for (int i = 0; i < NUM_RUN; i++)
      err = A.spmv(alpha, d_y);
  err_cuda = cudaDeviceSynchronize();

  double CSR5Spmv_time = CSR5Spmv_timer.Stop() / (double)NUM_RUN;

  if (NUM_RUN)
      std::cout << "CSR5-based SpMV time = " << CSR5Spmv_time
           << " ms. Bandwidth = " << gb/(1.0e+6 * CSR5Spmv_time)
           << " GB/s. GFlops = " << gflop/(1.0e+6 * CSR5Spmv_time)  
           << " GFlops." << std::endl;

  A.CSR5Postprocess();
  A.Destroy();

  checkCudaErrors(cudaFree(d_csr_row));
  checkCudaErrors(cudaFree(d_csr_col));
  checkCudaErrors(cudaFree(d_csr_val));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  return err;
}
//#endif


// #ifdef MERGE_SPMV
// #include <cub/device/device_spmv.cuh>
// #include <cub/util_allocator.cuh>
// #include <cub/iterator/tex_ref_input_iterator.cuh>
// using namespace cub;

// int TestMergeSpmv(int m, int n, int mat_nnz,
//                  int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
//                  VALUE_TYPE* x, VALUE_TYPE* y, VALUE_TYPE alpha) {
//   int err = 0;
//   cudaError_t err_cuda = cudaSuccess;

//   int device_id = 0;
//   cudaSetDevice(device_id);
//   cudaDeviceProp deviceProp;
//   cudaGetDeviceProperties(&deviceProp, device_id);
//   float device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000; 
//   std::cout <<"bus width = " << deviceProp.memoryBusWidth 
//             << "deviceProp.memoryClockRate" << deviceProp.memoryClockRate << std::endl;
//   std::cout <<"peak bandwidth = " << device_giga_bandwidth << std::endl;
//   std::cout << "Device [" <<  device_id << "] " 
//             << deviceProp.name << ", " << " @ " 
//             << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;

//   double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
//   double gflop = SpmvGetFlops<int>(mat_nnz);

//   // Define pointers of matrix A, vector x and y
//   int* d_csr_row;
//   int* d_csr_col;
//   VALUE_TYPE* d_csr_val;
//   VALUE_TYPE* d_x;
//   VALUE_TYPE* d_y;

//   // Matrix A
//   checkCudaErrors(cudaMalloc((void **)&d_csr_row, (m + 1) * sizeof(int)));
//   checkCudaErrors(cudaMalloc((void **)&d_csr_col, mat_nnz * sizeof(int)));
//   checkCudaErrors(cudaMalloc((void **)&d_csr_val, 
//                   mat_nnz * sizeof(VALUE_TYPE)));

//   checkCudaErrors(cudaMemcpy(d_csr_row, csr_row, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
//   checkCudaErrors(cudaMemcpy(d_csr_col, csr_col, mat_nnz * sizeof(int), cudaMemcpyHostToDevice));
//   checkCudaErrors(cudaMemcpy(d_csr_val, csr_val, mat_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

//   // Vector x
//   checkCudaErrors(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));
//   checkCudaErrors(cudaMemcpy(d_x, x, n * sizeof(VALUE_TYPE), 
//                   cudaMemcpyHostToDevice));

//   // Vector y
//   checkCudaErrors(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
//   checkCudaErrors(cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE)));


//   float setup_ms = 0.0;
//   SpmspvTimer timer;
//   timer.Start();
//   // Allocate temporary storage
//   size_t temp_storage_bytes = 0;
//   void *d_temp_storage = NULL;

//   // Get amount of temporary storage needed
//   CubDebugExit(DeviceSpmv::CsrMV(
//       d_temp_storage, temp_storage_bytes,
//       d_csr_val, d_csr_row, d_csr_col,
//       /*params.d_values, params.d_row_end_offsets, params.d_column_indices,*/
//       d_x, d_y,
//       /*params.d_vector_x, params.d_vector_y,*/
//       /*params.num_rows, params.num_cols, params.num_nonzeros,*/
//       m, n, mat_nnz,
//       (cudaStream_t) 0, false));

//   // Allocate
//   CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
//                temp_storage_bytes));
//   setup_ms = timer.Stop() / (double)NUM_RUN;
//   std::cout << "setup_ms = " << setup_ms << std::endl;
//   // Reset input/output vector y
//   // CubDebugExit(cudaMemcpy(params.d_vector_y, vector_y_in, sizeof(VALUE_TYPE) * params.num_rows, cudaMemcpyHostToDevice));

//   // Warmup
//   CubDebugExit(DeviceSpmv::CsrMV(
//       d_temp_storage, temp_storage_bytes,
//        d_csr_val, d_csr_row, d_csr_col,
//       /*params.d_values, params.d_row_end_offsets, params.d_column_indices,*/
//       d_x, d_y,
//       /*params.d_vector_x, params.d_vector_y,*/
//       /*params.num_rows, params.num_cols, params.num_nonzeros,*/
//       m, n, mat_nnz,
//       (cudaStream_t) 0, 1/*!g_quiet*/));

//   // if (!g_quiet)
//   // {
//   //     int compare = CompareDeviceResults(reference_vector_y_out, params.d_vector_y, params.num_rows, true, g_verbose);
//   //     printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);
//   // }

//   // Timing
//   //SpmspvTimer timer;
//   float elapsed_ms = 0.0;

//   timer.Start();
//   for(int it = 0; it < NUM_RUN; ++it)
//   {
//       CubDebugExit(DeviceSpmv::CsrMV(
//           d_temp_storage, temp_storage_bytes,
//           d_csr_val, d_csr_row, d_csr_col,
//           /*params.d_values, params.d_row_end_offsets, params.d_column_indices,*/
//           d_x, d_y,
//           /*params.d_vector_x, params.d_vector_y,*/
//           /*params.num_rows, params.num_cols, params.num_nonzeros,*/
//           m, n, mat_nnz,
//           (cudaStream_t) 0, false));
//   }
//   elapsed_ms = timer.Stop() / (double)NUM_RUN;
//   std::cout << "Merge SpMV time = " << elapsed_ms
//            << " ms. Bandwidth = " << gb/(1.0e+6 * elapsed_ms)
//            << " GB/s. GFlops = " << gflop/(1.0e+6 * elapsed_ms)  
//            << " GFlops." << std::endl;

//   checkCudaErrors(cudaFree(d_csr_row));
//   checkCudaErrors(cudaFree(d_csr_col));
//   checkCudaErrors(cudaFree(d_csr_val));
//   checkCudaErrors(cudaFree(d_x));
//   checkCudaErrors(cudaFree(d_y));

//   return err;
// }
// #endif

//NOTE: arr[len+1]
void ExclusiveScan(int* arr, int len) {
    int old_val, new_val;
    old_val = arr[0];
    arr[0] = 0;
    for (int i = 1; i <= len; i++) {
        new_val = arr[i];
        arr[i] = old_val + arr[i-1];
        old_val = new_val;
    }
}

int main(int argc, char** argv) {
  int m, n, mat_nnz;
  int* csr_row = NULL;
  int* csr_col = NULL;
  VALUE_TYPE* csr_val = NULL;

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

  VALUE_TYPE* x = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x);
  for (int i = 0; i < n; i++) {
    x[i] = 1;
  }

  VALUE_TYPE* y = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  VALUE_TYPE* y_ref = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));

  CHECK_MALLOC(y);
  CHECK_MALLOC(y_ref);

  double gb = SpmvGetBytes<int, VALUE_TYPE>(m, mat_nnz);
  double gflop = SpmvGetFlops<int>(mat_nnz);

  VALUE_TYPE alpha = 1.0;

  std::cout << "-------------------------------" << std::endl;

// #ifdef STATISTIC
//   TestMatInfo(m, n, mat_nnz,
//                  csr_row, csr_col, csr_val);
//   std::cout << "-------------------------------"  << std::endl;
// #endif

//#ifdef SPMV_SERIAL
  // TestSerialSpmv(m, n, mat_nnz,
  //                 csr_row, csr_col, csr_val,
  //                 x, y_ref, alpha, gb, gflop);
  //Vec2File(y_ref, m, "../../res/spmv_serial_ref.vec");
 
  std::cout << "-------------------------------"  << std::endl;
//#endif

//#ifdef CUSPARSE
  TestCusparseCsrSpmv(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                        x, y, alpha, gb, gflop);
  //Vec2File(y, m, "../../res/spmv_cusparse_csrspmv.vec");

  std::cout << "-------------------------------"  << std::endl;
//#endif
 
//#ifdef CUSPARSE_HYBRID_SPMV
  TestCusparseHybridSpmv(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                        x, y, alpha, gb, gflop);
  Vec2File(y, m, "../../res/cusparse_hybrid_spmv.vec");

  std::cout << "-------------------------------"  << std::endl;
//#endif

//#ifdef  CSR5 
  // TestCsr5Spmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x, y, alpha);
  // Vec2File(y, m, "../../res/spmv_csr5.vec");
  // std::cout << "-------------------------------"  << std::endl;
//#endif

// #ifdef  MERGE_SPMV 
//   TestMergeSpmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x, y, alpha);
//   Vec2File(y, m, "../../res/spmv_mergevec");
//   std::cout << "-------------------------------"  << std::endl;
// #endif

#ifndef READ_SVM_DATA
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#endif

  if (x)         free(x);
  if (y)         free(y);
  if (y_ref)     free(y_ref);
  return 0;
}
