// This is used to TestCscNoSortOrBucketSpmspv: my work.

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

// void PrintSparseVec(SparseVec* y_sparse, int* y_sparse_key, 
//                     VALUE_TYPE* y_sparse_val, int len) {
//   std::cout << "print SparseVec" << std::endl;
//   for (int i = 0; i < len; i++) {
//     //std::cout << y_sparse[i].key << "," << y_sparse[i].val << std::endl;
//     std::cout << y_sparse_key[i] << "," << y_sparse_val[i] << std::endl;
//   }
// }

// void SparseVec2DenseVec(VALUE_TYPE* y_dense, int len, 
//                         SparseVec* y_sparse, 
//                         int* y_sparse_key, VALUE_TYPE* y_sparse_val,
//                         int y_nnz) {
//   memset(y_dense, 0, len * sizeof(VALUE_TYPE));
//   for (int i = 0; i < y_nnz; i++) {
//     //y_dense[y_sparse[i].key] = y_sparse[i].val;
//     y_dense[y_sparse_key[i]] = y_sparse_val[i];
//   }
// }
// //TODO: to optimize
// void SparseVec2DenseVecSeparate(VALUE_TYPE* y_dense, int len, 
//                           int* y_sparse_inx, VALUE_TYPE* y_sparse_val, 
//                           int y_nnz) {
//   memset(y_dense, 0, len * sizeof(VALUE_TYPE));
//   for (int i = 0; i < y_nnz; i++) {
//     y_dense[y_sparse_inx[i]] = y_sparse_val[i];
//   }
// }

// //TODO: to optimize
// void DenseVec2SparseVec(VALUE_TYPE* y_dense, int len, 
//                         SparseVec* y_sparse, 
//                         int* y_sparse_key, VALUE_TYPE* y_sparse_val,
//                         int y_nnz) {
  
//   int inx = 0;
//   for (int i = 0; i < len; i++) {
//     //if(y_dense[i] != 0){
//     if (IS_DOUBLE_ZERO(y_dense[i])) {
//       // y_sparse[inx].key = i;
//       // y_sparse[inx].val = y_dense[i];
//       y_sparse_key[inx] = i;
//       y_sparse_val[inx] = y_dense[i];
//       inx++;
//     }
//   }
// }

// void Vec2File(VALUE_TYPE* y, int len, const char* file_name) {

//   std::ofstream fout;
//   fout.open(file_name);
//   if (!fout.is_open()) {
//       std::cout << "open file " << file_name << " failed." << std::endl;
//       exit(1);
//     }
   
//   //std::cout << "result:" << std::endl;
//   for (size_t i = 0; i < len; i++) {
//     fout << y[i] << std::endl;
//   }
//   fout.close();
// }
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


void TestCsr2Coo(int m, int mat_nnz,
                 int* csr_row, int* coo_row) {

  int* d_csr_row = NULL;
  int* d_coo_row = NULL;
  
  checkCudaErrors(cudaMalloc((void** )&d_csr_row,  (m + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void** )&d_coo_row, mat_nnz * sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_csr_row, csr_row, (m + 1) * sizeof(int),   
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

  CUSP_CALL(cusparseXcsr2coo(sparse_handle, 
            d_csr_row, mat_nnz, m, d_coo_row, 
            CUSPARSE_INDEX_BASE_ZERO));

  cudaEventRecord(ed, 0);
  cudaEventSynchronize(ed);
  //unit: ms.
  cudaEventElapsedTime(&tm, st, ed);
  cudaEventDestroy(st);
  cudaEventDestroy(ed);

  std::cout << "csr2coo time = " << tm <<" ms." << std::endl;

  checkCudaErrors(cudaMemcpy(coo_row, d_coo_row, mat_nnz * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_csr_row));
  checkCudaErrors(cudaFree(d_coo_row));
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

int TestCscNoSortOrBucketSpmspv(int m, int n, int mat_nnz,
                    int* csr_row, int* csr_col, 
                    VALUE_TYPE* csr_val,
                    int x_nnz, SparseVec* x,
                    int* x_key, VALUE_TYPE* x_val,
                    int* y_nnz, SparseVec* y,
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
  std::cout << "DEBUG: xnnz = " << x_nnz << std::endl;

#if 1
  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  A.allocPreBuffer();

  err = A.InputCsr(mat_nnz, csr_row, csr_col, csr_val);
  std::cout << "DEBUG: InputCsr err = " << err << std::endl;

  // you only need to do it once!
  err = A.set_sparse_x(/*x,*/ x_key, x_val, x_nnz); 
  std::cout << "DEBUG: set_sparse_x err = " << err << std::endl;

  //int group_num = 2;
  //err = A.ToBcsc(group_num)；
  err = A.ToCsc();
  std::cout << "DEBUG: ToCsc err = " << err << std::endl;

  // SpmspvTimer timer;
  //  timer.Start();
#ifdef CUDA_BUCKET_METHOD
    //for(int buckets = 32; buckets <= 8192; buckets *= 2){
    for (int buckets = 4; buckets <= 256; buckets *= 2) {
    //for(int buckets = 4; buckets <= 4; buckets *= 2){
      err = A.CscBasedBucketSpmspv(alpha, y_nnz, y_dense, buckets);
    }
#else
    //err = A.spmspv_csc(alpha, y_nnz, y, 1);
    err = A.CscBasedNoSortSpmspv(alpha, y_nnz, y_dense);
#endif
  
  //  double time = timer.Stop();
  // cout << "smsv cuda time = " << time << "ms." << endl;
  std::cout << "DEBUG: spmspv_csc err = " << err << std::endl;

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

  SparseVec* x_sparse = (SparseVec* )malloc(n * sizeof(SparseVec));
  CHECK_MALLOC(x_sparse);
  int* x_sparse_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_sparse_key);
  VALUE_TYPE* x_sparse_val = (VALUE_TYPE* )malloc(n * 
                            sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_sparse_val);


  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));//TODO
  SparseVec* y_sparse = (SparseVec* )malloc(m * sizeof(SparseVec));//TODO
  int* y_sparse_key =  (int* )malloc(m * sizeof(int));
  VALUE_TYPE* y_sparse_val =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));

  CHECK_MALLOC(y_dense);
  CHECK_MALLOC(y_sparse);
  CHECK_MALLOC(y_sparse_key);
  CHECK_MALLOC(y_sparse_val);

#ifdef READ_SVM_DATA
  //read sparse vector from file
  char* svm_file_name = "/home/news20.out";
  int iter = 19994;

  int* ind = (int*)malloc(iter*sizeof(int));
  CHECK_MALLOC(ind);

  readSparseXInxFromFile(svm_file_name, iter, ind);

  int new_iter=0;
  removeRepetition(iter, ind, &new_iter, ind);
  std::cout <<  "iter = " << iter 
      << " ,new_iter = " << new_iter << std::endl;

  std::cout << "-------------------------------" << std::endl;

  //the iterations of spmspv
  for (int i = 0; i < new_iter; i++) {
    //the number of the nonzeroes of the ind[i]-th row.
    int x_nnz = csr_row[ind[i]+1] - csr_row[ind[i]];
    printf("ind[%d] = %d, x_nnz = %d\n", i, ind[i], x_nnz);

    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));

    extractSparseXfromMat<int, VALUE_TYPE>(ind[i], csr_row, csr_col, csr_val, 
                      m, n, mat_nnz, x_sparse_key, x_sparse_val);
#else
 

  // int xnnz_vec[63] = {1000,3000,5000,7000,9000,11000,13000,16000,19000,22000,25000,28000,31000,34000,37000,40000,50000,64000,98000,123000,166000,
  //   240000,260000,280000,315000,340000,380000,420000,460000,484000,540000,580000,616000,650000,700000,735000,853000,955000,1129000,1213000,1366000,1538000,1729000,1902000,2061000,2252000,2403000,2563000,2740000,2899000,3069000,3200000,3400000,2600000,3800000,4000000,4200000,4400000,4600000,4800000,5000000,5200000,5363260};
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
  for (int i = 0; i < 63; i++) {
    int x_nnz = xnnz_vec[i];
    if(x_nnz >= n)  x_nnz = n;
    printf("x_nnz = %d\n", x_nnz);
    
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));

#ifdef SPARSE_X_RANDOM
    srand(time(NULL));
    generate_random_sparse_vector(n, x_nnz, x_sparse_key, x_sparse_val);
#else
    for (int i = 0; i < x_nnz; i++) {
      x_sparse[i].key = i;
      x_sparse[i].val = 1.;
      x_sparse_key[i] = i;
      x_sparse_val[i] = 1;
    }
#endif

#endif //end of different methods to read sparse x.

#ifdef SMSV_CUDA
    VALUE_TYPE alpha = 1.0;
    int cu_y_nnz = 0;
    memset(y_dense, 0, m * sizeof(VALUE_TYPE));
    TestCscNoSortOrBucketSpmspv(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                              x_nnz, x_sparse, x_sparse_key, x_sparse_val, 
                              &cu_y_nnz, y_sparse,
                              y_sparse_key, y_sparse_val, y_dense, alpha);
#endif
  }//end of iteration.

  if (x_sparse)     free(x_sparse);
  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);
  
  if (y_sparse)    free(y_sparse);
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
