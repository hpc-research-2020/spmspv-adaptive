// This is used to produce the raw trainning data sets.
// Test my Spmspv performance when the nnz_x changing.

#include <iostream>
#include <string>
#include <float.h>
#include <typeinfo>
#include <limits>
#include <algorithm>
#include <vector>
#include <random>

#include <dirent.h>
#include <sys/stat.h>

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

//#define SWITCH_POINT 20000

//#define TEST_S2D_D2S
//#define CORRECT
#define OUTPUT_SPARSE

//produce sparse vector at a interval of 2000
//TODO: how to set the interval
//2000, 2000 = 400, 0000.
//4000, 2000 = 800, 0000
//8000, 2000 = 1600, 0000

//#define INTERVAL

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

  //TODO: check d_x_key and d_x_val 需要初始化为0吗？

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

#ifdef CORRECT
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
   
int Run(std::string file_name, std::string matrix_name, int m, int n, int mat_nnz,
        int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
        int* x_sparse_key, VALUE_TYPE* x_sparse_val, VALUE_TYPE* x_dense, 
        VALUE_TYPE* y_dense) {
  int err = 0;
  cudaError_t err_cuda = cudaSuccess;

  int device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);

  std::cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " 
            << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << std::endl;
  // write to file.
  std::ofstream fout;
  fout.open(file_name.c_str(), std::ofstream::app);
  if(!fout){
    std::cout << "file can't open" << std::endl;
    exit(1);
  }    
  fout << "mat_name m n nnz hola-spmv csr5-spmv selected-spmv(0/1) \n";
  fout << matrix_name << " " << m << " " << n << " " << mat_nnz << " ";

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
    
  int num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  int* d_bit_vector = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_bit_vector, num_ints * sizeof(int)));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  err = A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  
  err = A.set_vector_type(1);//current vector type is densetype.
  err = A.set_x(d_x); //
  err = A.set_bitvector(d_bit_vector, num_ints);//
 
  VALUE_TYPE* d_y = NULL; //result vec.
  cudaErrCheck(cudaMalloc((void** )&d_y, m * sizeof(VALUE_TYPE)));
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
  
  VALUE_TYPE alpha = 1.0;

  SpmspvTimer timer;
/*********select spmv.***********/
  err = A.holaPreprocess();
  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
    err = A.holaspmv(alpha, d_y);
  err_cuda = cudaDeviceSynchronize();
  double holaspmv_time = timer.Stop()/ (double)NUM_RUN;
  std::cout << "holaspmv time " << holaspmv_time << std::endl;
  fout << holaspmv_time << " ";
  cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.

#if USE_CSR5
  //for csr5 spmv.
  A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
  err = A.CSR5Preprocess();

  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
      err = A.csr5spmv(alpha, d_y);
  err_cuda = cudaDeviceSynchronize();
  double csr5spmv_time = timer.Stop() / (double)NUM_RUN;
  std::cout << "csr5spmv time " << csr5spmv_time << std::endl;
  fout << csr5spmv_time << " ";

 
  int spmv_type = 0;
  if (csr5spmv_time/holaspmv_time > 1.0) {
    spmv_type = 0;
    A.set_spmv_type(0);//
    A.CSR5Postprocess();
    std::cout << "DEBUG: select hola spmv\n";
  }else{
    spmv_type = 1;
    A.set_spmv_type(1);//
    A.holaPostprocess();
    std::cout << "DEBUG: select csr5 spmv\n";
  }
  fout << spmv_type << std::endl;
#else
fout << " 0.0 ";
 int spmv_type = 0;
 A.set_spmv_type(0);//
 fout << spmv_type << std::endl;
#endif

#ifdef CORRECT
  //for serial spmv.
  VALUE_TYPE* hres =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(hres);
#endif

  err = A.ToCSC();

#ifdef INTERVAL

#else
  //int iter = 103;//86, +17
  int iter = 86;//86, +17
  //int xnnz_vec[103] = {1,5,10,20,30,40,60,80,100,200,300,400,500,600,700,800,900,
  int xnnz_vec[86] = {1000,3000,5000,7000,9000,11000,13000,16000,19000,
                      22000,25000,28000,31000,34000,37000,40000,50000,
                      64000,98000,123000,166000,240000,260000,280000,
                      315000,340000,380000,420000,460000,484000,540000,
                      580000,616000,650000,700000,735000,853000,955000,
                      1129000,1213000,1366000,1538000,1729000,1902000,
                      2061000,2252000,2403000,2563000,2740000,2899000,
                      3069000,3200000,3400000,3600000,3800000,4000000,
                      4200000,4400000,4600000,4800000,5000000,5200000,
                      5400000,5558326,6000000,6500000,7000000,7500000,
                      8000000,8500000,9000000,9500000,10500000,11000000,
                      11500000,12000000,12500000,13000000,13500000,14000000,
                      14500000,15000000,15500000,16000000,16500000,
                      17000000}; 
#endif
  //test for csc spmspv.
  // int iter = 1;
  // int xnnz_vec[1] = {5000}; 

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (n) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (n) * sizeof(VALUE_TYPE)));

//#ifdef OUTPUT_SPARSE
  int* d_y_key_output = NULL;
  VALUE_TYPE* d_y_val_output = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_y_key_output, (m) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_y_val_output, (m) * sizeof(VALUE_TYPE)));
//#endif

  fout << "x_nnz, sparse2dense, sparse2bitarray, bin_len, max, min, xnnz/n, bin_len/nnz, xnnz_range, GM1, GM2, GM3, GM1/GM2, GM2/GM3, GM1/GM3, sort-naive-col, naive-col, sort-lb-col, lb-col, naive-rspmspv,  naive-rspmspv+s2a, lb-rspmspv, lb-rspmspv+s2a, naive-spmv, naive-spmv+s2d, lb-spmv, lb-spmv+s2d \n";

  int y_nnz = 0;
  int quit = 0;
  
  int interval_val = 2000;
  
#ifdef INTERVAL
/*
2000个一次，若共1000条，则n为200,0000；
4000个一次，若共1000条，则n为400,0000；
8000个一次，若共1000条，则n为800,0000；
16000个一次，若共1000条，则n为1600,0000;

=>
n<200,0000时，          interval为2000；
200,0000<=n<400,0000时，interval为4000；
800,0000>n>=400,0000时,  interval为8000；
1600,0000>N>=800,000,    interval为16000；
3200,0000>n>=1600,0000,  interval为32000;
*/
   if(n < 2000000){
     interval_val = 2000;
   }else{
     if(n < 4000000){
       interval_val = 4000;
     }else{
       if(n < 8000000){
         interval_val = 8000;
       }else{
         if(n < 16000000){
           interval_val = 16000;
         }else{
           interval_val = 32000;
         }
       }
     }
   }
    
   int x_nnz = 0;
   small_interval_val = 1;
   do{
    x_nnz += interval_val;
#if 1 
    if(x_nnz < std::min((int)(n*0.0001), 2000)){
      x_nnz += small_interval_val;
      small_interval_val *= 2;//1,2,4,8
    }else{
      x_nnz += interval_val;
    }
#endif
    if (quit) break;
    if(x_nnz >= n) {
      x_nnz = n;
      quit = 1;
    } 
#else
  for (int i = 0; i < iter; i++) {
    int x_nnz = xnnz_vec[i];
    if (quit) break;
    if(x_nnz >= n) {
      x_nnz = n;
      quit = 1;
    } 
#endif
    printf("x_nnz = %d\n", x_nnz);
    fout << x_nnz << " ";
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
#ifdef CORRECT
    //serial spmv computation.
    memset(hres, 0, m * sizeof(VALUE_TYPE));//已经初始化了的
    serialspmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, hres, alpha);
#endif
    cudaErrCheck(cudaMemcpy(d_x_key, x_sparse_key, x_nnz * sizeof(int),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x_val, x_sparse_val, x_nnz * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));

    err = A.set_vector_type(0);//current vector type is sparse type.
    err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);//指针传递
    err = A.set_x(d_x);//指针传递
    //这个kernel输入是d_x_key和d_x_val，输出是d_x和d_bit_vector.
    
    timer.Start();
    A.sparse2dense();//generate values in dense vector.
    double s2dtime = timer.Stop();
    std::cout << "DEBUG: sparse2dense time = " << s2dtime << std::endl;
    fout << s2dtime << " ";

    timer.Start();
    A.sparse2bitarray();// generate values in bitvector.
    double s2atime = timer.Stop();
    std::cout << "DEBUG: sparse2bitarray time = " << s2atime << std::endl;
    fout << s2atime << " ";

    //malloc for d_col_len[] and d_pre_alloc_buffer (csc spmspv preprocess and binlen) .
    A.allocPreBuffer();

    timer.Start();
    //long long int bin_len = A.computeBinlen();
    int bin_len = A.computeBinlenVer2();
    int max_elems = 0;
    int min_elems = 0;
    A.computeVecFeture_serial(x_nnz, x_sparse_key, &bin_len, &max_elems, &min_elems);
    
    double time = timer.Stop();
    std::cout << "DEBUG: compute bin_len time = " << time << "ms." << std::endl;
    std::cout << "DEBUG: bin_len = " << bin_len << std::endl;
    std::cout << "DEBUG: max = " << max_elems << std::endl;
    std::cout << "DEBUG: min = " << min_elems << std::endl;
    fout << bin_len << " ";
    //Add this three features.
    fout << max_elems << " ";
    fout << min_elems << " ";
    fout << 1.0*x_nnz/n << " ";
    fout << 1.0*bin_len/mat_nnz << " ";
    fout << (max_elems - min_elems)/(1.0*n) << " ";//relative range of degree
    //fout << (max_elems - min_elems)/(1.0*n) << " ";
    
    double GM1 = A.computeGM1();
    std::cout << "DEBUG: GM1 = " << GM1 << std::endl;
    fout << GM1 << " ";
    double GM2 =  A.computeGM2();
    std::cout << "DEBUG: GM2 = " << GM2 << std::endl;
    fout << GM2 << " ";
    double GM3 = A.computeGM3();
    std::cout << "DEBUG: GM3 = " << GM3 << std::endl;
    fout << GM3 << " ";
    std::cout << "DEBUG: GM1/GM2 = " << GM1/(double)(GM2) << ", GM2/GM3= " << GM2/(double)GM3 << ", GM1/GM3= " << GM1/(double)GM3 << std::endl;
    fout << GM1/(double)(GM2) << " " << GM2/(double)GM3 << " "<< GM1/(double)GM3 << " ";

#if 1 
  cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
  timer.Start();
  A.CscBasedSortNaiveSpmspv(alpha, &y_nnz, d_y_key_output, d_y_val_output, 0);
  time = timer.Stop();
  std::cout << "sort naive col-spmspv time = " << time << " ms." << std::endl;
  fout << time << " ";
//#ifdef CORRECT
//  A.sparse2dense_outer(*y_nnz, d_y_key_output, d_y_val_output, m, d_y);
//  cudaMemcpy(y_sort_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
//#endif
  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("cscspmspv() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif
   

#if 0 
  cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
  timer.Start();
  A.CscBasedNoSortNaiveSpmspv_keyval(alpha, &y_nnz, d_y, d_y_key_output, d_y_val_output);
  time = timer.Stop();
  std::cout << "no-sort naive col-spmspv time (sparse output) = " << time << " ms." << std::endl;
  //fout << time << " ";
  fout << time << std::endl;
  //cudaMemcpy(y_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost); 
  err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("cscspmspv() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif


#if 1
  timer.Start();
  //A.CscBasedSortSpmspv(alpha, y_nnz, d_y_key, d_y_val, 0);
  A.CscBasedSortMergeSpmspv(false, alpha, &y_nnz, d_y, d_y_key_output, d_y_val_output);
  time = timer.Stop();
  std::cout << "sort  lb-merge col-spmspv time = " << time << " ms." << std::endl << std::endl;
  //fout << time << " ";
  fout << time << std::endl;
  //printf("y_nnz = %d\n", *y_nnz);
//#ifdef CORRECT
//  A.sparse2dense_outer(*y_nnz, d_y_key_output, d_y_val_output, m, d_y);
//  cudaMemcpy(y_sort_dense, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
//#endif
#endif

#if 0 
    //
    cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
    timer.Start();
#ifdef OUTPUT_SPARSE
    //err = A.CscBasedMyNoSortSpmspv_keyval(true, alpha, &y_nnz, d_y, d_y_key_output, d_y_val_output);
    err = A.CscBasedNoSortMergeSpmspv_keyval(true, alpha, &y_nnz, d_y, d_y_key_output, d_y_val_output);
#else
    //err = A.CscBasedMyNoSortSpmspv(true, alpha, &y_nnz, d_y, d_y_key_output, d_y_val_output);
    err = A.CscBasedNoSortMergeSpmspv(true, alpha, &y_nnz, d_y, d_y_key_output, d_y_val_output);
#endif

    time = timer.Stop();
#ifdef CORRECT
    cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
    CheckVector<VALUE_TYPE>(hres, y_dense, m);
#endif
    std::cout << "no-sort my lb col-spmspv time = " << time << "ms." << std::endl;
    //fout << time << " ";
    fout << time << std::endl;
    
   cudaError_t  err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("cscspmspv() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }
#endif

#if 0 
    //3! 
    cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
    timer.Start();
    err = A.naivespmspv(alpha, d_y);
    time = timer.Stop();
#ifdef CORRECT
    cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
    CheckVector<VALUE_TYPE>(hres, y_dense, m);
#endif
    std::cout << "naive spmspv time = " << time << "ms." << std::endl;
    std::cout << "naive spmspv time + saprse2bitarray = " << time + s2atime << "ms." << std::endl;
    fout << time << " ";
    fout << time + s2atime << " ";
   

    cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
    timer.Start();
    err = A.spmspv(alpha, d_y);
    time = timer.Stop();
#ifdef CORRECT
    cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
    CheckVector<VALUE_TYPE>(hres, y_dense, m);
#endif
    std::cout << "lb spmspv time = " << time << "ms." << std::endl;
    std::cout << "lb spmspv time + saprse2bitarray = " << time + s2atime << "ms." << std::endl;
    fout << time << " ";
    fout << time + s2atime << " ";
   
    //5!!
    cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
    timer.Start();
    err = A.naivespmv(alpha, d_y);
    time = timer.Stop();
#ifdef CORRECT
    cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
    CheckVector<VALUE_TYPE>(hres, y_dense, m);
#endif
    std::cout << "naive spmv time = " << time << "ms." << std::endl;
    std::cout << "naive spmv time + sparse2dense = " << time + s2dtime << "ms." << std::endl;
    fout << time << " ";
    fout << time + s2dtime << " ";
   
    //6!!
    cudaErrCheck(cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE))); //initialized to zero.
    timer.Start();
    err = A.spmv(alpha, d_y);
    time = timer.Stop();
#ifdef CORRECT
    cudaErrCheck(cudaMemcpy(y_dense, d_y, (m) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost)); 
    CheckVector<VALUE_TYPE>(hres, y_dense, m);
#endif
    std::cout << "lb spmv time = " << time << "ms." << std::endl;
    std::cout << "lb spmv time + sparse2dense = " << time + s2dtime << "ms." << std::endl;
    fout << time << " ";
    fout << time + s2dtime << std::endl;
#endif

    A.deallocPreBuffer();

#ifdef INTERVAL
  }while(x_nnz <= n);
#else
  }
#endif

#if USE_CSR5
  if(spmv_type == 0) {
    A.holaPostprocess();
  }else{
    A.CSR5Postprocess();
  }
#else
  A.holaPostprocess();
#endif

A.Destroy();

#ifdef CORRECT 
  if (hres)    free(hres);
#endif

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);
  if (d_y)       cudaFree(d_y);

//#ifdef OUTPUT_SPARSE
  if(d_y_key_output)    cudaErrCheck(cudaFree(d_y_key_output)); 
  if(d_y_val_output)    cudaErrCheck(cudaFree(d_y_val_output)); 
//#endif

  if (d_x_key)        cudaFree(d_x_key);
  if (d_x_val)        cudaFree(d_x_val);
  if (d_bit_vector)   cudaFree(d_bit_vector);

  fout.close();
return err;
}

int doThis(std::string s_file_name, std::string dir_name) {
  const char* file_name = s_file_name.c_str(); 
  //std::cout << "-------" << file_name << "---------" << std::endl;
  std::string file = dir_name + "/" + s_file_name;
  const char* real_file_name = file.c_str();
  std::cout << "file path = " << real_file_name << std::endl;

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

#ifndef READ_SVM_DATA
  MTX<VALUE_TYPE> mtx;
  //add fast reading:
  std::string csr_name = std::string(real_file_name) + "_" + ".csr";
  try
  {
    std::cout << "trying to load csr file \"" << csr_name << "\"\n";
    loadCSR_header(csr_name.c_str(), &m, &n, &mat_nnz);
    
    csr_row = (int* )malloc((m + 1) * sizeof(int));
    CHECK_MALLOC(csr_row);
    csr_col = (int* )malloc((mat_nnz) * sizeof(int));
    CHECK_MALLOC(csr_col);
    csr_val = (VALUE_TYPE* )malloc((mat_nnz) * sizeof(VALUE_TYPE));
    CHECK_MALLOC(csr_val);
    
    loadCSR(csr_name.c_str(), m, n, mat_nnz, csr_row, csr_col, csr_val);
  }
  catch (std::exception& ex){
    std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
  
    fileToMtxCoo<VALUE_TYPE>(real_file_name, &mtx, true); 
    m = mtx.rows;
    n = mtx.cols;
    mat_nnz = mtx.nnz;

    //coo2csr: attention memory alloc and free.
    csr_row = (int* )malloc((m + 1) * sizeof(int));
    CHECK_MALLOC(csr_row);
  
    TestCoo2Csr(m, mat_nnz, mtx.row, csr_row);
    csr_col = mtx.col;
    csr_val = mtx.data;
  
    try
    {
      storeCSR(m, n, mat_nnz, csr_row, csr_col, csr_val,csr_name.c_str());
    }
    catch (std::exception& ex)
    {
    std::cout << ex.what() << std::endl;
    }
    
 }
#else
  SvmData<VALUE_TYPE> svm_data;
  readSVMToCoo<VALUE_TYPE>(real_file_name, &svm_data);

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

  for(int i=0; i<n; i++) {
    x_dense[i] = (VALUE_TYPE)i;
  }

  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(y_dense);
  
  // get matrix name.
  std::string matrix_name;
  std::string real_file = (std::string)real_file_name;
  int nPos1 = real_file.find_last_of(".");
  int nPos2 = real_file.find_last_of("/", nPos1 - 1);
  if(nPos1 != -1 && nPos2 != -1) {
    matrix_name = file.substr(nPos2 + 1, nPos1 - nPos2 - 1);
  }
  std::cout << "matrix_name = " << matrix_name << std::endl;
  std::cout << "m = " << m << ", n = " << n << ", nnz = " << mat_nnz << std::endl;


  std::string out_file = "/home/*/motivation-sort/" 
                       + matrix_name + "_feature_perf.info"; 
  Run(out_file, matrix_name, m, n, mat_nnz, csr_row, csr_col, csr_val, 
      x_sparse_key, x_sparse_val, x_dense, y_dense);

  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);
  if (x_dense)      free(x_dense);
  if (y_dense)      free(y_dense);

#ifndef READ_SVM_DATA
  //if (mtx.row)   free(mtx.row);
  //if (mtx.col)   free(mtx.col);
  //if (mtx.data)  free(mtx.data);
  if (csr_row)   free(csr_row);//forget before.
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#endif

  return 0;
}


int main(int argc, char** argv) {
  std::string dir_name;
  if (argc == 2) {
    dir_name = (std::string)argv[1];
  }else{
    std::cout << "Usage: dir_name " << std::endl;
    exit(1);
  }
  // check the parameter !
	//if( NULL == dir_name ) {
  if(dir_name.empty()) {
		std::cout << " dir_name is null ! " << std::endl;
		return 1;
	}
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat( dir_name.c_str() , &s );
	if( ! S_ISDIR( s.st_mode ) )
	{
		std::cout << "dir_name is not a valid directory !" << std::endl;
		return 1;
	}
	
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir(dir_name.c_str());
	if(NULL == dir) {
		std::cout << "Can not open dir " << dir_name << std::endl;
		return 1;
	}
	std::cout << "Successfully opened the dir !" << std::endl << std::endl;
  
  std::vector<std::string> files;
  std::vector<std::string> mtx_files;
	/* read all the files in the dir ~ */
	while((filename = readdir(dir)) != NULL) {
		// get rid of "." and ".."
		if(strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0)
			continue;
    std::cout << filename ->d_name << std::endl;
    files.push_back(filename ->d_name);
	}
  int count = files.size();
  for (int i = 0; i < count; i++) {
    std::string cur_f = files[i];
    int len = cur_f.length();
    if(cur_f[len-1] = 'x' && cur_f[len-2]=='t' && cur_f[len-3]=='m' && cur_f[len-4] == '.')
      mtx_files.push_back(files[i]);
  }
  
  count = mtx_files.size();
  for (int i = 0; i < count; i++) {
      std::cout << mtx_files[i] << std::endl;
      doThis(mtx_files[i], dir_name);
  }
  
return 0;
}

