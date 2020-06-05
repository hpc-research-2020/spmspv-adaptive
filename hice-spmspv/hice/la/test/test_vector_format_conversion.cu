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


//#define TEST_MAT_FEATURE_TIME

//#define TEST_S2D_D2S
//#define TEST_COMPUTE_BINLEN_TIME

#define SPARSE_THRESHOLD_SMALL 0.08
#define SPARSE_THRESHOLD_BIG 0.9
#define NNZ_THRESHOLD 20000

#define TEST_OVERHEAD


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
  
  //warmup.
  err = A.sparse2dense();

  SpmspvTimer s2d_timer;
  s2d_timer.Start();
  //sparse vect to dense vec.
  for(int i=0; i<10; i++)
    err = A.sparse2dense();
  double time = s2d_timer.Stop()/10;
  
  cudaErrCheck(cudaMemcpy(x_dense, d_x, (n) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));  

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


SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, nnz);


//only assignment. Don not have memory alloc.
//current vector type is dense type.
err = A.set_vector_type(1);
err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);
err = A.set_x(d_x); 

A.allocPreBuffer();
//warmup.
err = A.dense2sparse();

SpmspvTimer d2s_timer;
d2s_timer.Start();
//dense vect to sparse vec.
for(int i=0; i<10; i++)
  err = A.dense2sparse();
double time = d2s_timer.Stop()/10;

std::cout << "dense2sparse time = " << time << " ms." << std::endl;

cudaErrCheck(cudaMemcpy(x_key, d_x_key, (x_nnz) * sizeof(int), cudaMemcpyDeviceToHost));  
cudaErrCheck(cudaMemcpy(x_val, d_x_val, (x_nnz) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));  

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
int num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
int* d_bit_vector = NULL;
cudaErrCheck(cudaMalloc((void** )&d_bit_vector, num_ints * sizeof(int)));

err = A.set_bitvector(d_bit_vector, num_ints);//

A.allocPreBuffer();
//warmup.
err = A.sparse2bitarray();

SpmspvTimer s2b_timer;
s2b_timer.Start();
//sparse vect to bitarray vec.
for(int i=0; i<10; i++)
  err = A.sparse2bitarray();
double time = s2b_timer.Stop()/10;

std::cout << "sparse2bitarray time = " <<  time << " ms." << std::endl;

cudaErrCheck(cudaFree(d_x));        
cudaErrCheck(cudaFree(d_x_key)); 
cudaErrCheck(cudaFree(d_x_val)); 
cudaErrCheck(cudaFree(d_bit_vector)); 

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
void generate_random_sparse_vector(int n, int nnz, int* key, vT* value, vT* x_dense) {
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

    x_dense[dst] = 1;
  }
}

template <typename vT>
void Vec2File(vT* y, int len, const char* file_name) {
  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout << "open file " << file_name << " failed." << std::endl;
      exit(1);
  }
  for (size_t i = 0; i < len; i++) {
    fout << y[i] << std::endl;
  }
  fout.close();
}

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

#ifdef TEST_MAT_FEATURE_TIME
  void testMatFeatureTime(int m, int n, int mat_nnz, int* csr_row, int* csr_col, VALUE_TYPE* csr_val){

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
    A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
    A.allocPreBuffer();

    SpmspvTimer timer;
    double time1, time2;

    timer.Start();
    int cpu_max_nnz=0, cpu_min_nnz=0, gpu_max_nnz=0, gpu_min_nnz=0;
    float cpu_avg_nnz=0, cpu_sd_nnz=0, gpu_avg_nnz=0, gpu_sd_nnz=0;
    cpu_sd_nnz = A.extractMatFeature(csr_row/*, &cpu_max_nnz, &cpu_min_nnz,*/ /*&cpu_avg_nnz, &cpu_sd_nnz*/);
    time1 = timer.Stop();
    std::cout << "DEBUG: extractmatfeature time = " << time1 << std::endl;

    timer.Start();
    gpu_sd_nnz = A.extractMatFeatureParallel(csr_row/*, &gpu_max_nnz, &gpu_min_nnz,*/ /*&gpu_avg_nnz, &gpu_sd_nnz*/);
    time2 = timer.Stop();
    std::cout << "DEBUG: extractmatfeatureParallel time = " << time2 << std::endl;
    std::cout << "speedup = " << time1/time2 << std::endl;
    if(cpu_max_nnz == gpu_max_nnz && cpu_min_nnz == gpu_min_nnz && abs(cpu_sd_nnz - gpu_sd_nnz)<1e-6){
      std::cout << "Passed" << std::endl;
    }else{
      std::cout << "Failed" << std::endl;
      std::cout << "cpu_max_nnz = " << cpu_max_nnz << ", gpu_max_nnz = " << gpu_max_nnz << std::endl;
      std::cout << "cpu_min_nnz = " << cpu_min_nnz << ", gpu_min_nnz = " << gpu_min_nnz << std::endl;
      std::cout << "cpu_sd_nnz = " << cpu_sd_nnz << ", gpu_sd_nnz = " << gpu_sd_nnz << std::endl;
    }
    A.deallocPreBuffer();
    A.Destroy();
    if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
    if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
    if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  }
#endif


#ifdef TEST_COMPUTE_BINLEN_TIME
  void testComputeBinlen(int m, int n, int mat_nnz, int* csr_row, int* csr_col, VALUE_TYPE* csr_val,
                            int x_nnz, int* x_key, VALUE_TYPE* x_val){

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

    int* d_x_key = NULL;
    VALUE_TYPE* d_x_val = NULL;
    cudaErrCheck(cudaMalloc((void** )&d_x_key, (x_nnz) * sizeof(int)));
    cudaErrCheck(cudaMalloc((void** )&d_x_val, (x_nnz) * sizeof(VALUE_TYPE)));

    cudaErrCheck(cudaMemcpy(d_x_key, x_key, x_nnz * sizeof(int),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x_val, x_val, x_nnz * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));

    SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
    A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
    A.ToCSC();//malloc for d_csc_row[] and so on.

    //current vector type is sparse type.
    A.set_vector_type(0);
    A.set_sparse_x(d_x_key, d_x_val, x_nnz);
    A.set_x(d_x); 

    A.allocPreBuffer();

    SpmspvTimer timer;
    double time1, time2;

    timer.Start();
    int cpu_bin_len = A.computeBinlen();
    time1 = timer.Stop();
    std::cout << "DEBUG: computeBinlen-ver1 time = " << time1 << std::endl;

    timer.Start();
    int gpu_bin_len = A.computeBinlenVer2();
    time2 = timer.Stop();
    std::cout << "DEBUG: computeBinlen-ver2 time = " << time2 << std::endl;
    std::cout << "speedup = " << time1/time2 << std::endl;
    if(cpu_bin_len == gpu_bin_len){
      std::cout << "Passed" << std::endl;
    }else{
      std::cout << "Failed" << std::endl;
      std::cout << "cpu_bin_len = " << cpu_bin_len << ", gpu_bin_len = " << gpu_bin_len << std::endl;
    }
    
    if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
    if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
    if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));

    if(d_x)        cudaErrCheck(cudaFree(d_x));        
    if(d_x_key)    cudaErrCheck(cudaFree(d_x_key)); 
    if(d_x_val)    cudaErrCheck(cudaFree(d_x_val)); 

    A.deallocPreBuffer();
    A.Destroy();

  }
#endif

#ifdef TEST_OVERHEAD
//the sparse vector is produced from application.
void testDriver2(std::string file_name, int m, int n, int mat_nnz, 
          int* csr_row, int* csr_col, VALUE_TYPE* csr_val){
  
  std::ofstream fout;
  //fout.open("./overhead-bfs-perf.out", std::ofstream::app);
  fout.open("./overhead-pr-perf.out", std::ofstream::app);
  if(!fout){
    std::cout << "file can't open" << std::endl;
    exit(1);
  }
  fout << file_name << std::endl;

  SpmspvTimer timer;
  double overhead_time = 0.0;
  double overhead_time_filter = 0.0;

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

  int num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  int* d_bit_vector = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_bit_vector, num_ints * sizeof(int)));

  SpmspvHandle<int, unsigned int, VALUE_TYPE> A(m, n, mat_nnz);
  A.InputCSR(mat_nnz, d_csr_row, d_csr_col, d_csr_val);
  A.ToCSC();//malloc for d_csc_col[] and so on.
  A.set_bitvector(d_bit_vector, num_ints);//指针传递
  
  //malloc for d_col_len[] and d_pre_alloc_buffer (csc spmspv preprocess and binlen).
  timer.Start();
  A.allocPreBuffer();
  double prealloctime = timer.Stop();
  std::cout << "DEBUG: prealloctime = " << prealloctime << std::endl;
  fout << prealloctime << " ";

  //extract matrix feature.
  int gpu_max_nnz = 0, gpu_min_nnz = 0;
  float gpu_avg_nnz = 0, gpu_sd_nnz = 0;

  timer.Start();
  gpu_sd_nnz = A.extractMatFeatureParallel(csr_row/*, &gpu_max_nnz, &gpu_min_nnz,*/ /*&gpu_avg_nnz, &gpu_sd_nnz*/);
  double matFeatureTime = timer.Stop();
  overhead_time += matFeatureTime;
  overhead_time_filter += matFeatureTime;
  std::cout << "DEBUG: extractmatfeatureParallel time = " << matFeatureTime << std::endl;
  fout << matFeatureTime << " ";

  int* x_sparse_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_sparse_key);
  VALUE_TYPE* x_sparse_val = (VALUE_TYPE* )malloc(n * 
                            sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_sparse_val);
  
  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);

  for(int i=0; i<n; i++) {
    x_dense[i] = (VALUE_TYPE)i;
  }
  
  // get matrix name.
  std::string matrix_name;
  int nPos1 = file_name.find_last_of(".");
  int nPos2 = file_name.find_last_of("/", nPos1 - 1);
  if(nPos1 != -1 && nPos2 != -1) {
    matrix_name = file_name.substr(nPos2 + 1, nPos1 - nPos2 - 1);
  }
  std::cout << "matrix_name = " << matrix_name << std::endl;
  std::cout << "m = " << m << ", n = " << n << ", nnz = " << mat_nnz << std::endl;

  int iter = 0;

  std::string file_prex = "/home/*/pr_xnnz_";
  std::string file_suffix = ".txt";
  std::string file_all = file_prex + matrix_name + file_suffix;
  std::cout << "reading xnnz from file: " << file_all << std::endl;

  readNNZXFromFile(file_all.c_str(), &iter);
  std::cout << "iter = " << iter << std::endl;

  int* xnnz_vec = (int* )malloc(iter * sizeof(int));
  CHECK_MALLOC(xnnz_vec);
  readSparseXInxFromFile(file_all.c_str(), iter, xnnz_vec);

  VALUE_TYPE* d_x = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x, n * sizeof(VALUE_TYPE)));

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (n) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (n) * sizeof(VALUE_TYPE)));

  double all_time = 0.0;
  double vec_time = 0.0;
  double vec_filter_time = 0.0;
  int y_nnz = 0;
  int quit = 0;
  

  for (int i = 0; i < iter; i++) {
    int x_nnz = xnnz_vec[i];
    if (quit) break;
    if(x_nnz >= n) {
      x_nnz = n;
    } 
    printf("x_nnz = %d\n", x_nnz);
    
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));
    memset(x_dense, 0, n * sizeof(VALUE_TYPE));
    
    //std::string in_file_prex = "/home/*/bfs_x_";
    std::string in_file_prex = "/home/*/pr_x_";
    std::string in_file_suffix = ".txt";
    std::string in_file_name = in_file_prex + matrix_name + in_file_suffix;
    if(i == 0)
      std::cout << "reading sparse x from file: " << in_file_name << std::endl;
    extractSparseXfromFile(in_file_name, i, x_nnz, x_sparse_key, x_sparse_val, x_dense);

    cudaErrCheck(cudaMemcpy(d_x_key, x_sparse_key, x_nnz * sizeof(int),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x_val, x_sparse_val, x_nnz * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));

    A.set_vector_type(0);//current vector type is sparse type.
    A.set_sparse_x(d_x_key, d_x_val, x_nnz);//
    A.set_x(d_x);//
    
    timer.Start();
    A.sparse2dense();//generate values in dense vector.
    double s2dtime = timer.Stop();
    std::cout << "DEBUG: sparse2dense time = " << s2dtime << std::endl;
  
    timer.Start();
    A.dense2sparse();
    double d2stime = timer.Stop();
    std::cout << "DEBUG: dense2sparse time = " << d2stime << std::endl;

    timer.Start();
    A.sparse2bitarray();// generate values in bitvector.
    double s2atime = timer.Stop();
    std::cout << "DEBUG: sparse2bitarray time = " << s2atime << std::endl;

    timer.Start();
    long long int bin_len = A.computeBinlen();
    double GM1 = A.computeGM1();
    double GM2 =  A.computeGM2();
    double GM3 = A.computeGM3();
    double time1 = timer.Stop();

    timer.Start();
    long long int bin_len_1 = A.computeBinlenVer2();
    double time2 = timer.Stop();
    std::cout << "DEBUG: compute binlen time-ver1 = " << time1 << "ms." << std::endl;
    std::cout << "DEBUG: compute binlen time-ver2 = " << time2 << "ms." << std::endl;
    std::cout << "speedup = " << time1/time2 << std::endl;
    std::cout << "DEBUG: bin_len = " << bin_len << std::endl;
    
    overhead_time += time2;
    vec_time += time2;

    float sparsity = x_nnz/(float)n;
    if(sparsity > SPARSE_THRESHOLD_SMALL && sparsity < SPARSE_THRESHOLD_BIG && mat_nnz > NNZ_THRESHOLD){
      overhead_time_filter += time2;
      vec_filter_time += time2;
    }
  }
  std::cout << "ALL: extract mat and vector feature time = " << overhead_time << std::endl;
  std::cout << "ALL: extract mat and vector feature (filter) time = " << overhead_time_filter << std::endl;
  fout << vec_time << " " << vec_filter_time << " "; 

  if(xnnz_vec)     free(xnnz_vec);
  if(x_sparse_key) free(x_sparse_key);
  if(x_sparse_val) free(x_sparse_val);

  if(d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if(d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if(d_csr_val) cudaErrCheck(cudaFree(d_csr_val));

  if(d_x)        cudaErrCheck(cudaFree(d_x));        
  if(d_x_key)    cudaErrCheck(cudaFree(d_x_key)); 
  if(d_x_val)    cudaErrCheck(cudaFree(d_x_val)); 

  if(d_bit_vector) cudaErrCheck(cudaFree(d_bit_vector));

  timer.Start();
  A.deallocPreBuffer();
  double dealloctime = timer.Stop();
  std::cout << "DEBUG: dealloctime = " << dealloctime << std::endl;
  fout << dealloctime << std::endl << std::endl;
  A.Destroy();

  fout.close();
}
#endif

#ifdef TEST_S2D_D2S
//the sparse vector is random acquired.
void testDriver1(char* file_name, int m, int n, int mat_nnz, 
          int* csr_row, int* csr_col, VALUE_TYPE* csr_val){

  int* x_sparse_key = (int* )malloc(n * sizeof(int));
  CHECK_MALLOC(x_sparse_key);
  VALUE_TYPE* x_sparse_val = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_sparse_val);

  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);

  VALUE_TYPE* x_base = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_base);

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
  // int iter = 86;
  // int xnnz_vec[86] = {1000,3000,5000,7000,9000,11000,13000,16000,19000,
  //                     22000,25000,28000,31000,34000,37000,40000,50000,
  //                     64000,98000,123000,166000,240000,260000,280000,
  //                     315000,340000,380000,420000,460000,484000,540000,
  //                     580000,616000,650000,700000,735000,853000,955000,
  //                     1129000,1213000,1366000,1538000,1729000,1902000,
  //                     2061000,2252000,2403000,2563000,2740000,2899000,
  //                     3069000,3200000,3400000,3600000,3800000,4000000,
  //                     4200000,4400000,4600000,4800000,5000000,5200000,
  //                     5400000,5558326,6000000,6500000,7000000,7500000,
  //                     8000000,8500000,9000000,9500000,10500000,11000000,
  //                     11500000,12000000,12500000,13000000,13500000,14000000,
  //                     14500000,15000000,15500000,16000000,16500000,
  //                     17000000}; 
  int iter = 1;
  //int xnnz_vec[1] = {17000000};
  int xnnz_vec[1] = {65536};
  int quit = 0;
  for (int i = 0; i < iter; i++) {
    int x_nnz = xnnz_vec[i];
    if (quit) break;
    if(x_nnz >= n) {
      x_nnz = n;
      quit = 1;
    } 
    printf("x_nnz = %d\n", x_nnz);
    
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));
    memset(x_dense, 0, n * sizeof(VALUE_TYPE));
    memset(x_base, 0, n * sizeof(VALUE_TYPE));

  #ifdef SPARSE_X_RANDOM
    srand(time(NULL));
    generate_random_sparse_vector(n, x_nnz, x_sparse_key, x_sparse_val, x_base);
  #else
    for (int i = 0; i < x_nnz; i++) {
      x_sparse_key[i] = i;
      x_sparse_val[i] = 1;
      x_base[i] = 1;
    }
  #endif

  #endif //end of different methods to read sparse x. 

  #ifdef TEST_COMPUTE_BINLEN_TIME
    testComputeBinlen(m, n, mat_nnz, csr_row, csr_col, csr_val,
                  x_nnz, x_sparse_key, x_sparse_val);
  #endif

    TestSparse2Dense(x_nnz, x_sparse_key, x_sparse_val, 
                    m, n, mat_nnz, x_dense);
    CheckVector<VALUE_TYPE>(x_base, x_dense, n);//result ok!
    
    // <=> dense 2 sparse
    TestDense2Sparse(x_nnz, x_sparse_key, x_sparse_val, 
                    m, n, mat_nnz, x_dense); 
    
    Testsparse2bitarray(x_nnz, x_sparse_key, x_sparse_val, 
                    m, n, mat_nnz, x_dense);                

  }//end of iteration.

  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);

  if (x_dense)      free(x_dense);
  if (x_base)       free(x_base);
}
#endif


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

  //Read sparse matrix.
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

#ifdef TEST_MAT_FEATURE_TIME
  testMatFeatureTime(m, n, mat_nnz, csr_row, csr_col, csr_val);
#endif

//test vector format conversion and computbinlen's correctness when sparse vector is random generated.
#ifdef TEST_S2D_D2S
  testDriver1(file_name, m, n, mat_nnz, csr_row, csr_col, csr_val);
#endif

//test overheads in real apps.
#ifdef TEST_OVERHEAD
  testDriver2((std::string)file_name, m, n, mat_nnz, csr_row, csr_col, csr_val);
#endif

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
