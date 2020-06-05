// This is used to Test the adaptive Spmspv/spmv performance in applications. 

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


//#define BFS
//#define CORRECT

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

  cudaErrCheck(cudaMemcpy(csr_row, d_csr_row, (m + 1) * sizeof(int),   
                 cudaMemcpyDeviceToHost));
  
  cudaErrCheck(cudaDeviceSynchronize());

  cudaErrCheck(cudaFree(d_csr_row));
  cudaErrCheck(cudaFree(d_coo_row));
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




int predictSolution(float nnz, float x_sparsity){
  return 0;
}

int predictWorkloadDistribution(float maxRow, float avgRow, float stdRow){
  return 0;
}

int predictWritingBack(float m, float x_sparisity){
  return 0;
}

int Run(std::string file_name, std::string matrix_name, int iter, float alpha, float beta, 
        int m, int n, int mat_nnz,
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
  if (!fout.is_open()) {
    std::cout << "can't open output file\n" << file_name << std::endl;
    exit(1);
  }
  fout << "mat_name, m, n, nnz, max, min, avg, m_range, std, equlity, gini, x_nnz, sparse2dense, sparse2bitarray,  bin_len, max, min, xnnz/n, bin_len/nnz, xnnz_range, m1_id, m2_id, m3_id, time_r, time_c, single_atomic_col_kernel, single_sort_col_kernel, single_spmv, framework\n";
  fout << matrix_name << " ";

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
  
  int* d_y_key = NULL; //result vec.
  VALUE_TYPE* d_y_val = NULL; 
  cudaErrCheck(cudaMalloc((void** )&d_y_key, m * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_y_val, m * sizeof(VALUE_TYPE)));

  VALUE_TYPE alpha_i = 1.0;

  SpmspvTimer timer;
/*********select spmv: use holaspmv directly***********/
  err = A.holaPreprocess();
  timer.Start();
  for (int i = 0; i < NUM_RUN; i++)
    err = A.holaspmv(alpha_i, d_y);
  err_cuda = cudaDeviceSynchronize();
  double holaspmv_time = timer.Stop()/ (double)NUM_RUN;
  //double holaspmv_time = timer.Stop();
  std::cout << "holaspmv time " << holaspmv_time << std::endl;
  //fout << holaspmv_time << " ";
  
  double csr5spmv_time = 0;
  //fout << csr5spmv_time << " ";

  int spmv_type = 0;
  spmv_type = 0;
  A.set_spmv_type(0);
  //fout << spmv_type << " ";
  //fout << "alpha = " << alpha << ", beta = " << beta << std::endl;

#ifdef CORRECT
  //for serial spmv.
  VALUE_TYPE* hres =  (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(hres);
#endif

  err = A.ToCSC();
#ifdef BFS
  std::string file_prex = "/home/*/bfs_xnnz_";
#else
  std::string file_prex = "/home/*/pr_xnnz_";
#endif 
  std::string file_suffix = ".txt";
  std::string file_all = file_prex + matrix_name + file_suffix;
  std::cout << "reading xnnz from file: " << file_all << std::endl;

  readNNZXFromFile(file_all.c_str(), &iter);
  std::cout << "iter = " << iter << std::endl;

  int* xnnz_vec = (int* )malloc(iter * sizeof(int));
  CHECK_MALLOC(xnnz_vec);
  readSparseXInxFromFile(file_all.c_str(), iter, xnnz_vec);

  int* d_x_key = NULL;
  VALUE_TYPE* d_x_val = NULL;
  cudaErrCheck(cudaMalloc((void** )&d_x_key, (n) * sizeof(int)));
  cudaErrCheck(cudaMalloc((void** )&d_x_val, (n) * sizeof(VALUE_TYPE)));

  //fout << "x_nnz, sparse2dense, sparse2bitarray, bin_len, max, min, xnnz/n, bin_len/nnz, xnnz_range, GM1, GM2, GM3, GM1/GM2, GM2/GM3, GM1/GM3, naive-col, lb-col, naive-rspmspv,  naive-rspmspv+s2a, lb-rspmspv, lb-rspmspv+s2a, naive-spmv, naive-spmv+s2d, lb-spmv, lb-spmv+s2d \n";

  double all_time = 0.0;

  int y_nnz = 0;
  int quit = 0;

   //malloc for d_col_len[] and d_pre_alloc_buffer (csc spmspv preprocess and binlen) .
   A.allocPreBuffer();

   double time_sa=0, time_ss=0, time_sspmv=0;
   double time_r, time_c;
   double time_single_atomic_col_spmspv = 0.0; 
   double time_single_sort_col_spmspv = 0.0;
  double time_single_spmv = 0.0;
  double time_rule = 0, time_spmv = 0;
    
  int mat_max_elems, mat_min_elems;
  float mat_avg_elems, mat_x_range, mat_standard_row, mat_equlity, mat_gini;
  A.computeMatFeture_serial(m, n, mat_nnz, csr_row, &mat_max_elems, &mat_min_elems, &mat_avg_elems, 
        &mat_x_range, &mat_standard_row, &mat_equlity, &mat_gini);
    
  fout << m << " ";
  fout << n << " ";
  fout << mat_nnz << " ";
  fout << mat_max_elems << " ";
  fout << mat_min_elems << " ";
  fout << mat_avg_elems << " ";
  fout << mat_x_range << " ";//relative range of degree
  fout << mat_standard_row<< " ";
  fout << mat_equlity << " ";
  fout << mat_gini << std::endl;
  
  for (int i = 0; i < iter; i++) {
    int x_nnz = xnnz_vec[i];
    if (quit) break;
    if(x_nnz >= n) {
      x_nnz = n;
    } 
    printf("x_nnz = %d\n", x_nnz);
    fout << x_nnz << " ";
    memset(x_sparse_key, 0, n * sizeof(int));
    memset(x_sparse_val, 0, n * sizeof(VALUE_TYPE));
    memset(x_dense, 0, n * sizeof(VALUE_TYPE));

#ifdef BFS
    std::string in_file_prex = "/home/*/bfs_x_";
#else
    std::string in_file_prex = "/home/*/pr_x_";
#endif    
    std::string in_file_suffix = ".txt";
    std::string in_file_name = in_file_prex + matrix_name + in_file_suffix;
    std::cout << "reading sparse x from file: " << in_file_name << std::endl;
    extractSparseXfromFile(in_file_name, i, x_nnz, x_sparse_key, x_sparse_val, x_dense);

#ifdef CORRECT
    //serial spmv computation.
    memset(hres, 0, m * sizeof(VALUE_TYPE));//
    serialspmv(m, n, mat_nnz, csr_row, csr_col, csr_val, x_dense, hres, alpha_i);
#endif
    cudaErrCheck(cudaMemcpy(d_x_key, x_sparse_key, x_nnz * sizeof(int),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x_val, x_sparse_val, x_nnz * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_x, x_dense, n * sizeof(VALUE_TYPE),   
              cudaMemcpyHostToDevice));

    err = A.set_vector_type(0);//current vector type is sparse type.
    err = A.set_sparse_x(d_x_key, d_x_val, x_nnz);//
    err = A.set_x(d_x);//
   
    timer.Start();
    A.sparse2dense();//generate values in dense vector.
    double s2dtime = timer.Stop();
    std::cout << "DEBUG: sparse2dense time = " << s2dtime << std::endl;
    //fout << "s2dtime = " << s2dtime << " ";
    fout << s2dtime << " ";

    timer.Start();
    A.sparse2bitarray();// generate values in bitvector.
    double s2atime = timer.Stop();
    std::cout << "DEBUG: sparse2bitarray time = " << s2atime << std::endl;
    //fout << " s2atime = " << s2atime << " ";
    fout << s2atime << " ";

    timer.Start();
    int bin_len = A.computeBinlenVer2();
    double time = timer.Stop();
    std::cout << "DEBUG: compute bin_len time = " << time << "ms." << std::endl;
    std::cout << "DEBUG: bin_len = " << bin_len << std::endl;
    
    int max_elems = 0;
    int min_elems = 0;
    A.computeVecFeture_serial(x_nnz, x_sparse_key, &bin_len, &max_elems, &min_elems);
    
    fout << bin_len << " ";
    fout << max_elems << " ";
    fout << min_elems << " ";
    fout << 1.0*x_nnz/n << " ";
    fout << 1.0*bin_len/mat_nnz << " ";
    fout << (max_elems - min_elems)/(1.0*n) << " ";//relative range of degree
    
    float x_sparsity = 1.0*x_nnz/n;
    //rules judgement
    int solution_id = 2;//0,1,2
    int workloadDistribution_id = 1;//0,1
    int writingBack_id = 1;//0,1(sort, atomic)
    
    timer.Start();
    //int predictSolution(int nnz, int x_sparsity){
    solution_id = predictSolution((float)mat_nnz, x_sparsity);
    //int predictWorkloadDistribution(int maxRow, int avgRow, int stdRow){
    workloadDistribution_id = predictWorkloadDistribution((float)mat_max_elems, mat_avg_elems, mat_standard_row);
    
    if(solution_id == 0){
      //int predictWritingBack(int m, int x_sparisity){
      writingBack_id = predictWritingBack((float)m, x_sparsity);
    }
    time_r = timer.Stop();
    time_rule += time_r;
    
    fout << " " << solution_id << " " << workloadDistribution_id << " " << writingBack_id << "  ";
#if 1 
    //execute sinle kernel
    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    timer.Start();
    err = A.CscBasedNoSortMergeSpmspv_keyval(true, alpha_i, &y_nnz, d_y, d_y_key, d_y_val);
    time_sa = timer.Stop();
    time_single_atomic_col_spmspv += time_sa;
    cudaError_t err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("cscspmspv() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    } 
#endif
   
#if 0 
    //execute sinle kernel
    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    timer.Start();
    err = A.CscBasedSortMergeSpmspv(false, alpha, &y_nnz, d_y, d_y_key, d_y_val);
    time_ss = timer.Stop();
    time_single_sort_col_spmspv += time_ss;
    err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("cscspmspv() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    } 
#endif

    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    timer.Start();
    err = A.spmv(alpha_i, d_y);
    time_sspmv = timer.Stop();
    time_single_spmv += time_sspmv;
#if 0 
    //execute adaptive framework
    //SpMV
    cudaMemset(d_y, 0,  m * sizeof(VALUE_TYPE));
    if(solution_id == 2){
      if(workloadDistribution_id == 0){
        timer.Start();
        err = A.naivespmv(alpha, d_y);
        time_c = timer.Stop();
      }else{
        timer.Start();
        err = A.spmv(alpha_i, d_y);
        time_c = timer.Stop();
      }
    }else if(solution_id == 1){
      if(workloadDistribution_id == 0){
        timer.Start();
        err = A.naivespmspv(alpha, d_y);
        time_c = timer.Stop();
      }else{
        timer.Start();
        err = A.spmspv(alpha_i, d_y);
        time_c = timer.Stop();
      }
    }else if(solution_id == 0){
      //naive
      if(workloadDistribution_id == 0){
        if(writingBack_id == 0){
          //sort
          timer.Start();
          err = A.CscBasedSortNaiveSpmspv(alpha, &y_nnz, d_y_key, d_y_val, 0);
          time_c = timer.Stop();
        }else{
          //atomic  
          timer.Start();
          err = A.CscBasedNoSortNaiveSpmspv_keyval(alpha, &y_nnz, d_y, d_y_key, d_y_val);
          time_c = timer.Stop();
        }
      }else{
      //load-balanced
        if(writingBack_id == 0){
          //sort
          timer.Start();
          err = A.CscBasedSortMergeSpmspv(false, alpha, &y_nnz, d_y, d_y_key, d_y_val);
          time_c = timer.Stop();
        }else{
          //atomic
          timer.Start();
          err = A.CscBasedNoSortMergeSpmspv_keyval(true, alpha_i, &y_nnz, d_y, d_y_key, d_y_val);
          time_c = timer.Stop();
        }
      }
    }
    cudaError_t err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("framework: cscspmspv() invocate error.\n");
      printf("solution_id = %d, workloadDistribution_id =%d, writingBack_id=%d.\n", 
          solution_id, workloadDistribution_id, writingBack_id);
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }
#endif 
    time_spmv += time_c;
    fout << time_sa << " " << time_ss << " " << time_sspmv << "  " << time_r << " " << time_c << std::endl; 

  }
  fout << time_single_atomic_col_spmspv << " ";
  fout << time_single_sort_col_spmspv << " " << time_single_spmv << " "; 
  fout << time_rule << " " << time_spmv << std::endl; 

  A.deallocPreBuffer();
  if(spmv_type == 0) {
    A.holaPostprocess();
  }
  A.Destroy();

#ifdef CORRECT 
  if (hres)    free(hres);
#endif

  if (d_csr_row) cudaErrCheck(cudaFree(d_csr_row));
  if (d_csr_col) cudaErrCheck(cudaFree(d_csr_col));
  if (d_csr_val) cudaErrCheck(cudaFree(d_csr_val));
  if (d_x)       cudaFree(d_x);

  if (d_y)       cudaFree(d_y);
  if (d_y_key)       cudaFree(d_y_key);
  if (d_y_val)       cudaFree(d_y_val);

  if (d_x_key)   cudaFree(d_x_key);
  if (d_x_val)   cudaFree(d_x_val);
  if (d_bit_vector)   cudaFree(d_bit_vector);
  
  if(xnnz_vec)   free(xnnz_vec);
  fout.close();

return err;
}

int doThis(std::string file_name, int iter, float alpha, float beta) {
  const char* real_file_name = file_name.c_str();
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


  MTX<VALUE_TYPE> mtx;
  
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
  
  VALUE_TYPE* x_dense = (VALUE_TYPE* )malloc(n * sizeof(VALUE_TYPE));
  CHECK_MALLOC(x_dense);

  for(int i=0; i<n; i++) {
    x_dense[i] = (VALUE_TYPE)i;
  }

  VALUE_TYPE* y_dense = (VALUE_TYPE* )malloc(m * sizeof(VALUE_TYPE));
  CHECK_MALLOC(y_dense);
  
  // get matrix name.
  std::string matrix_name;
  int nPos1 = file_name.find_last_of(".");
  int nPos2 = file_name.find_last_of("/", nPos1 - 1);
  if(nPos1 != -1 && nPos2 != -1) {
    matrix_name = file_name.substr(nPos2 + 1, nPos1 - nPos2 - 1);
  }
  std::cout << "matrix_name = " << matrix_name << std::endl;
  std::cout << "m = " << m << ", n = " << n << ", nnz = " << mat_nnz << std::endl;

#ifdef BFS
  std::string out_file = "/home/*/framework-time/bfs/" 
                      + matrix_name + "_bfs_perf.info";
#else
  std::string out_file = "/home/*/framework-time/pagerank/" 
                      + matrix_name + "_pr_perf.info"; 
#endif 
  std::cout << out_file << std::endl;
  Run(out_file, matrix_name, iter, alpha, beta, m, n, mat_nnz, csr_row, csr_col, csr_val, 
      x_sparse_key, x_sparse_val, x_dense, y_dense);

  if (x_sparse_key) free(x_sparse_key);
  if (x_sparse_val) free(x_sparse_val);
  if (x_dense)      free(x_dense);
  if (y_dense)      free(y_dense);
#if 0
  if (mtx.row)   free(mtx.row);
  if (mtx.col)   free(mtx.col);
  if (mtx.data)  free(mtx.data);
#endif
  if(csr_row) free(csr_row);
  if(csr_col) free(csr_col);
  if(csr_val) free(csr_val);
  return 0;
}


int main(int argc, char** argv) {
  std::string file_name;
  int iter = 0;
  float alpha = 0, beta = 0;
  if (argc == 2) {
    file_name = argv[1];
    // iter = atoi(argv[2]);
    // alpha = atof(argv[3]);
    // beta = atof(argv[4]);
    std::cout << "---------file_name: " << file_name << "---------" << std::endl;
    // std::cout << "---------iter: " << iter << "---------" << std::endl;
    // std::cout << "---------alpha: " << alpha << "---------" << std::endl;
    // std::cout << "---------beta: " << beta << "---------" << std::endl;
  } else {
    std::cout << "Usage: matrix_file_name " << std::endl;
    exit(1);
  }
  //std::cout << "i am new-test-app.cu" << std::endl;
  doThis(file_name, iter, alpha, beta);
  
return 0;
}

