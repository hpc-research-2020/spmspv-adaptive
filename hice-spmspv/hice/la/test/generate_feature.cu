// This is the test source file of spmspv.

#include <iostream>
#include <string>
#include <float.h>
#include <cmath>
#include <typeinfo>
#include <limits>
#include <algorithm>
#include <vector>


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

void CountNonzeroPerRow(std::string file_name, const char* matrix_name, int m, int n, int mat_nnz,
                        int* csr_row, int* csr_col, VALUE_TYPE* csr_val) {
  std::ofstream fout;
  fout.open(file_name.c_str(), std::ofstream::app);
  if (fout.is_open()) {
    std::cout << "Output operation successfully performed\n";
  }

  /**********CountNonzeroPerRow ******************/
  int max = 0;
  int min = std::numeric_limits<int>::max();
  int positive_min = std::numeric_limits<int>::max();
  int count = 0;
  float avg_nnz_per_row = mat_nnz/(float)m;
  int num = 0; 
  float sum = 0.0;
  float gini_coefficiency = 0.0;
  float edge_equlity = 0.0;//
  double sum_edge_equlity = 0.0;
  std::vector<int> degrees;
  //std::vector<int> col_degrees;
  
  int i = 0;
  int j = 0;
  for (i = 0; i < m; i++) {
    num = csr_row[i+1] - csr_row[i];
    degrees.push_back(num);
    sum += (num - avg_nnz_per_row) * (num - avg_nnz_per_row);
    if(num != 0){
      sum_edge_equlity += -(num/(1.0*mat_nnz))*log(num/(1.0*mat_nnz));
      //std::cout << sum_edge_equlity << std::endl;
    }
    //fout << num << std::endl;
    if(num > max)
      max = num;
    if(num < min)
      min = num;
    if(num > 0 && num < positive_min)
      positive_min = num;
    if(num == 0)
      count++;
  }
  float standard_deviation_nnz_row = std::sqrt(sum/m);
  edge_equlity = (float)sum_edge_equlity * (1.0/log(m*1.0));
  
  std::sort(degrees.begin(), degrees.begin()+m);
  i = 0;
  j = 1;
  float sum_gini_1 = 0.0;
  float sum_gini_2 = 0.0;
  for (std::vector<int>::iterator it = degrees.begin() ; (it+1) != degrees.end(); ++it){ 
    sum_gini_1 += 1.0*j*(*(it+1)); 
    sum_gini_2 += 1.0*(*it);
    j++;
  }
  gini_coefficiency = (2.0*sum_gini_1)/(m*sum_gini_2) - (m+1)/(float)m; 
  if(gini_coefficiency < 0 ) gini_coefficiency *= -1; 
  
  // fout << "mat_name = " << matrix_name << " ";
  // fout << "m = " << m << " ";
  // fout << "n = " << n << " ";
  // fout << "nnz = " << mat_nnz << " ";
  // fout << "max nonzeros per row = " << max << " ";
  // fout << "min nonzeros per row = " << min << " ";
  // fout << "avg nonzeros per row = " << mat_nnz/m << " ";
  //fout << "num of zero rows= " << count << std::endl;
  std::cout << edge_equlity << std::endl;
  std::cout << gini_coefficiency << std::endl;

#if 1 
  fout << matrix_name << " ";
  fout << m << " ";
  fout << n << " ";
  fout << mat_nnz << " ";
  fout << max << " ";//max
  fout << min << " ";//min
  fout << mat_nnz/m << " ";//avg
  fout << (max-min)/(1.0*n) << " ";//relative range of degree
  fout << standard_deviation_nnz_row << " ";
  fout << edge_equlity << " ";
  fout << gini_coefficiency << std::endl;
#endif

#if 0 
  /**********CountNonzeroPerCol ******************/
  float avg_nnz_per_col = 0.0;
  sum = 0.0;

  int* csc_row = NULL;
  int* csc_col = NULL;
  VALUE_TYPE* csc_val = NULL;
  
  csc_row = (int* )malloc(mat_nnz * sizeof(int));
  csc_col = (int* )malloc((n + 1) * sizeof(int));
  csc_val = (VALUE_TYPE* )malloc(mat_nnz  * sizeof(VALUE_TYPE));
  CHECK_MALLOC(csc_row);
  CHECK_MALLOC(csc_col);
  CHECK_MALLOC(csc_val);

  TestCsr2Csc(m, n, mat_nnz, csr_row, csr_col, csr_val, 
                    csc_row, csc_col, csc_val);

  max = 0;
  min = std::numeric_limits<int>::max();
  positive_min = std::numeric_limits<int>::max();
  count = 0;
  num = 0;
  avg_nnz_per_col =  mat_nnz/(float)n;
  sum_edge_equlity = 0.0;

  for (i = 0; i < n; i++) {
    num = csc_col[i+1] - csc_col[i];
    col_degrees.push_back(num);
    sum += (num - avg_nnz_per_col)*(num - avg_nnz_per_col);
    sum_edge_equlity += -num/(2.0*mat_nnz)*log(num/(2.0*mat_nnz));
    //fout << num << std::endl;
    if (num > max)
      max = num;
    if (num < min)
      min = num;
    if(num > 0 && num < positive_min)
      positive_min = num;
    if (num == 0)
      count++;
  }
  float standard_deviation_nnz_col = std::sqrt(sum/n);
  edge_equlity = sum_edge_equlity * (1.0/log(m));
  
  std::sort(col_degrees.begin(), col_degrees.begin()+m);
  i = 1;
  sum_gini_1 = 0.0;
  sum_gini_2 = 0.0;
  for (std::vector<int>::iterator it = col_degrees.begin() ; it != col_degrees.end(); ++it){
    sum_gini_1 += 1.0*i*(*it); 
    sum_gini_1 += 2.0*(*it);
    i++;
  }
  gini_coefficiency = (2.0*sum_gini_1)/(m*sum_gini_2) - (m+1)/(float)m; 
  
  std::cout << edge_equlity << std::endl;
  std::cout << gini_coefficiency << std::endl;

  fout << max << " ";
  fout << min << " ";
  fout << mat_nnz/n << " ";
  fout << standard_deviation_nnz_col << " ";
  fout << edge_equlity << " ";
  fout << gini_coefficiency << " ";
  double all_elements = (double)m * n;
  double sparsity = (double)mat_nnz/all_elements;
  fout << sparsity << std::endl;

  if (csc_row)  free(csc_row);
  if (csc_col)  free(csc_col);
  if (csc_val)  free(csc_val);
#endif

  if (fout)     fout.close();
}

void TestMatInfo(const char* matrix_name, int m, int n, int mat_nnz, 
                int* csr_row, int* csr_col, VALUE_TYPE* csr_val) {
  CountNonzeroPerRow("/home/feature.info", matrix_name, m, n, mat_nnz, 
                      csr_row, csr_col, csr_val);
}

int doThis(std::string s_file_name, std::string dir_name) {
  const char* file_name = s_file_name.c_str(); 
  //std::cout << "-------" << file_name << "---------" << std::endl;
  std::string file = dir_name + "/" + s_file_name;
  //std::cout << file << std::endl;
  const char* real_file_name = file.c_str();
  std::cout << real_file_name << std::endl;

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

#ifndef READ_SVM_DATA

  MTX<VALUE_TYPE> mtx;
  //add fast reading:
  //std::string csr_name = std::string(real_file_name) + "_" + typeext<DataType>() + ".csr";
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

  TestMatInfo(file_name, m, n, mat_nnz, csr_row, csr_col, csr_val);

#ifndef READ_SVM_DATA
  //if (mtx.row)   free(mtx.row);
  //if (mtx.col)   free(mtx.col);
  //if (mtx.data)  free(mtx.data);
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#else
  if (csr_row)   free(csr_row);
  if (csr_col)   free(csr_col);
  if (csr_val)   free(csr_val);
#endif

return 0;
}

 //ref: https://blog.csdn.net/yangguangqizhi/article/details/50414029
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
