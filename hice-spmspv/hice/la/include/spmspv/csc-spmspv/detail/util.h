#ifndef SPMSPV_UTILS_H_
#define SPMSPV_UTILS_H_

#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <fstream>

#include "common.h"
#include "spmspv/csc-spmspv/sparse_vec.h"

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}


#define CUSP_CALL(func)                           \
 {                                                 \
   cusparseStatus_t e = (func);                    \
   if(e != CUSPARSE_STATUS_SUCCESS)                \
    std::cout << "CUSPerror: " << e << std::endl;  \
  }

#define checkCudaRoutineReturn(name) \
    err_r = cudaGetLastError();\
    if ( cudaSuccess != err_r) {\
      printf("%s ()invocate error.\n", name);\
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;\
    }


void CHECK_MALLOC(void* ptr){
  if(ptr == NULL) {
    std::cout<<"malloc failed on "<< __FILE__ <<", "<<__LINE__<< std::endl;
    exit(1);
  }
}

struct anonymouslib_timer {
    cudaEvent_t start_event, stop_event;

    void start() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, 0);
        cudaDeviceSynchronize();
    }

    float stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
        return elapsedTime;
    }
};

struct SpmspvTimer {
  cudaEvent_t start_event, stop_event;
  void Start() {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    cudaDeviceSynchronize();
  }
  float Stop() {
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
    return elapsedTime;
  }
};

// template<typename iT, typename vT>
// double getB(const iT m, const iT nnz)
// {
//     return (double)((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
// }

// template<typename iT>
// double getFLOP(const iT nnz)
// {
//     return (double)(2 * nnz);
// }

template<typename T>
void print_tile_t(T *input, int m, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int local_id = 0; local_id < m; local_id++)
        {
            std::cout << input[local_id * n + i] << ", ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_tile(T *input, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int local_id = 0; local_id < n; local_id++)
        {
            std::cout << input[i * n + local_id] << ", ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_1darray(T *input, int l)
{
    for (int i = 0; i < l; i++)
        std::cout << input[i] << ", ";
    std::cout << std::endl;
}

template<typename T>
void print_vec_file(T *input, int l, char* file_name)
{
    std::ofstream fout;
    fout.open(file_name);
    if (!fout.is_open()){
      std::cout<<"open file "<< file_name <<" failed."<<std::endl;
      exit(1);
    }
    
    for (int i = 0; i < l; i++)
        fout << input[i] << ", ";
    fout << std::endl;
    
    fout.close();

}

template<typename iT, typename vT>
double SpmvGetBytes(const iT m, const iT nnz) {
  return (double)((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

template<typename iT>
double SpmvGetFlops(const iT nnz) {
  return (double)(2 * nnz);
}

//arr[len+1]
void inner_ExclusiveScan(int* arr, int len) {
  int old_val, new_val;
  old_val = arr[0];
  arr[0] = 0;
  for (int i = 1; i <= len; i++) {
      new_val = arr[i];
      arr[i] = old_val + arr[i-1];
      old_val = new_val;
  }
}

template<typename iT, typename uiT, typename vT>
void PrintBucket2FileDevice(int len, iT* d_bin_row, vT* d_bin_val, 
                       const char* file_name) {
  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout<<"open file failed."<<std::endl;
      exit(1);
  }

  iT* bin_row = (iT*)malloc(len * sizeof(iT));
  vT* bin_val = (vT*)malloc(len * sizeof(vT));
  CHECK_MALLOC(bin_row);
  CHECK_MALLOC(bin_val);

  cudaErrCheck(cudaMemcpy(bin_row, d_bin_row, (len) * sizeof(iT),   
               cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(bin_val, d_bin_val, (len) * sizeof(vT), 
               cudaMemcpyDeviceToHost));   

  for (int i = 0; i < len; i++) {
      fout << bin_row[i] <<":" << bin_val[i] << std::endl;
      //fout << bin_row[i]+1 << std::endl;
  }
  fout << std::endl;
  
  fout.close();

  if (bin_row) free(bin_row);
  if (bin_val) free(bin_val);
}

void PrintMat2FileDevice(int row, int col, int* d_ptr_count, 
                         const char* file_name) {
  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout << "open file failed." << std::endl;
      exit(1);
  }
  int len = row * col + 1;
  int* ptr_count = (int*)malloc(len * sizeof(int));
  CHECK_MALLOC(ptr_count);

  cudaErrCheck(cudaMemcpy(ptr_count, d_ptr_count, (len) * sizeof(int), 
                          cudaMemcpyDeviceToHost)); 

  for (int j = 0; j < row; j++) {
    for(int i = 0; i < col; i++)
      fout << ptr_count[j * col + i] <<", ";
    fout << std::endl;   
  }
  fout << ptr_count[len - 1] << std::endl;
  fout << std::endl;

  fout.close();
  if (ptr_count) free(ptr_count);
}
void PrintDenseVec2File(int len, int* vec, const char* file_name) {
  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout << "open file failed." << std::endl;
      exit(1);
  }
  for (int j = 0; j < len; j++) {
      fout << vec[j] <<", ";  
  }
  fout << std::endl;
  fout.close();
}

void PrintDenseVec2FileDevice(int len, int* d_vec, const char* file_name) {
  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout << "open file failed." << std::endl;
      exit(1);
  }

  int* vec = (int*)malloc(len * sizeof(int));
  CHECK_MALLOC(vec);
  cudaErrCheck(cudaMemcpy(vec, d_vec, (len) * sizeof(int), 
               cudaMemcpyDeviceToHost)); 
  for(int j = 0; j < len; j++) {
      fout << vec[j] <<", ";  
  }
  fout << std::endl;
  fout.close();
  if (vec) free(vec);
}

#if 0
void PrintSparseVec2FileDevice(int len, SparseVec* d_vec, 
                               const char* file_name) {
  std::ofstream fout;
  fout.open(file_name);
  if (!fout.is_open()) {
      std::cout << "open file failed." << std::endl;
      exit(1);
  }
  SparseVec* vec = (SparseVec*)malloc(len * sizeof(SparseVec));
  CHECK_MALLOC(vec);
  cudaErrCheck(cudaMemcpy(vec, d_vec, (len) * sizeof(SparseVec), 
               cudaMemcpyDeviceToHost)); 

  for(int j = 0; j < len; j++) {
      fout << vec[j].key <<":"<< vec[j].val <<", ";  
  }
  fout << std::endl;

  fout.close();
  if (vec) free(vec);
}
#endif

#endif // SPMSPVUTILS_H_
