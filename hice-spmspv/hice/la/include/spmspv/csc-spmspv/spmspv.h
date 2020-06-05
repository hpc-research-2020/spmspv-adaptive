// This file provides the spmspv class.

#ifndef SPMSPV_H_
#define SPMSPV_H_

#include "helper_cuda.h"
#include "helper_functions.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


#include "spmspv/config.h"
//#include "sparse_vec.h"

#include "detail/common.h"
#include "detail/util.h"
#include "detail/device/csc_sort_scan_fuse.cuh"

//#include "detail/device/bcsc_spmspv.cuh"
//TODO: 
//#include "detail/device/csc_bucket_spmspv.cuh"

#ifdef BUCKET_LOAD_BALANCE 
#include "detail/device/csc_bucket_load_balance_spmspv.cuh"
#endif

#include "detail/device/csc_sort_load_balance.cuh"
#include "detail/device/csc_sort_load_balance_fuse.cuh"
#include "detail/device/csc_sort_load_balance_fuse_reduction.cuh"
#include "detail/device/csc_sort_load_balance_modify.cuh"
#include "detail/device/csc_two_layer_load_balance.cuh"
#include "detail/device/csc_two_layer_load_balance_fuse_reduction.cuh"
#include "detail/device/csc_two_layer_load_balance_fuse_reduction_modify.cuh"
#include "detail/device/csc_sort_load_balance_fuse_reduction_modify.cuh"
#include "detail/device/csc_sort_spmspv.cuh"

//#include "detail/device/csr_spmspv.cuh"

#include "detail/device/bitvector.h"

//for csr5
#include "detail/device/utils_kernel.h"
//#include "detail/cuda/common_cuda.h"
#include "detail/device/format_conversion.h"
#include "detail/device/csr5_spmv_cuda.h"
#include "detail/device/csr5_spmspv.h"

// for hola spmv.
#include "detail/device/holaspmv.h"

//for naive spmv.
#include "detail/device/naivespmv.h"

// #include <cub/cub.cuh>
// using namespace cub;
// CachingDeviceAllocator  g_allocator(true);

#define JUDGE_OPT 1

//for reduction operation in computeBinlenVer2()
#define REDUCE_BLOCKS 64

//Added for sm=700.
#define FULL_MASK 0xffffffff
#define __shfl(val, offset) __shfl_sync(FULL_MASK, val, offset)
#define __shfl_down(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define __shfl_up(val, offset) __shfl_up_sync(FULL_MASK, val, offset)
#define __any(predicate) __any_sync(FULL_MASK, predicate)
#define __ballot(predicate) __ballot_sync(FULL_MASK, predicate)


template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
class SpmspvHandle {
 public:
  SpmspvHandle(TYPE_IT m, TYPE_IT n, TYPE_IT nnz) {
    m_ = m; 
    n_ = n; 
    nnz_ = nnz;
    
    format_ = -1;
    vec_type_ = -1;
    spmv_type_ = 0;//

    //csr format.
    d_csr_row_ = NULL;
    d_csr_col_ = NULL;
    d_csr_val_ = NULL;

    //csc format: malloc in this file.
    d_csc_col_ = NULL;
    d_csc_row_ = NULL;
    d_csc_val_ = NULL;
    
    //vector format.
    d_x_ = NULL;
    d_x_sparse_key_ = NULL;
    d_x_sparse_val_ = NULL;
    
    num_ints_ = (n_ + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    d_bit_vector_ = NULL;
    
    d_buffer_ = NULL;
    d_ptr_col_len_ = NULL; 
    d_ptr_pre_alloc_buffer_ = NULL;
    h_odata_ = NULL;
    d_odata_ = NULL;
    
    //for kernel selection.
    stdRow_ = 0;
    isStdRowReady_ = false;

    bin_len_ = -1;
    isBinlenReady_ = false;
    GM1_ = -1;
    GM2_ = -1;
    GM3_ = -1;
    
    // //bcsc format.
    // d_bcsc_val_ = NULL;
    // d_bcsc_row_ = NULL;
    // d_bcsc_col_ = NULL;
    // d_bcsc_group_ptr_ = NULL;
    holatemp_req_ = 0;
    dholatempmem_ = NULL;
  }

  int warmup();
  int InputCSR(TYPE_IT  nnz, TYPE_IT *csr_row, 
               TYPE_IT *csr_col, TYPE_VT *csr_sval);
  int InputCSC(TYPE_IT  nnz, TYPE_IT *csc_row, 
               TYPE_IT *csc_col, TYPE_VT *csc_sval);
  int ToCSC();
  float extractMatFeature(int* csr_row/*, int* max_nnz, int* min_nnz,*/ /*float* avg_nnz_per_row, float* sd_nnz*/);
  float extractMatFeatureParallel(int* csr_row/*, int* max_nnz, int* min_nnz,*/ /*float* avg_nnz_per_row, float* sd_nnz*/);
  // int InputCoo(TYPE_IT  nnz, TYPE_IT *coo_row, 
  //              TYPE_IT *coo_col, TYPE_VT *coo_val);
  //int ToBcsc(int group_num);
  //int ToHybrid(int threshold, int nrows, int ncols, int nnz, TYPE_IT* coo_row, TYPE_IT* coo_col, TYPE_VT* coo_val);

  //csr5's set vector.
  int setX(TYPE_VT *x);
  int set_bitvector_x(TYPE_IT *x_key, TYPE_VT* x_dense, TYPE_IT xnnz);
  //csr'5 set vector: NEED TO MODIFY

  //alloc buffer for d_ptr_pre_alloc_buffer_ and d_ptr_col_len_.
  int allocPreBuffer();
  int deallocPreBuffer();

  int set_x(TYPE_VT *x);
  int set_sparse_x(TYPE_IT *x_key, TYPE_VT* x_val, TYPE_IT xnnz);
  int set_bitvector(TYPE_IT* bitvector, TYPE_IT num_ints);
  //input sparse_x and dense_x.
  int set_sparse_x_dense_x(TYPE_IT *x_key, TYPE_VT* x_val, 
                            TYPE_VT* x_dense, TYPE_IT xnnz);

  //TODO: modify code's type to enum.
  int set_vector_type(int code);
  int set_spmv_type(int code);

  /******vector format conversion********************/
  int dense2sparse();
  int sparse2dense();
  int sparse2bitarray();
  int bitarray2sparse();//same to dense2sparse.
  int dense2bitarray();
  int dense2sparse_outer(TYPE_IT m, TYPE_VT* d_dense, TYPE_IT* ynnz, TYPE_IT* d_key, TYPE_VT* d_val);
  int sparse2dense_outer(TYPE_IT ynnz, TYPE_IT* d_key, TYPE_VT* d_val, TYPE_IT m, TYPE_VT* d_dense);
  //bitarray2dense is not needed to implemented.

  /******vector format conversion********************/
  //by calling cub's sum()
  long long computeBinlen();
  //by using my own reduction();
  long long computeBinlenVer2();
  long long getBinlen();
  double computeGM1();
  double computeGM2();
  double computeGM3();
  double getGM1();
  double getGM2();
  double getGM3();
  //int fill();
  //void computeVecFeture_serial(x_nnz, x_sparse_key, csc_col, &bin_len, &max_elems, &min_elems);
  void computeVecFeture_serial(int x_nnz, int* x_sparse_key, int* bin_len, int* max_elems, int* min_elems);
  
  void computeMatFeture_serial(int m, int n, int mat_nnz, int* csr_row, int* max_elems, int* min_elems, float* avg_elems, float* x_range, float* standard_row, float* equlity, float* gini);



  // for csr5 spmv and csr5 spmspv.
  void setSigma(int sigma);
  int CSR5Preprocess();//csr2csr5
  int csr5spmv(const TYPE_VT alpha, TYPE_VT *y);
  int csr5spmspv(const TYPE_VT alpha, TYPE_VT *y);
  int CSR5Postprocess();
 
  // for hola spmv and hola spmspv.
  int holaPreprocess();
  int holaspmv(const TYPE_VT alpha, TYPE_VT *y);
  int holaspmspv(const TYPE_VT alpha, TYPE_VT *y);
  int holaPostprocess();
  // end of hola spmv and hola spmspv.
 
  //naive spmv series.
  int naivespmv(const TYPE_VT alpha, TYPE_VT *y);
  int naivespmspv(const TYPE_VT alpha, TYPE_VT *y);
  
  int spmv(const TYPE_VT alpha, TYPE_VT *y);
  int spmspv(const TYPE_VT alpha, TYPE_VT *y);

  //DEPRECATED
  //for csr-based smsv: matrix-driven.
  //int CsrBasedSpmspv(const TYPE_VT alpha, TYPE_VT *y);

  
  // int CscBasedBucketSpmspv(const TYPE_VT alpha,  TYPE_IT* y_nnz, TYPE_VT* y, 
  //                       int buckets);

  //the final version-1: dense output
  int CscBasedMyNoSortSpmspv(bool IsBinLenComputed, const TYPE_VT alpha, TYPE_IT* y_nnz, TYPE_VT* y, 
                            TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val);
  //the final version-1: sparse output
  int CscBasedMyNoSortSpmspv_keyval(bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, 
              TYPE_VT* d_y, TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val);
  //the final version-3: 排序，sparse output 
  //TODO: remove d_y. 
  int CscBasedSortMySpmspv(bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, 
              TYPE_VT* d_y, TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val);

  //cub's extract.
  int CscBasedNoSortMergeSpmspv(bool IsBinLenComputed, const TYPE_VT alpha, TYPE_IT* y_nnz, TYPE_VT* y, 
                               TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val);
  int CscBasedNoSortMergeSpmspv_keyval(bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, 
              TYPE_VT* d_y, TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val);
  //new added.
  //TODO: remove d_y.
  int CscBasedSortMergeSpmspv(bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, 
              TYPE_VT* d_y, TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val);           

  //naive extract.
  int CscBasedNoSortNaiveSpmspv(const TYPE_VT alpha, TYPE_IT* y_nnz, TYPE_VT* y);
  int CscBasedNoSortNaiveSpmspv_keyval(const TYPE_VT  alpha, TYPE_IT* ynnz, 
              TYPE_VT* d_y, TYPE_IT* d_x_sparse_key, TYPE_VT*d_x_sparse_val);
  //new added.
  int CscBasedSortNaiveSpmspv(const TYPE_VT alpha,  TYPE_IT* y_nnz,      
                            TYPE_IT* y_inx, TYPE_VT* y_val, int method);

  // //blocked csc format based spmspv.
  // int BcscBasedSpmspv(const TYPE_VT alpha, TYPE_VT *y);
  int Destroy();

 private:
  int computeSigma();
  int format_;
  int vec_type_;
  int spmv_type_;

  TYPE_IT m_;
  TYPE_IT n_;
  TYPE_IT nnz_;//TODO: overflow?
  
  float stdRow_;
  bool isStdRowReady_;

  TYPE_IT* d_csr_row_;
  TYPE_IT* d_csr_col_;
  TYPE_VT* d_csr_val_;

  // TYPE_IT* csc_row_;
  // TYPE_IT* csc_col_;
  // TYPE_VT* csc_val_;
  TYPE_IT* d_csc_row_;
  TYPE_IT* d_csc_col_;
  TYPE_VT* d_csc_val_;

  //GOOGLE:
  float* d_buffer_;
  int* d_ptr_col_len_;
  void* d_ptr_pre_alloc_buffer_;

  //dense x on device
  TYPE_VT* d_x_;  //dense vector
  TYPE_IT* d_x_sparse_key_;
  TYPE_VT* d_x_sparse_val_;
  TYPE_IT x_nnz_;
  
  int num_ints_;
  int* d_bit_vector_;
  
  //temp buffer for reduction.
  float* h_odata_;
  float* d_odata_;

  long long int bin_len_;
  bool isBinlenReady_;
  double GM1_;
  double GM2_;
  double GM3_;

  // TYPE_IT* d_bcsc_row_;
  // TYPE_IT* d_bcsc_col_;
  // TYPE_VT* d_bcsc_val_;
  // //[rowgroupNum + 1]
  // TYPE_IT* d_bcsc_group_ptr_;
  //TYPE_IT group_num_;

  //for csr5
  int         _csr5_sigma;
  int         _bit_y_offset;
  int         _bit_scansum_offset;
  int         _num_packet;
  TYPE_IT _tail_partition_start;

  TYPE_IT _p;
  TYPE_UIT *_csr5_partition_pointer;
  TYPE_UIT *_csr5_partition_descriptor;

  TYPE_IT   _num_offsets;
  TYPE_IT  *_csr5_partition_descriptor_offset_pointer;
  TYPE_IT  *_csr5_partition_descriptor_offset;
  TYPE_VT  *_temp_calibrator;

  cudaTextureObject_t  _x_tex;

  //for hola 
  size_t holatemp_req_;
  void* dholatempmem_;
};

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::warmup() {
  //in format_conversion.h
  format_warmup();
  return 0;
}

//input already in device mem.
//TODO: remove nnz parameter!!
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::InputCSR(TYPE_IT  nnz,
                                                      TYPE_IT *csr_row,
                                                      TYPE_IT *csr_col,
                                                      TYPE_VT *csr_val) {
  format_ = SPMSPV_FORMAT_CSR;
  nnz_ = nnz;
  d_csr_row_  = csr_row;
  d_csr_col_  = csr_col;
  d_csr_val_  = csr_val;
return SPMSPV_SUCCESS;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::InputCSC(TYPE_IT  nnz,
                                                      TYPE_IT *csc_row,
                                                      TYPE_IT *csc_col,
                                                      TYPE_VT *csc_val) {
  format_ = SPMSPV_FORMAT_CSC;
  nnz_ = nnz;
  d_csc_row_  = csc_row;
  d_csc_col_  = csc_col;
  d_csc_val_  = csc_val;
return SPMSPV_SUCCESS;
}


// format conversion: csr2csc
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::ToCSC() {
  int err = SPMSPV_SUCCESS;

  cudaErrCheck(cudaMalloc((void **)&d_csc_row_, nnz_ * sizeof(TYPE_IT)));
  cudaErrCheck(cudaMalloc((void **)&d_csc_val_, nnz_  * sizeof(TYPE_VT)));
  cudaErrCheck(cudaMalloc((void **)&d_csc_col_, 
              (n_ + 1) * sizeof(TYPE_IT)));    

  cusparseHandle_t sparse_handle;
  CUSP_CALL(cusparseCreate(&sparse_handle));
  cusparseMatDescr_t descr = 0;
  CUSP_CALL(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  cudaErrCheck(cudaDeviceSynchronize());

#ifdef DOUBLE
    CUSP_CALL(cusparseDcsr2csc(sparse_handle, 
                  m_, n_, nnz_, 
                  d_csr_val_, d_csr_row_, d_csr_col_, 
                  d_csc_val_, d_csc_row_, d_csc_col_, 
                  CUSPARSE_ACTION_NUMERIC, 
                  CUSPARSE_INDEX_BASE_ZERO));
#else
    CUSP_CALL(cusparseScsr2csc(sparse_handle, 
                  m_, n_, nnz_, 
                  d_csr_val_, d_csr_row_, d_csr_col_, 
                  d_csc_val_, d_csc_row_, d_csc_col_, 
                  CUSPARSE_ACTION_NUMERIC, 
                  CUSPARSE_INDEX_BASE_ZERO));
#endif
  cudaErrCheck(cudaDeviceSynchronize());

  return err;
}

//serial version.
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
float SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::extractMatFeature(
            int* csr_row/*, int* max_nnz, int* min_nnz,*/ 
            /*float avg_nnz_per_row, float* sd_nnz*/){
  // int max = 0;
  // int min = std::numeric_limits<int>::max();
  int count = 0;
  int avg_nnz = (nnz_)/(float)m_;

  int num = 0;
  float sum = 0.0;
  for (int i = 0; i < m_; i++) {
    num = csr_row[i+1] - csr_row[i];
    sum += (num - avg_nnz) * (num - avg_nnz);
    // if(num > max)  max = num;
    // if(num < min)  min = num;
  }
  // *max_nnz = max;
  // *min_nnz = min;
  //*avg_nnz_per_row = avg_nnz;
  //*sd_nnz = std::sqrt(sum/m_);
  return std::sqrt(sum/m_);
}

//parallel version of extractmatfeature.
#if 1
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

//
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduceMatFeature(T *g_idata, /*T *g_odata, T *g_odata1,*/ float *g_odata2, 
                unsigned int n, float avg_nnz)
{
#if 1
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int threads = blockSize;
    float * shamem = SharedMemory<float>();
    //T *sdata = (T*)shamem;//max
    //T *sdata1 = (T*)(shamem + threads);//min
    //float *sdata2 = (float*)(shamem + 2* threads);//std_nnz
    float *sdata2 = (float*)shamem;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    // T myMax = -1;
    // T myMin = (T)INT_MAX;
    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        //T temp_i = g_idata[i];
        T temp_i = g_idata[i+1] - g_idata[i];
        float temp = temp_i - avg_nnz;
        mySum += temp * temp;

        // if(temp_i > myMax)
        //     myMax = temp_i;
        // if(temp_i < myMin)
        //     myMin = temp_i;
#ifdef DEBUG
        if(threadIdx.x == 0/* && blockIdx.x == 0*/)
          printf("%d, %d, %f.\n", myMax, myMin, mySum);
#endif
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n){
          //mySum += g_idata[i+blockSize];
          temp_i = g_idata[i+1+blockSize]-g_idata[i+blockSize];
          temp = temp_i - avg_nnz;
          mySum += temp * temp;

          // if(temp_i > myMax)
          //   myMax = temp_i;
          // if(temp_i < myMin)
          //   myMin = temp_i;
#ifdef DEBUG
          if(threadIdx.x == 0/* && blockIdx.x == 0*/)
            printf("2: %d, %d, %f.\n", myMax, myMin, mySum);
#endif
        }
        i += gridSize;
    }
#ifdef DEBUG
    if(threadIdx.x == 0/* && blockIdx.x == 0*/)
      printf("outer: %d, %d, %f.\n", myMax, myMin, mySum);
#endif
    // each thread puts its local sum into shared memory
    // sdata[tid] = myMax;
    // sdata1[tid] = myMin; 
    sdata2[tid] = mySum;

#ifdef DEBUG
    //TODO: wrong!!
    if(threadIdx.x == 0)
      printf("sdata[0] = %d\n", sdata[tid]);
#endif

    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
      //int temp = sdata[tid + 256];
      //int temp1 = sdata1[tid + 256];
      float temp2 = sdata2[tid + 256];

      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }

      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum + temp2;
#endif      
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
      // int temp = sdata[tid + 128];   
      // int temp1 = sdata1[tid + 128];
      float temp2 = sdata2[tid + 128];
   
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      sdata2[tid] = mySum = mySum + temp2;
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
      // int temp = sdata[tid + 64];
      // int temp1 = sdata1[tid + 64];
      float temp2 = sdata2[tid + 64];
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
      
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      sdata2[tid] = mySum = mySum +  temp2;
    }

    cg::sync(cta);

// #if (__CUDA_ARCH__ >= 300 )
//     if ( tid < 32 )
//     {
//         //TODO: 
//         cg::coalesced_group active = cg::coalesced_threads();

//         // Fetch final intermediate sum from 2nd warp
//         if (blockSize >=  64) mySum += sdata2[tid + 32];
//         // Reduce final warp using shuffle
//         for (int offset = warpSize/2; offset > 0; offset /= 2) 
//         {
//              mySum += active.shfl_down(mySum, offset);
//         }
//     }
// #else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
      // int temp = sdata[tid + 32];
     
      // int temp1 = sdata1[tid + 32];
      #if 1
      float temp2 = sdata2[tid + 32];
      #endif
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
      
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum +  temp2;
      #endif
      //sdata2[tid] = mySum = mySum + sdata2[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
      // int temp = sdata[tid + 16];
     
      // int temp1 = sdata1[tid + 16];
      #if 1
      float temp2 = sdata2[tid + 16];
      #endif
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
      
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum +  temp2;
      #endif
      //sdata2[tid] = mySum = mySum + sdata2[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    { 
      // int temp = sdata[tid + 8];
     
      // int temp1 = sdata1[tid + 8];
      #if 1
      float temp2 = sdata2[tid + 8];
      #endif
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
     
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum +  temp2;
      #endif
      //sdata2[tid] = mySum = mySum + sdata2[tid +  8];
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
      // int temp = sdata[tid + 4];
     
      // int temp1 = sdata1[tid + 4];
      #if 1
      float temp2 = sdata2[tid + 4];
      #endif
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
     
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum +  temp2;
      #endif
      //sdata2[tid] = mySum = mySum + sdata2[tid +  4];
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
      // int temp = sdata[tid + 2];
      
      // int temp1 = sdata1[tid + 2];
      #if 1
      float temp2 = sdata2[tid + 2];
      #endif
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
      
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum +  temp2;
      #endif
      //sdata2[tid] = mySum = mySum + sdata2[tid +  2];
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
      // int temp = sdata[tid + 1];
      
      // int temp1 = sdata1[tid + 1];
      #if 1
      float temp2 = sdata2[tid + 1];
      #endif
      // if(temp > myMax){
      //   sdata[tid] = myMax = temp;
      // }
      // if(temp1 < myMin){
      //   sdata1[tid] = myMin = temp1;
      // }
      #if 1
      sdata2[tid] = mySum = mySum +  temp2;
      #endif
      //sdata2[tid] = mySum = mySum + sdata2[tid +  1];
    }

    cg::sync(cta);
// #endif

    // write result for this block to global mem
    if (tid == 0){
      // g_odata[blockIdx.x] = myMax;
      // g_odata1[blockIdx.x] = myMin;
      #if 1
      g_odata2[blockIdx.x] = mySum;
      #endif
    }
#endif
}


template <class T>
void
reduceMatFeatureWrapper(int size, int threads, int blocks,
       T *d_idata, /*T *d_odata, T *d_odata1,*/ float *d_odata2, 
       float avg_nnz)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    //int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    //when threads=64, then smemsize = 768 bytes.
    int smemSize = (threads <= 32) ? 
                   (4 * threads * sizeof(T) + 2 * threads * sizeof(float)) : 
                   (2 * threads * sizeof(T) + threads * sizeof(float));
    //printf("smemsize = %d\n", smemSize);
    
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduceMatFeature<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 256:
                reduceMatFeature<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 128:
                reduceMatFeature<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 64:
                reduceMatFeature<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 32:
                reduceMatFeature<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 16:
                reduceMatFeature<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  8:
                reduceMatFeature<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  4:
                reduceMatFeature<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  2:
                reduceMatFeature<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  1:
                reduceMatFeature<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceMatFeature<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 256:
                reduceMatFeature<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 128:
                reduceMatFeature<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 64:
                reduceMatFeature<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 32:
                reduceMatFeature<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case 16:
                reduceMatFeature<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  8:
                reduceMatFeature<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  4:
                reduceMatFeature<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  2:
                reduceMatFeature<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;

            case  1:
                reduceMatFeature<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, /*d_odata, d_odata1,*/ d_odata2, size, avg_nnz);
                break;
        }
    }
}

template <class T>
T extractMatFeatureParallelInter(int  n,
                  int  numThreads,
                  int  numBlocks,
                  // T *h_odata,
                  // T *h_odata1,
                  float *h_odata2,
                  T *d_csr_row,
                  // T *d_odata,
                  // T *d_odata1,
                  float *d_odata2,
                  // T& max, T& min, 
                  float& std_val, float avg_nnz)
{
    // T gpu_max = -1;
    // T gpu_min = (T)INT_MAX;
    float gpu_std_val = 0.0;

    // execute the kernel
    reduceMatFeatureWrapper<T>(n, numThreads, numBlocks, d_csr_row, /*d_odata, d_odata1,*/ d_odata2, avg_nnz);

    // check if kernel execution generated an error
    //getLastCudaError("Kernel execution failed");
    cudaError_t err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("reduceMatFeature() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    // sum partial sums from each block on CPU
    // copy result from device to host
    // cudaErrCheck(cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(T), cudaMemcpyDeviceToHost));
    // cudaErrCheck(cudaMemcpy(h_odata1, d_odata1, numBlocks*sizeof(T), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(h_odata2, d_odata2, numBlocks*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<numBlocks; i++)
    {
        //std::cout << h_odata[i] << std::endl;
        // if(h_odata[i] > gpu_max)
        //   gpu_max = h_odata[i];
        // if(h_odata1[i] < gpu_min)
        //   gpu_min = h_odata1[i];

        gpu_std_val += h_odata2[i];
    }
    //printf("gpu_std_val = %lf\n", gpu_std_val);

    std_val = sqrt(gpu_std_val/n);
    // max = gpu_max;
    // min = gpu_min;

    return 0;
}

#endif

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
float SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>:: extractMatFeatureParallel(
                int* csr_row/*, int* max_nnz, int* min_nnz, 
                float* avg_nnz_per_row, float* sd_nnz*/){
  // int max = 0;
  // int min = std::numeric_limits<int>::max();
  if(!isStdRowReady_){

    float std_val = 0.0;
    int avg_nnz = nnz_/(float)m_;

    if(m_ < 8192){
      std_val = extractMatFeature(csr_row/*, max_nnz, min_nnz,*/ /*avg_nnz, sd_nnz*/);
    }else{
      //*avg_nnz_per_row = avg_nnz;

      //TODO: 
      int threads = 64;
      int blocks = REDUCE_BLOCKS;
      
      //float* h_space = (float*) malloc(blocks*3*sizeof(float));
      //  int *h_odata = (int *)h_space;
      //  int *h_odata1 = (int *)(h_space + blocks);
      //float *h_odata2 = (float*)(h_space + 2 * blocks);
      float* h_odata2 =  (float*)(h_odata_);

      //  float *d_space = NULL;
      //  cudaErrCheck(cudaMalloc((void **) &d_space, blocks*3*sizeof(float)));
      //  int *d_odata = (int*)d_space;
      //  int *d_odata1 = (int*)(d_space + blocks);
      //  float *d_odata2 = (float*)(d_space + blocks * 2);
      float* d_odata2 = (float*)d_odata_;

  #if 1
      //  SpmspvTimer timer;
      //  timer.Start();
      extractMatFeatureParallelInter<int>(m_, threads, blocks, 
                /*h_odata, h_odata1, */h_odata2, 
                d_csr_row_, /*d_odata, d_odata1,*/ d_odata2,
                /*max, min,*/ std_val, avg_nnz);
      // double time = timer.Stop();
      // printf("extractMatFeatureParallelInter time is %lf\n", time);
  #endif   
      
      // if(h_space) free(h_space);
      // if(d_space) cudaErrCheck(cudaFree(d_space));

      // *max_nnz = max;
      // *min_nnz = min;
      // *avg_nnz_per_row = avg_nnz;
      // *sd_nnz = std_val;
    }
    stdRow_ = std_val;
    isStdRowReady_ = true;
  } 
  return stdRow_;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::set_vector_type(int code) {
  vec_type_ = code;
return 0;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::set_x(TYPE_VT* x) {
  int err = SPMSPV_SUCCESS;
  d_x_ = x;
return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::set_sparse_x(
                              TYPE_IT* x_key, TYPE_VT* x_val, 
                              TYPE_IT xnnz) {
  int err = SPMSPV_SUCCESS;
  x_nnz_ = xnnz;
  //std::cout << "x_nnz_ = " << x_nnz_ << std::endl;
  d_x_sparse_key_ = x_key;
  d_x_sparse_val_ = x_val;

return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::set_bitvector(
                      TYPE_IT* bitvector, TYPE_IT num_ints) {
  int err = SPMSPV_SUCCESS;
  d_bit_vector_ = bitvector;
  num_ints_ = num_ints;                      
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::set_sparse_x_dense_x(
                      TYPE_IT* x_key, TYPE_VT* x_val, TYPE_VT* x_dense,
                      TYPE_IT xnnz) {
  int err = SPMSPV_SUCCESS;
  x_nnz_ = xnnz;
  
  d_x_sparse_key_ = x_key;
  d_x_sparse_val_ = x_val;
  
  d_x_ = x_dense;
return err;
}

//csr5
//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::setX(TYPE_VT *x) {
    int err = SPMSPV_SUCCESS;

    d_x_ = x;

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_x_;
    resDesc.res.linear.sizeInBytes = n_ * sizeof(TYPE_VT);
    if (sizeof(TYPE_VT) == sizeof(float)) {
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
    }
    else if (sizeof(TYPE_VT) == sizeof(double)) {
        resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.desc.y = 32; // bits per channel
    }
    else {
        return SPMSPV_UNSUPPORTED_VALUE_TYPE;
    }
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to do this once!
    _x_tex = 0;
    cudaCreateTextureObject(&_x_tex, &resDesc, &texDesc, NULL);

    return err;
}

//NOTE: d_bit_vector需要在外部进行内存分配。内部没有对其进行内存的分配。
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, 
TYPE_UIT, TYPE_VT>::set_bitvector_x(
                      TYPE_IT* x_key, TYPE_VT* x_dense,
                      TYPE_IT xnnz) {
  int err = SPMSPV_SUCCESS;
  
  d_x_ = x_dense;
// create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_x_;
  resDesc.res.linear.sizeInBytes = n_ * sizeof(TYPE_VT);
  if (sizeof(TYPE_VT) == sizeof(float)) {
      resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
      resDesc.res.linear.desc.x = 32; // bits per channel
  }
  else if (sizeof(TYPE_VT) == sizeof(double)) {
      resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
      resDesc.res.linear.desc.x = 32; // bits per channel
      resDesc.res.linear.desc.y = 32; // bits per channel
  }
  else {
      return SPMSPV_UNSUPPORTED_VALUE_TYPE;
  }

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  _x_tex = 0;
  cudaCreateTextureObject(&_x_tex, &resDesc, &texDesc, NULL);

  //for sparse info.
  x_nnz_ = xnnz;
  d_x_sparse_key_ = x_key;

  num_ints_ = (n_ + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  
  //cudaErrCheck(cudaMalloc((void **)&d_bit_vector_,  num_ints_ * sizeof(int)));
 
  cudaErrCheck(cudaMemset(d_bit_vector_, 0, num_ints_ * sizeof(int)));
  
  //TODO: 
  // int threads =  128;
  // int blocks = (x_nnz_ + threads - 1)/threads;
  int threads =  1;
  int blocks = 1;
  set_sparse_x_bitvector<<<blocks, threads>>>
                        (d_x_sparse_key_, x_nnz_, d_bit_vector_);                         
  return err;
}

/***********************vector format conversion***********************/
template <typename VT>
__global__ void updateFlagKernel(VT identity,
                                int n,
                                const VT* dense_vec,       
                                int*  d_flag) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  //one thread deals with one element in a dense vec.
  for( ; row < n; row += gridDim.x * blockDim.x) {
    VT val = __ldg(dense_vec + row);
    if(val == identity)
      d_flag[row] = 0;
    else
      d_flag[row] = 1;
  }
}

// generage keys and vals of nonzeroes.
template <typename VT>
__global__ void streamCompactKeyValKernel(int*     x_key,
                                          VT*        x_val,
                                          const int* d_scan,
                                          VT         identity,
                                          const VT*  x_dense,
                                          int        n) {
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  //one thread deals with one element in a dense vec.
  for( ; row < n; row += gridDim.x * blockDim.x ) {
    int pos = __ldg( d_scan + row );
    VT val  = __ldg( x_dense + row );

    if(val != identity) {
      x_key[pos] = row;
      x_val[pos] = val;
    }
  }
}

//only generage keys of nonzeroes.
template <typename VT>
__global__ void streamCompactKeyKernel( int*     x_key,
                                      const int* d_scan,
                                      VT         identity,
                                      const VT*  x_dense,
                                      int        n) {
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  //one thread deals with one element in a dense vec.
  for( ; row < n; row += gridDim.x * blockDim.x ) {
    int pos = __ldg( d_scan + row );
    VT val  = __ldg( x_dense + row );
    if(val != identity) {
      x_key[pos] = row;
    }
  }
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::dense2sparse() {
  int err = SPMSPV_SUCCESS;

  // 2*n_*sizeof(int/float)
  //desc->resize((2*nvals)*max(sizeof(int),sizeof(TYPE_VT)), "buffer");
  // if( vec_type_ != 1 ) {
  //   std::cout << "input is not dense type!" << std::endl;
  //   exit(0);
  // }

  //std::cout << "Converting from dense to sparse!\n";

  //int nt = 256;
  int NT = 256;
  int NB = (n_ + NT - 1)/NT;

  int* d_flag = (int*) d_ptr_pre_alloc_buffer_;
	int* d_scan = (int*) d_ptr_pre_alloc_buffer_ + n_;

	updateFlagKernel<<<NB, NT>>>(0.f, n_, d_x_, d_flag);
  
#ifndef USE_CUB_SCAN
  thrust::device_ptr<int> d_flag_ptr = thrust::device_pointer_cast(d_flag);
  thrust::device_ptr<int> d_scan_ptr = thrust::device_pointer_cast(d_scan);
  ////thrust::exclusive_scan(thrust::device, d_ptr, d_ptr+x_nnz+1, d_ptr);
  thrust::exclusive_scan(d_flag_ptr, d_flag_ptr + n_, d_scan_ptr);
#else
  void* d_temp_storage = NULL;
  size_t  temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_flag, d_scan, n_));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
               temp_storage_bytes));
  
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,   
                                        d_flag, d_scan, n_));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

  streamCompactKeyValKernel<<<NB, NT>>>(d_x_sparse_key_, 
        d_x_sparse_val_, d_scan, (TYPE_VT)0, d_x_, n_);

  vec_type_ = 0; 
  
  return err;
}

// KEYVALUE scatter
template <typename VT>
__global__ void scatter(int x_nnz,
                        const int* sparse_key,
                        const VT*  sparse_val,
                        VT*  dense_vec) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < x_nnz) {
    int ind = __ldg(sparse_key + gid);
    VT  val = __ldg(sparse_val + gid);
    dense_vec[ind] = val;//no atomic operation.
  }
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::sparse2dense() {
  int err = SPMSPV_SUCCESS;
  // if( vec_type_ != 0 ) {
  //   std::cout << "input is not sparse type!" << std::endl;
  //   exit(0);
  // };

  // std::cout << "Converting from sparse to dense!\n";
 
  const int nt = 256;
  const int nvals = x_nnz_;

  int NT = nt;
  int NB = (nvals + nt -1)/nt;
  
  cudaErrCheck(cudaMemset(d_x_, 0, n_ * sizeof(TYPE_VT)));

  scatter<<<NB, NT>>>(nvals, d_x_sparse_key_, d_x_sparse_val_, d_x_);
  
  // cudaError_t err_r = cudaGetLastError();
  // if ( cudaSuccess != err_r) {
  //   printf("sparse2dense() invocate error.\n");
  //   printf("xnnz=%d, nb=%d, bt=%d.\n", x_nnz_, NB, NT);
  //   std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
  // }

  vec_type_  = 1; //dense type.
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::sparse2bitarray() {
  int err = SPMSPV_SUCCESS;
  // if( vec_type_ != 0 ) {
  //   std::cout << "input is not sparse type!" << std::endl;
  //   exit(0);
  // };
  //std::cout << "Converting from sparse to bitarray: call sparse2dense and then generate bits.!\n";
  
  /*******step 1: call sparse2dense ******/
  const int nt = 256;
  const int nvals = x_nnz_;

  int NT = nt;
  int NB = (nvals + nt -1)/nt;
  
  cudaErrCheck(cudaMemset(d_x_, 0, n_ * sizeof(TYPE_VT)));
  cudaErrCheck(cudaMemset(d_bit_vector_, 0, num_ints_ * sizeof(int)));

  // generate d_x[]
  scatter<<<NB, NT>>>(nvals, d_x_sparse_key_, d_x_sparse_val_, d_x_);
  
  /*******step 2: set bit_vector: to optimize******/
  NT = 256;
  NB = (x_nnz_ + NT - 1)/ NT;
  
  set_sparse_x_bitvector<<<NB, NT>>>(d_x_sparse_key_, x_nnz_, d_bit_vector_);          
  
  // cudaError_t err_r = cudaGetLastError();
  // if ( cudaSuccess != err_r) {
  //   printf("sparse2bitarray() invocate error.\n");
  //   printf("nb=%d, bt=%d.\n", NB, NT);
  //   std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
  // }

  vec_type_  = 2; //bitarray type.
  return err;
}

//TODO: same to dense2sparse.
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::bitarray2sparse() {
  int err = SPMSPV_SUCCESS;
  if( vec_type_ != 2 ) {
    std::cout << "input is not bitarray type!" << std::endl;
    exit(0);
  };

  std::cout << "Converting from bitarray to sparse: call dense2sparse.\n";

  //int nt = 256;
  int NT = 256;
  int NB = (n_ + NT - 1)/NT;

  int* d_flag = (int*) d_ptr_pre_alloc_buffer_;
	int* d_scan = (int*) d_ptr_pre_alloc_buffer_ + n_;

	updateFlagKernel<<<NB, NT>>>(0.f, n_, d_x_, d_flag);
  
#ifndef USE_CUB_SCAN
  thrust::device_ptr<int> d_flag_ptr = thrust::device_pointer_cast(d_flag);
  thrust::device_ptr<int> d_scan_ptr = thrust::device_pointer_cast(d_scan);
  ////thrust::exclusive_scan(thrust::device, d_ptr, d_ptr+x_nnz+1, d_ptr);
  thrust::exclusive_scan(d_flag_ptr, d_flag_ptr + n_, d_scan_ptr);
#else
  void* d_temp_storage = NULL;
  size_t  temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_flag, d_scan, n_));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
               temp_storage_bytes));
  
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,   
                                        d_flag, d_scan, n_));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

  streamCompactKeyValKernel<<<NB, NT>>>(d_x_sparse_key_, 
        d_x_sparse_val_, d_scan, (TYPE_VT)0, d_x_, n_);   

  vec_type_  = 0; //sparse type.
  return err;
}

template <typename VT>
__global__ void dense2bitarray_ker(VT* d_x, int len, int*  d_bitvec) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //one thread deals with one element in a dense vec.
  for( ; idx < len; idx += gridDim.x * blockDim.x ) {
    VT  val = __ldg(d_x + idx);
    if(!val) {
      unsigned int neighbor_id = idx;
      int dword = (neighbor_id >> 5);
      int bit = neighbor_id  & 0x1F;
      atomicOr(&d_bitvec[dword], (1<<bit));
    }
  }
}

//TODO: to test!
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::dense2bitarray() {
  int err = SPMSPV_SUCCESS;
  // if( vec_type_ != 1 ) {
  //   std::cout << "input is not dense type!" << std::endl;
  //   exit(0);
  // }

  //std::cout << "Converting from dense to bitarray:\n";
  
  /*generate bit_vector. */
  int NT = 128;
  int NB = (n_ + NT - 1)/ NT;
  cudaErrCheck(cudaMemset(d_bit_vector_, 0, num_ints_ * sizeof(int)));
  dense2bitarray_ker<<<NB, NT>>>(d_x_, n_, d_bit_vector_);           

  vec_type_  = 2; //bitarray type.
  return err;
}
/********************************vector format conversion***************************/

/*******************************csr5-spmv and spmspv *****************************/
//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
void SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::setSigma(int sigma) {
  if (sigma == ANONYMOUSLIB_AUTO_TUNED_SIGMA) {
    int r = 4;
    int s = 32;
    int t = 256;
    int u = 6;
    
    int nnz_per_row = nnz_ / m_;
    if (nnz_per_row <= r)
        _csr5_sigma = r;
    else if (nnz_per_row > r && nnz_per_row <= s)
        _csr5_sigma = nnz_per_row;
    else if (nnz_per_row <= t && nnz_per_row > s)
        _csr5_sigma = s;
    else // nnz_per_row > t
        _csr5_sigma = u;
  }
  else {
    _csr5_sigma = sigma;
  }
}
//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeSigma()
{
  return _csr5_sigma;
}

//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CSR5Preprocess() {
  int err = SPMSPV_SUCCESS;

  if (format_ == SPMSPV_FORMAT_CSR5)
    return err;

  if (format_ == SPMSPV_FORMAT_CSR) {
    double malloc_time = 0, tile_ptr_time = 0, tile_desc_time = 0, transpose_time = 0;
    anonymouslib_timer malloc_timer, tile_ptr_timer, tile_desc_timer, transpose_timer;
    // compute sigma
    _csr5_sigma = computeSigma();
    std::cout << "omega = " << ANONYMOUSLIB_CSR5_OMEGA << ", sigma = " << _csr5_sigma << ". " << std::endl;

    // compute how many bits required for `y_offset' and `carry_offset'
    int base = 2;
    _bit_y_offset = 1;
    while (base < ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma) { base *= 2; _bit_y_offset++; }

    base = 2;
    _bit_scansum_offset = 1;
    while (base < ANONYMOUSLIB_CSR5_OMEGA) { base *= 2; _bit_scansum_offset++; }

    if (_bit_y_offset + _bit_scansum_offset > sizeof(TYPE_UIT) * 8 - 1) //the 1st bit of bit-flag should be in the first packet
        return ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA;

    int bit_all = _bit_y_offset + _bit_scansum_offset + _csr5_sigma;
    _num_packet = ceil((double)bit_all / (double)(sizeof(TYPE_UIT) * 8));

    // calculate the number of partitions
    _p = ceil((double)nnz_ / (double)(ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma));
    //cout << "#partition = " << _p << endl;

    malloc_timer.start();
    // malloc the newly added arrays for CSR5
    checkCudaErrors(cudaMalloc((void **)&_csr5_partition_pointer, (_p + 1) * sizeof(TYPE_UIT)));
    
    checkCudaErrors(cudaMalloc((void **)&_csr5_partition_descriptor, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(TYPE_UIT)));
    checkCudaErrors(cudaMemset(_csr5_partition_descriptor, 0, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(TYPE_UIT)));
    
    checkCudaErrors(cudaMalloc((void **)&_temp_calibrator, _p * sizeof(TYPE_VT)));
    checkCudaErrors(cudaMemset(_temp_calibrator, 0, _p * sizeof(TYPE_VT)));
    
    //TODO: csr5_partition_descriptor_offset_pointer
    checkCudaErrors(cudaMalloc((void **)&_csr5_partition_descriptor_offset_pointer, (_p + 1) * sizeof(TYPE_IT)));
    checkCudaErrors(cudaMemset(_csr5_partition_descriptor_offset_pointer, 0, (_p + 1) * sizeof(TYPE_IT)));
    malloc_time += malloc_timer.stop();
    
    cudaError_t err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("here-0 invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    // convert csr data to csr5 data (3 steps)
    // step 1. generate partition pointer
    tile_ptr_timer.start();
    err = generate_partition_pointer(_csr5_sigma, _p, m_, nnz_,
                                      _csr5_partition_pointer, d_csr_row_);
    if (err != SPMSPV_SUCCESS)
        return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
    cudaDeviceSynchronize();
    tile_ptr_time += tile_ptr_timer.stop();
    
    err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("here-1 invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    malloc_timer.start();
    //tail
    TYPE_UIT tail;
    checkCudaErrors(cudaMemcpy(&tail, &_csr5_partition_pointer[_p-1], sizeof(TYPE_UIT),   cudaMemcpyDeviceToHost));
    _tail_partition_start = (tail << 1) >> 1;
    //cout << "_tail_partition_start = " << _tail_partition_start << endl;
    malloc_time += malloc_timer.stop();
    
     err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("here-2 invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    // step 2. generate partition descriptor
    _num_offsets = 0;
    tile_desc_timer.start();
    err = generate_partition_descriptor(_csr5_sigma, _p, m_,
                                        _bit_y_offset, _bit_scansum_offset, _num_packet,
                                        d_csr_row_, _csr5_partition_pointer, _csr5_partition_descriptor,
                                        _csr5_partition_descriptor_offset_pointer, &_num_offsets);
    if (err != SPMSPV_SUCCESS)
      return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
    cudaDeviceSynchronize();
    tile_desc_time += tile_desc_timer.stop(); // fixed a bug here (April 2016)
    
     err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("here-3 invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    if (_num_offsets) {
      //cout << "has empty rows, _num_offsets = " << _num_offsets << endl;
      malloc_timer.start();
      checkCudaErrors(cudaMalloc((void **)&_csr5_partition_descriptor_offset, _num_offsets * sizeof(TYPE_IT)));
      malloc_time += malloc_timer.stop();

      tile_desc_timer.start();
      err = generate_partition_descriptor_offset(_csr5_sigma, _p,
                                          _bit_y_offset, _bit_scansum_offset, _num_packet,
                                          d_csr_row_, _csr5_partition_pointer, _csr5_partition_descriptor,
                                          _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset);
      if (err != SPMSPV_SUCCESS)
          return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
      cudaDeviceSynchronize();
      tile_desc_time += tile_desc_timer.stop();
    }

     err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("here-4 invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    // step 3. transpose column_index and value arrays
    transpose_timer.start();
    err = aosoa_transpose(_csr5_sigma, nnz_,
                          _csr5_partition_pointer, d_csr_col_, d_csr_val_, true);
    if (err != SPMSPV_SUCCESS)
        return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
    cudaDeviceSynchronize();
    transpose_time += transpose_timer.stop();
    
     err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("here-5 invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

#ifdef JUDGE_OPTJUDGE_OPT
    count_fast_track_num<<<1, 1>>>(_csr5_partition_pointer, _p,  _tail_partition_start, _m);
#endif
    // std::cout << "CSR->CSR5 malloc time = " << malloc_time << " ms." << std::endl;
    // std::cout << "CSR->CSR5 tile_ptr time = " << tile_ptr_time << " ms." << std::endl;
    // std::cout << "CSR->CSR5 tile_desc time = " << tile_desc_time << " ms." << std::endl;
    // std::cout << "CSR->CSR5 transpose time = " << transpose_time << " ms." << std::endl;

    format_ = SPMSPV_FORMAT_CSR5;
  }
return err;
}

//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::csr5spmv(const TYPE_VT  alpha,
                                                      TYPE_VT* y)
{
  int err = SPMSPV_SUCCESS;
  // if (format_ == SPMSPV_FORMAT_CSR) {
  //   return SPMSPV_UNSUPPORTED_CSR_SPMV;
  // }
  //if (format_ == SPMSPV_FORMAT_CSR5) {
    csr5_spmv(_csr5_sigma, _p, m_,
              _bit_y_offset, _bit_scansum_offset, _num_packet,
              d_csr_row_, d_csr_col_, d_csr_val_,
              _csr5_partition_pointer, _csr5_partition_descriptor,
              _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset,
              _temp_calibrator, _tail_partition_start,
              alpha, d_x_, _x_tex, /*beta,*/ y);
  //}
return err;
}

//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::csr5spmspv(const TYPE_VT alpha,
                                 TYPE_VT* y) {
  int err = SPMSPV_SUCCESS;
  // if (format_ == SPMSPV_FORMAT_CSR) {
  //   return SPMSPV_UNSUPPORTED_CSR_SPMV;
  // }
  //if (format_ == SPMSPV_FORMAT_CSR5) {
    csr5_spmspv(_csr5_sigma, _p, m_,
              _bit_y_offset, _bit_scansum_offset, _num_packet,
              d_csr_row_, d_csr_col_, d_csr_val_,
              _csr5_partition_pointer, _csr5_partition_descriptor,
              _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset,
              _temp_calibrator, _tail_partition_start,
              alpha, d_x_, _x_tex, 
              d_bit_vector_,/*Added by*/ 
              /*beta,*/ y);
  //}
return err;
}

//anonymouslibHandle
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CSR5Postprocess()
{
  int err = SPMSPV_SUCCESS;
  if (format_ == SPMSPV_FORMAT_CSR)
    return err;
  if (format_ == SPMSPV_FORMAT_CSR5) {
    // convert csr5 data to csr data
    err = aosoa_transpose(_csr5_sigma, nnz_,
                          _csr5_partition_pointer, d_csr_col_, d_csr_val_, false);

    // free the two newly added CSR5 arrays
    checkCudaErrors(cudaFree(_csr5_partition_pointer));
    checkCudaErrors(cudaFree(_csr5_partition_descriptor));
    checkCudaErrors(cudaFree(_temp_calibrator));
    checkCudaErrors(cudaFree(_csr5_partition_descriptor_offset_pointer));
    if (_num_offsets) checkCudaErrors(cudaFree(_csr5_partition_descriptor_offset));

    format_ = SPMSPV_FORMAT_CSR;
  }
return err;
}
/*******************************csr5-spmv and spmspv *****************************/

/*******************************hola-spmv and spmspv ****************************/
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::holaPreprocess() {
  int err = SPMSPV_SUCCESS;
  if (format_ == SPMSPV_FORMAT_CSR5 || format_ == SPMSPV_FORMAT_CSC) {
    return FORMAT_ERR_HOLA_SPMV;
  }
  if (format_ == SPMSPV_FORMAT_CSR) {
    hola_pre(holatemp_req_, m_, n_, nnz_, n_, d_x_/*, HolaMode::Default*/);
    //std::cout << "DEBUG: holapreprocess: holatemp_req size is " << holatemp_req_ << std::endl;
    cudaErrCheck(cudaMalloc((void **)&dholatempmem_, holatemp_req_));
  }
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::holaPostprocess() {
  int err = SPMSPV_SUCCESS;
  if(dholatempmem_) {
    cudaErrCheck(cudaFree(dholatempmem_));
  }
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::naivespmv(const TYPE_VT  alpha,
                                                       TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
  //naive_spmv(dDenseVector<T>& res, const dCSR<T>& m, const dDenseVector<T>& v, bool transpose)
  naive_spmv(m_, n_, nnz_, d_csr_row_, d_csr_col_, d_csr_val_, d_x_, d_y);
  return err;
}


template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::holaspmv(const TYPE_VT  alpha,
                                                       TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
  // if (format_ == SPMSPV_FORMAT_CSR5 || format_ == SPMSPV_FORMAT_CSC) {
  //   return FORMAT_ERR_HOLA_SPMV;
  // }
  // if (format_ == SPMSPV_FORMAT_CSR) {
    hola_spmv(dholatempmem_, holatemp_req_, m_, n_, nnz_, d_csr_row_, d_csr_col_, d_csr_val_, 
          n_, d_x_, d_y, /*HolaMode::Default,*/ false);
  //}
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::naivespmspv(const TYPE_VT  alpha,
                                                       TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
  naive_spmspv(m_, n_, nnz_, d_csr_row_, d_csr_col_, d_csr_val_, d_x_, d_y, d_bit_vector_);
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::holaspmspv(const TYPE_VT  alpha,
                                                       TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
  // if (format_ == SPMSPV_FORMAT_CSR5 || format_ == SPMSPV_FORMAT_CSC) {
  //   return FORMAT_ERR_HOLA_SPMV;
  // }
  // if (format_ == SPMSPV_FORMAT_CSR) {
    hola_spmspv(dholatempmem_, holatemp_req_, m_, n_, nnz_, d_csr_row_, d_csr_col_, d_csr_val_, 
        n_, d_x_, d_y, d_bit_vector_, /*HolaMode::Default,*/ false);
  //}
  return err;
}
/*******************************hola-spmv and spmspv *****************************/

/*******************************column-based spmspv *****************************/
//Usage: need to know n and nnz before.
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::allocPreBuffer() {
  int err = SPMSPV_SUCCESS;

  //TODO: 这里开辟内存的时候应该是多少呢: 最大值？
  //std::cout <<"debug: alloclPreBuffer, nnz_ = " << nnz_ << std::endl;
  
//   cudaErrCheck(cudaMalloc((void **)&d_ptr_col_len_,  
//                           (n_ + 1) * sizeof(int)));
//   cudaErrCheck(cudaMemset(d_ptr_col_len_, 0, (n_ + 1) * sizeof(int)));
// #ifdef CUDA_BUCKET_METHOD
//   cudaErrCheck(cudaMalloc((void **)&d_ptr_pre_alloc_buffer_, 
//               (((MAX_BUCKETS*x_nnz_) + 1 + x_nnz_) * sizeof(TYPE_IT) 
//                + 65536 + (2 * nnz_ + 1) * sizeof(TYPE_VT))));
// #else
//   //no sort version.
//   //cudaErrCheck(cudaMalloc((void **)&d_ptr_pre_alloc_buffer_, (nnz_ + n_ + 1) * sizeof(int)));
//   //for sort version.
//   cudaErrCheck(cudaMalloc((void **)&d_ptr_pre_alloc_buffer_, (3 * nnz_ + n_ + 1) * sizeof(int)));
// #endif
  // h_odata_ = (float*) malloc(REDUCE_BLOCKS * sizeof(float));
  // d_odata_ = NULL;
  // cudaErrCheck(cudaMalloc((void **) &d_odata_, REDUCE_BLOCKS * sizeof(float)));
  
  //sort version.
  cudaErrCheck(cudaMalloc((void **)&d_buffer_, (2 * n_ + 2 + 3 * nnz_ + REDUCE_BLOCKS) * sizeof(float)));
  //no sort version.
  //cudaErrCheck(cudaMalloc((void **)&d_buffer_, (2 * n_ + 2 + nnz_ + REDUCE_BLOCKS) * sizeof(float)));
  d_ptr_col_len_ = (int*)d_buffer_;//(n+1)
  d_ptr_pre_alloc_buffer_ = (int*)(d_buffer_ + n_ + 1);//(n+1+nnz)
  //no sort
  //d_odata_ = (float*)(d_buffer_ + 2 * n_ + 2 + nnz_);//NOTE!!keep consistent with line 1744
  //sort
  d_odata_ = (float*)(d_buffer_ + 2 * n_ + 2 + 3*nnz_);//NOTE!!keep consistent with line 1744

  h_odata_ = (float*) malloc(REDUCE_BLOCKS * sizeof(float));

return err;
}

//format: csr, dense vec.
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::set_spmv_type(
               int code) {
  int err = SPMSPV_SUCCESS;
  spmv_type_ = code;
}

// //format: csr, dense vec.
// template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
// int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::spmvPreprocess() {
//   int err = SPMSPV_SUCCESS;
//   if (vec_type_ != 1) {
//     std::cout << "The vec format is dense yet." <<std::endl;
//     return SPMSPV_VEC_FORMAT_ERR;
//   }
//   if (spmv_type_ == 0) {// hola-spmv
//     holaPreprocess();
//   }else{//csr5-spmv.
//     CSR5Preprocess();
//   }
// }

//format: csr, dense vec.
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::spmv(
               const TYPE_VT  alpha,TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
  if (spmv_type_ == 0) {// hola-spmv
    holaspmv(alpha, d_y);
  }else{//csr5-spmv.
    csr5spmv(alpha, d_y);
  }
}

// //format: csr, dense vec.
// template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
// int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::spmvPostprocess() {
//   int err = SPMSPV_SUCCESS;
//   if (spmv_type_ == 0) {// hola-spmv
//      holaPostprocess();
//   }else{//csr5-spmv.
//      CSR5Postprocess();
//   }
// }

//format: csr, bitarray vec.
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::spmspv(
                   const TYPE_VT  alpha,TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
  if (spmv_type_ == 0) {// hola-spmspv
    holaspmspv(alpha, d_y);
  }else{//csr5-spmspv.
    csr5spmspv(alpha, d_y);
  }
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedMyNoSortSpmspv(
               bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, TYPE_VT* d_y,
               TYPE_IT* d_y_key, TYPE_VT*d_y_val) {
  int err = SPMSPV_SUCCESS;
 
  //int threads = THREADS_PER_BLOCK;
  int threads = 1024;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  //cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedMySpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_y_key, d_y_val,//new added,
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads, IsBinLenComputed,
                                false);  
  return err;
}


template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedSortMySpmspv(
               bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, TYPE_VT* d_y,
               TYPE_IT* d_y_key, TYPE_VT*d_y_val) {
  int err = SPMSPV_SUCCESS;
 
  //int threads = THREADS_PER_BLOCK;
  int threads = 1024;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  //cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedMySpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_y_key, d_y_val,//new added,
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads, IsBinLenComputed,
                                true);  
  return err;
}

//used for csc-spmspv.
//TODO: ynnz is not set value!
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int dense2sparse_inner(TYPE_IT m, TYPE_VT* d_dense, int* d_pre_buffer, 
                      TYPE_IT* ynnz, TYPE_IT* d_key, TYPE_VT* d_val) {
                                           
  int err = SPMSPV_SUCCESS;

#if 1   
  int NT = 256;
  int NB = (m + NT - 1)/NT;

  int* d_flag = (int*)d_pre_buffer;
	int* d_scan = (int*)d_pre_buffer + m;
  // int* d_flag = NULL;
  // int* d_scan = NULL;
  // cudaErrCheck(cudaMalloc((void **) &d_flag, (m)*sizeof(int)));
  // cudaErrCheck(cudaMalloc((void **) &d_scan, (m+1)*sizeof(int)));

	updateFlagKernel<<<NB, NT>>>(0.f, m, d_dense, d_flag);

#if 1
#ifndef USE_CUB_SCAN
  //thrust scan!
  thrust::device_ptr<int> d_flag_ptr = thrust::device_pointer_cast(d_flag);
  thrust::device_ptr<int> d_scan_ptr = thrust::device_pointer_cast(d_scan);
  ////thrust::exclusive_scan(thrust::device, d_ptr, d_ptr+x_nnz+1, d_ptr);
  thrust::exclusive_scan(d_flag_ptr, d_flag_ptr + m, d_scan_ptr);
#else
  void* d_temp_storage = NULL;
  size_t  temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_flag, d_scan, m));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
               temp_storage_bytes));
  
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,   
                                        d_flag, d_scan, m));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif
#endif
  streamCompactKeyValKernel<<<NB, NT>>>(d_key, d_val, d_scan, (TYPE_VT)0, d_dense, m);

#endif

  // cudaErrCheck(cudaFree(d_flag));
  // cudaErrCheck(cudaFree(d_scan));
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::sparse2dense_outer(
                      TYPE_IT ynnz, TYPE_IT* d_key, TYPE_VT* d_val,
                      TYPE_IT m, TYPE_VT* d_dense) {
                                           
  int err = SPMSPV_SUCCESS;

  const int nt = 256;
  const int nvals = ynnz;
  printf("in sparse2dense_outer: ynnz = %d\n", ynnz);

  int NT = nt;
  int NB = (nvals + nt -1)/nt;
  
  cudaErrCheck(cudaMemset(d_dense, 0, m * sizeof(TYPE_VT)));

  scatter<<<NB, NT>>>(nvals, d_key, d_val, d_dense);

  return err;
}


template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::dense2sparse_outer(TYPE_IT m, TYPE_VT* d_dense, 
                                            TYPE_IT* ynnz, TYPE_IT* d_key, TYPE_VT* d_val) {
                                           
  int err = SPMSPV_SUCCESS;

#if 1   
  int NT = 256;
  int NB = (m + NT - 1)/NT;

  // int* d_flag = (int*)d_ptr_pre_alloc_buffer_;
	// int* d_scan = (int*)d_ptr_pre_alloc_buffer_ + m;
  int* d_flag = NULL;
  int* d_scan = NULL;
  cudaErrCheck(cudaMalloc((void **) &d_flag, (m)*sizeof(int)));
  cudaErrCheck(cudaMalloc((void **) &d_scan, (m+1)*sizeof(int)));

	updateFlagKernel<<<NB, NT>>>(0.f, m, d_dense, d_flag);

#if 1
#ifndef USE_CUB_SCAN
  //thrust scan!
  thrust::device_ptr<int> d_flag_ptr = thrust::device_pointer_cast(d_flag);
  thrust::device_ptr<int> d_scan_ptr = thrust::device_pointer_cast(d_scan);
  ////thrust::exclusive_scan(thrust::device, d_ptr, d_ptr+x_nnz+1, d_ptr);
  thrust::exclusive_scan(d_flag_ptr, d_flag_ptr + m, d_scan_ptr);
#else
  void* d_temp_storage = NULL;
  size_t  temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_flag, d_scan, m));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
               temp_storage_bytes));
  
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,   
                                        d_flag, d_scan, m));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif
#endif
  streamCompactKeyValKernel<<<NB, NT>>>(d_key, d_val, d_scan, (TYPE_VT)0, d_dense, m);

#endif

  cudaErrCheck(cudaFree(d_flag));
  cudaErrCheck(cudaFree(d_scan));
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedMyNoSortSpmspv_keyval(
               bool IsBinLenComputed, const TYPE_VT  alpha, 
               TYPE_IT* ynnz, TYPE_VT* d_y, 
               TYPE_IT* d_y_key, TYPE_VT*d_y_val) {
  int err = SPMSPV_SUCCESS;
 
  int threads = THREADS_PER_BLOCK;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedMySpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_y_key, d_y_val,//new added 
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads, IsBinLenComputed,
                                false);
  //the resultant vector: should call dense2sparse.                           
  dense2sparse_inner<TYPE_IT, TYPE_UIT, TYPE_VT>(m_, d_y, (int*)d_ptr_pre_alloc_buffer_, 
                    ynnz, d_y_key, d_y_val);
  return err;
}


template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedNoSortMergeSpmspv(
               bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, TYPE_VT* d_y,
               TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val) {
  int err = SPMSPV_SUCCESS;
 
  int threads = THREADS_PER_BLOCK;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedMergeSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_y_sparse_key, d_y_sparse_val,
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads, IsBinLenComputed,
                                false);  
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedNoSortMergeSpmspv_keyval(
               bool IsBinLenComputed, const TYPE_VT  alpha, 
               TYPE_IT* ynnz, TYPE_VT* d_y, 
               TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val) {
  int err = SPMSPV_SUCCESS;
 
  int threads = THREADS_PER_BLOCK;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedMergeSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_y_sparse_key, d_y_sparse_val,
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads, IsBinLenComputed,
                                false);
  //the resultant vector: should call dense2sparse. 将结果vector以sparse格式存储下来                               
  dense2sparse_inner<TYPE_IT, TYPE_UIT, TYPE_VT>(m_, d_y, (int*)d_ptr_pre_alloc_buffer_, 
                    ynnz, d_y_sparse_key, d_y_sparse_val);
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedSortMergeSpmspv(
               bool IsBinLenComputed, const TYPE_VT  alpha, TYPE_IT* ynnz, TYPE_VT* d_y,
               TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val) {
  int err = SPMSPV_SUCCESS;
 
  int threads = THREADS_PER_BLOCK;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedMergeSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_y_sparse_key, d_y_sparse_val,
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads, IsBinLenComputed,
                                true);  
  return err;
}


template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedNoSortNaiveSpmspv(
               const TYPE_VT  alpha, TYPE_IT* ynnz, TYPE_VT* d_y) {
  int err = SPMSPV_SUCCESS;
 
  int threads = THREADS_PER_BLOCK;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  // std::cout << "debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //       << threads << ", num_blocks = " << num_blocks << std::endl;
  cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedNoSortNaiveSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads);
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedNoSortNaiveSpmspv_keyval(
               const TYPE_VT  alpha, 
               TYPE_IT* ynnz, TYPE_VT* d_y, 
               TYPE_IT* d_y_sparse_key, TYPE_VT*d_y_sparse_val) {
  int err = SPMSPV_SUCCESS;
 
  int threads = THREADS_PER_BLOCK;
  int num_blocks = ceil((x_nnz_)/(float)(threads));
  //std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
  //    << threads << ", num_blocks = " <<num_blocks <<std::endl;
  cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
  CscBasedNoSortNaiveSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                              (d_csc_row_, d_csc_col_, d_csc_val_, 
                                m_, n_, nnz_, 
                                bin_len_,
                                x_nnz_,
                                d_x_sparse_key_, d_x_sparse_val_,
                                alpha, 
                                ynnz, d_y, 
                                d_ptr_col_len_,
                                d_ptr_pre_alloc_buffer_,
                                num_blocks, threads);
  //the resultant vector: should call dense2sparse.                     
  dense2sparse_inner<TYPE_IT, TYPE_UIT, TYPE_VT>(m_, d_y, (int*)d_ptr_pre_alloc_buffer_, 
                    ynnz, d_y_sparse_key, d_y_sparse_val);
  return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedSortNaiveSpmspv( 
                              const TYPE_VT  alpha,
                              TYPE_IT* ynnz, TYPE_IT* d_y_inx, TYPE_VT* d_y_val,
                              int method) {
  int err = SPMSPV_SUCCESS;
  // if (format_ == SPMSPV_FORMAT_CSR) {
  //     std::cout << "Can not support csr spmspv yet" << std::endl;
  //     return SPMSPV_UNSUPPORTED_CSR_SPMSPV;
  // }
  // if (format_ == SPMSPV_FORMAT_CSC) {
    int threads = THREADS_PER_BLOCK;
    int num_blocks = ceil((x_nnz_)/(double)(threads));
   
    // std::cout<<"debug: x_nnz_ = " << x_nnz_ << ", threads = " 
    //     << threads << ", num_blocks = " <<num_blocks <<std::endl;

    CscBasedSortNaiveSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
                            (d_csc_row_, d_csc_col_, d_csc_val_, 
                             m_, n_, nnz_, 
                             x_nnz_, 
                             d_x_sparse_key_,
                             d_x_sparse_val_, 
                             alpha, 
                             ynnz, d_y_inx, d_y_val, 
                             d_ptr_col_len_,
                             d_ptr_pre_alloc_buffer_,
                             num_blocks, threads);
  //}
  return err;
}


// template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
// int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CscBasedBucketSpmspv(
//                                             const TYPE_VT  alpha,
//                                             TYPE_IT* ynnz, TYPE_VT* d_y,
//                                             int buckets) {
//   int err = SPMSPV_SUCCESS;
//   if (format_ == SPMSPV_FORMAT_CSR) {
//       std::cout << "Can not support csr spmspv yet" << std::endl;
//       return SPMSPV_UNSUPPORTED_CSR_SPMSPV;
//   }

//   if (format_ == SPMSPV_FORMAT_CSC)
//   {
//       int threads = THREADS_PER_BLOCK;
//       int num_blocks = ceil((x_nnz_)/(double)(threads));
     
//       // std::cout<<"debug: xnnz = " << _xnnz << ", threads = " 
//       //     << threads << ", num_blocks = " <<num_blocks <<std::endl;

// #ifdef BUCKET_LOAD_BALANCE 
//       CscBasedBucketLoadBalanceSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
//                                       (d_csc_row_, d_csc_col_, d_csc_val_, 
//                                        m_, n_, nnz_, 
//                                        x_nnz_, /*d_x_sparse_,*/
//                                        alpha, 
//                                        ynnz, d_y, 
//                                        d_ptr_col_len_,
//                                        d_ptr_pre_alloc_buffer_,
//                                        buckets,
//                                        num_blocks, threads);
// #else
//       CscBasedBucketSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>
//                                       (d_csc_row_, d_csc_col_, d_csc_val_, 
//                                        m_, n_, nnz_, 
//                                        x_nnz_, /*d_x_sparse_,*/ 
//                                        alpha, d_x_sparse_key_,
//                                        d_x_sparse_val_, 
//                                        ynnz, d_y, 
//                                        d_ptr_col_len_,
//                                        d_ptr_pre_alloc_buffer_,
//                                        buckets,
//                                        num_blocks, threads);
// #endif
//     }
//   return err;
// }

/*******************************column-based spmspv *****************************/


/*******************************kernel seclection *****************************/
// template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
// long long SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeBinlenSerial() {

// }
template<typename iT, typename uiT, typename vT>
__global__
void countNonzerosPerCol(iT* d_csc_col,  
                        iT x_nnz, 
                        iT* d_sparse_x_key,
                        int* d_col_len_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < x_nnz) {
    iT col_index = __ldg(d_sparse_x_key + tid);
    iT start = __ldg(d_csc_col + col_index);
    iT end = __ldg(d_csc_col + col_index + 1);
    d_col_len_ptr[tid] = end - start;
  }
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
void SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeMatFeture_serial(int m, int n, int mat_nnz, int* csr_row, 
      int* max_elems, int* min_elems, float* avg_elems, 
      float* x_range, float* standard_row, float* equlity, float* gini){
  int max = 0;
  int min = std::numeric_limits<int>::max();
  int positive_min = std::numeric_limits<int>::max();
  int count = 0;
  int num = 0; 
  float sum = 0.0;
  float gini_coefficiency = 0.0;
  float edge_equlity = 0.0;//
  double sum_edge_equlity = 0.0;
  int i = 0;
  int j = 0;
  float sum_gini_1 = 0.0;
  float sum_gini_2 = 0.0;
  std::vector<int> degrees;
  double avg_nnz_per_row = mat_nnz/(float)m; 
  
  for (i = 0; i < m_; i++) {
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
  for (std::vector<int>::iterator it = degrees.begin() ; (it+1) != degrees.end(); ++it){ 
    sum_gini_1 += 1.0*j*(*(it+1)); 
    sum_gini_2 += 1.0*(*it);
    j++;
  }
  gini_coefficiency = (2.0*sum_gini_1)/(m*sum_gini_2) - (m+1)/(float)m; 
  if(gini_coefficiency < 0 ) gini_coefficiency *= -1; 
  
  *max_elems = max;
  *min_elems = min;
  *avg_elems = avg_nnz_per_row;
  *x_range = (max-min)/(1.0*n);//relative range of degree
  
  *standard_row = standard_deviation_nnz_row;
  *equlity = edge_equlity;
  *gini = gini_coefficiency;
}

  
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
void SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeVecFeture_serial(int x_nnz, int* x_sparse_key, int* bin_len, int* max_elems, int* min_elems){
  
  int* csc_col = (int* )malloc((n_+1)*sizeof(int));
  CHECK_MALLOC(csc_col);

  cudaErrCheck(cudaMemcpy(csc_col, d_csc_col_, (n_+1)*sizeof(int), cudaMemcpyDeviceToHost));
  

  int min = std::numeric_limits<int>::max();
  int max = 0;
  int sum = 0;
  for(int i=0; i<x_nnz; i++){
    int col_id = x_sparse_key[i];
    int num = csc_col[col_id + 1] - csc_col[col_id];
    sum += num;
    if(num > max) max = num;
    if(num < min) min = num;
  }
  *max_elems = max;
  *min_elems = min;
  *bin_len = sum;
  
  free(csc_col);
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
long long SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeBinlen() {
  //int err = SPMSPV_SUCCESS;

  int NT = 256;
  int NB = (x_nnz_ + NT - 1)/NT;

  countNonzerosPerCol<TYPE_IT, TYPE_UIT,  TYPE_VT><<<NB, NT>>>(d_csc_col_, x_nnz_, d_x_sparse_key_, d_ptr_col_len_);
  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("CountNonzerosPerCol() invocate error.\n");
    printf("xnnz=%d, nb=%d, bt=%d.\n", x_nnz_, NB, NT);
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }

  // call cub.
  // http://nvlabs.github.io/cub/structcub_1_1_device_reduce.html#ab7f21e8255eb842aaf74305975ae607f
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  long long int* d_out = NULL;
  cudaMalloc(&d_out, 1 * sizeof(long long));
  //int* d_out = (int*)d_ptr_pre_alloc_buffer_;

  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_ptr_col_len_, d_out, x_nnz_);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //d_temp_storage = (int*)d_ptr_pre_alloc_buffer_ + 4;
  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_ptr_col_len_, d_out, x_nnz_);
  cudaErrCheck(cudaMemcpy(&bin_len_, d_out, sizeof(long long),   
                           cudaMemcpyDeviceToHost));

  //std::cout << "computeBinlen: bin_len_ = " << bin_len_ << std::endl;

  cudaFree(d_temp_storage);
  cudaFree(d_out);
return bin_len_;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduceBinlen(T* d_sparse_x_key, T* d_csc_col, /*T *g_idata,*/ 
             T* d_col_len_ptr, T *g_odata,    
            unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();//sum

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        //mySum += g_idata[i];
        int col_index = __ldg(d_sparse_x_key + i);
        T start = __ldg(d_csc_col + col_index);
        T end = __ldg(d_csc_col + col_index + 1);
        d_col_len_ptr[i] = end - start;
        mySum += end - start;

#ifdef DEBUG
        if(threadIdx.x == 0/* && blockIdx.x == 0*/)
          printf("%d.\n", mySum);
#endif

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n){
          //mySum += g_idata[i+blockSize];
          int col_index = __ldg(d_sparse_x_key + i+blockSize);
          T start = __ldg(d_csc_col + col_index);
          T end = __ldg(d_csc_col + col_index + 1);
          d_col_len_ptr[i+blockSize] = end - start;
          mySum += end - start;
#ifdef DEBUG
          if(threadIdx.x == 0/* && blockIdx.x == 0*/)
            printf("-2: %d.\n", mySum);
#endif            
        }
        i += gridSize;
    }

#ifdef DEBUG
    if(threadIdx.x == 0/* && blockIdx.x == 0*/)
      printf("-outer: %d.\n", mySum);
#endif

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;

#ifdef DEBUG
    //TODO: wrong!!
    if(threadIdx.x == 0)
      printf("sdata[0] = %d\n", sdata[tid]);
#endif

    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
      sdata[tid] = mySum = mySum + sdata[tid + 256];          
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        //TODO: 
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += active.shfl_down(mySum, offset);
            //mySum += active.shfl_down_sync(0xffffffff, mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
      sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
      sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    { 
      sdata[tid] = mySum = mySum + sdata[tid + 8];
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
      sdata[tid] = mySum = mySum + sdata[tid + 4];
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
      sdata[tid] = mySum = mySum + sdata[tid + 2];
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
      sdata[tid] = mySum = mySum + sdata[tid + 1];
    }

    cg::sync(cta);
#endif

    // write result for this block to global mem
    if (tid == 0){
      g_odata[blockIdx.x] = mySum;
    }
}

template <class T>
void
reduceBinlenWrapper(int size, int threads, int blocks, 
       T* d_sparse_x_key, T* d_csc_col,/*, T *d_idata,*/ 
       T* d_col_len_ptr, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduceBinlen<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 256:
                reduceBinlen<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 128:
                reduceBinlen<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 64:
                reduceBinlen<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 32:
                reduceBinlen<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 16:
                reduceBinlen<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  8:
                reduceBinlen<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  4:
                reduceBinlen<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  2:
                reduceBinlen<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  1:
                reduceBinlen<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceBinlen<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 256:
                reduceBinlen<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 128:
                reduceBinlen<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 64:
                reduceBinlen<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 32:
                reduceBinlen<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case 16:
                reduceBinlen<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  8:
                reduceBinlen<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  4:
                reduceBinlen<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  2:
                reduceBinlen<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;

            case  1:
                reduceBinlen<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_sparse_x_key, d_csc_col, /*d_idata,*/ d_col_len_ptr, d_odata, size);
                break;
        }
    }
}

template <class T>
T computeBinlenInter(int  n,
                  int  numThreads,
                  int  numBlocks,
                  T *h_odata,
                  T* d_sparse_x_key, 
                  T* d_csc_col,
                  T* d_col_len_ptr,
                  T *d_odata)
{
    T gpu_sum = 0;

    // execute the kernel
    reduceBinlenWrapper<T>(n, numThreads, numBlocks, d_sparse_x_key, d_csc_col, d_col_len_ptr, d_odata);

    // check if kernel execution generated an error
    //getLastCudaError("Kernel execution failed");
    cudaError_t err_r = cudaGetLastError();
    if ( cudaSuccess != err_r) {
      printf("reduceMatFeature() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }

    // sum partial sums from each block on CPU
    // copy result from device to host
    cudaErrCheck(cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(T), cudaMemcpyDeviceToHost));

    for (int i=0; i<numBlocks; i++)
    {
        //std::cout << h_odata[i] << std::endl;
        gpu_sum  += h_odata[i];
    }
    //printf("gpu_sum = %lf\n", gpu_sum);

    return gpu_sum;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
long long SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeBinlenVer2() {
  //TODO:
  int threads = 64;
  int blocks = REDUCE_BLOCKS;

#if 1
  //SpmspvTimer timer;
  //timer.Start();
  int* h_odata = (int*)h_odata_;
  int* d_odata = (int*)d_odata_;
  bin_len_ = computeBinlenInter<int>(x_nnz_, threads, blocks, h_odata, 
            d_x_sparse_key_, d_csc_col_, d_ptr_col_len_,
            d_odata);
  //double time = timer.Stop();
  //printf("computebinlenInter time is %lf\n", time);
#endif   

return bin_len_;
}


//前提：x_nnz和binlen已经计算完成.
// GM1: col-spmspv 
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
double SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeGM1() {
  //int err = SPMSPV_SUCCESS;
  GM1_ = x_nnz_ * sizeof(int) + bin_len_ * sizeof(TYPE_VT) 
       + bin_len_ * sizeof(int) 
       + x_nnz_ * sizeof(int) + x_nnz_ * sizeof(TYPE_VT)
       + bin_len_ * sizeof(TYPE_VT);
return GM1_;
}
// GM2: row-spmspv
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
double SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeGM2() {
  //int err = SPMSPV_SUCCESS;
  GM2_ = (m_ + 1) * sizeof(int) + bin_len_ * sizeof(TYPE_VT) 
       + nnz_ * sizeof(int)
       + bin_len_ * sizeof(TYPE_VT) + m_ * sizeof(TYPE_VT);
return GM2_;       
}

// GM3:  spmv
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
double SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::computeGM3() {
  //int err = SPMSPV_SUCCESS;
  GM3_ = (m_ + 1) * sizeof(int) + nnz_ * sizeof(TYPE_VT) 
       + nnz_ * sizeof(int)
       + nnz_ * sizeof(TYPE_VT) + m_ * sizeof(TYPE_VT);
return GM3_;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
long long SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::getBinlen() {
  //int err = SPMSPV_SUCCESS;
  return bin_len_;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
double SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::getGM1() {
  //int err = SPMSPV_SUCCESS;
  return GM1_;
}
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
double SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::getGM2() {
  //int err = SPMSPV_SUCCESS;
  return GM2_;
}
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
double SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::getGM3() {
  //int err = SPMSPV_SUCCESS;
  return GM3_;
}
/*******************************kernel seclection *****************************/



#if 0 
//matrix-driven smsv
template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::CsrBasedSpmspv(
                                                const TYPE_VT alpha,
                                                TYPE_VT* y) {
  int err = SPMSPV_SUCCESS;
  if (format_ == SPMSPV_FORMAT_CSR)
  {
    // std::cout << "Can not support csr spmspv yet" <<std::endl;
    // return SPMSPV_UNSUPPORTED_CSR_SPMSPV;//TODO
    int threads = THREADS_PER_BLOCK;
    int num_blocks = ceil((m_)/(double)(threads));//每个线程处理一行。
   
    std::cout<<"debug: m_ = " << m_ << ", threads = " 
        << threads << ", num_blocks = " <<num_blocks <<std::endl;

    TYPE_VT* d_y = NULL; 
    cudaErrCheck(cudaMalloc((void **)&d_y,  m_ * sizeof(TYPE_VT)));
    cudaErrCheck(cudaMemset(d_y, 0,  m_ * sizeof(TYPE_VT)));
#ifdef USE_BIT_VECTOR
     CsrBasedSpmspvBitvectorDriver<TYPE_IT, TYPE_UIT, TYPE_VT>(d_csr_row_, 
                                        d_csr_col_, d_csr_val_,
                                        m_, n_, nnz_, 
                                        x_nnz_, d_x_sparse_key_, 
                                        d_x_,
                                        alpha, 
                                        d_y,
                                        num_blocks, threads);
#else
    CsrBasedSpmspvDriver<TYPE_IT, TYPE_UIT, TYPE_VT>(d_csr_row_, 
                                        d_csr_col_, d_csr_val_,
                                        m_, n_, nnz_, 
                                        x_nnz_, d_x_sparse_key_, 
                                        d_x_sparse_val_,
                                        alpha, 
                                        d_y,
                                        num_blocks, threads);
#endif
    cudaErrCheck(cudaMemcpy(y, d_y, m_ * sizeof(TYPE_VT), 
                 cudaMemcpyDeviceToHost));                                
    if (d_y)
        cudaErrCheck(cudaFree(d_y));
  }
  else {
      std::cout << "Please check the format (shoule be csr)" << std::endl;
      return err;//这里需要改变一下错误码
  }
  return err;
}
#endif

// template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
// int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::InputCoo(TYPE_IT  nnz,
//                                                       TYPE_IT *coo_row,
//                                                       TYPE_IT *coo_col,
//                                                       TYPE_VT *coo_val) {
//   format_ = SPMSPV_FORMAT_COO;
//   nnz_ = nnz;

//   coo_row_  = coo_row;
//   coo_col_  = coo_col;
//   coo_val_  = coo_val;
//   return SPMSPV_SUCCESS;
// }



template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::deallocPreBuffer() {
  int err = SPMSPV_SUCCESS;
  
  // if (d_ptr_col_len_)
  //   cudaErrCheck(cudaFree(d_ptr_col_len_));

  // if (d_ptr_pre_alloc_buffer_)
  //   cudaErrCheck(cudaFree(d_ptr_pre_alloc_buffer_));  
  
  if(d_buffer_) cudaErrCheck(cudaFree(d_buffer_));
  if(h_odata_) free(h_odata_);
  // if(d_odata_) cudaErrCheck(cudaFree(d_odata_));

return err;
}

template <class TYPE_IT, class TYPE_UIT, class TYPE_VT>
int SpmspvHandle<TYPE_IT, TYPE_UIT, TYPE_VT>::Destroy() {
  
  cudaDestroyTextureObject(_x_tex);

  //csc-spmspv mem.
  if (d_csc_col_)
    cudaErrCheck(cudaFree(d_csc_col_));
  if (d_csc_row_)
    cudaErrCheck(cudaFree(d_csc_row_));
  if (d_csc_val_)
    cudaErrCheck(cudaFree(d_csc_val_));

  // if (d_bcsc_col_)
  //   cudaErrCheck(cudaFree(d_bcsc_col_));
  // if (d_bcsc_row_)
  //   cudaErrCheck(cudaFree(d_bcsc_row_));
  // if (d_bcsc_val_)
  //   cudaErrCheck(cudaFree(d_bcsc_val_));
  // if (d_bcsc_group_ptr_)
  //   cudaErrCheck(cudaFree(d_bcsc_group_ptr_));
  return 0;
}

#endif // SPMSPV_H_
