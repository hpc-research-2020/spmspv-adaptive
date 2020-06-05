#ifndef CONFIG_H_
#define CONFIG_H_

#include <assert.h>
//#define DEBUG

//256 or 512.
#define THREADS_PER_BLOCK 256 //1024
//warning: WARP_SIZE is conflict with cub.
#define LM_WARP_SIZE 32

//#define COL_NUM_PER_BLOCK LM_WARP_SIZE
#define COL_NUM_PER_BLOCK (THREADS_PER_BLOCK/32)

#define MAX_BUCKETS 256 //1024 

//test svm dataset or matrix market?

//statistic test
#define STATISTIC
#define STATISTIC_NONZEROS_PER_COL
#define STATISTIC_NONZEROS_PER_ROW

//spmv test
#define SPMV_SERIAL
#define CUSPARSE
#define CUSPARSE_HYBRID_SPMV
#define CSR5

//spmsv test
#define SMSV_SERIAL
//#define SMSV_UPBOUND
//#define THREE_PHASE

#define SMSV_CUDA  //push-pull-no-sort version
//#define CSC_SORT_CUDA //based on sort, use reduce_by_key to reduce.
//#define CUDA_BUCKET_METHOD  //based on bucket; when it is on, SMSV_CUDA must be on.

//#define BUCKET_SERIAL
//#define BUCKET_LOAD_BALANCE
//#define CSR_SMSV_CUDA //csr based smsv on gpu: matrix driven: initil version
#define CSR5_SPMSPV
//#define BCSC_SPMSPV

#define USE_CUB //open this switch when using cub lib
#define USE_CUB_SCAN
//#define USE_CUB_RADIX_SORT
//#define MODERNGPU_REDUCE_BY_KEY


#ifdef USE_CUB
#include "cub/cub.cuh"
using namespace cub;
#include "cub/util_allocator.cuh"
CachingDeviceAllocator  g_allocator(true);
#endif


#ifdef MODERNGPU_REDUCE_BY_KEY
#include <moderngpu.cuh>
#endif


////#define SORT_BASED_REDUCE
////#define BLOCK_SORT_BASED_REDUCE

//optimize every phase in bucket-based version
//#define SHAMEM_COUNT
//#define SHAMEM_EXTRACT
#define ATOMIC_REDUCTION
//end of optimize every phase in bucket-based version

//#define USE_BIT_VECTOR
#endif //CONFIG_H_ 
