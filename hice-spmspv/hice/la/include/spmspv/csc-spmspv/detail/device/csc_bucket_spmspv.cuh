// This file provides the csc-based bucket spmspv method.
//bucket based method: all inputs are on device memory.
//
//
//
#ifndef CSC_SPMSPV_H_
#define CSC_SPMSPV_H_

//#include "../detail/cuda/common_cuda.h"
//#include "../detail/cuda/utils_cuda.h"

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "spmspv/config.h"
//#include "./print_util.cuh"

//using cub.
#ifdef USE_CUB 
	#define CUB_STDERR
	#include "cub/util_allocator.cuh"
	using namespace cub;
	CachingDeviceAllocator  g_allocator(true);// Caching allocator for device memory
#endif

#ifdef USE_CUB_SCAN
	#include "cub/device/device_scan.cuh"
#endif

// #ifdef USE_MY_SCAN
// #include "csc_sort_scan_fuse.cuh"
// #endif

#ifdef USE_CUB_RADIX_SORT
	#include "cub/device/device_radix_sort.cuh"
#endif

#ifdef SORT_BASED_REDUCE
	#include "cub/device/device_segmented_radix_sort.cuh"
#endif
//end of using cub.

//https://moderngpu.github.io/segreduce.html
//
#ifdef MODERNGPU_REDUCE_BY_KEY
	//#include "include/kernels/reducebykey.cuh"
	#include "kernels/reducebykey.cuh"
	using namespace mgpu;
#endif


//
//
#define TIMING 
//#define PRINT_TEMP_RES

template<typename iT, typename uiT, typename vT>
__global__
void CountElemsPerBucketPerThreadKernel(int x_nnz,
					           /*SparseVec* d_sparse_x,*/
                           iT* d_sparse_x_key,
			                     const iT* d_csc_col,
			                     const iT* d_csc_row,
			                     int rows_per_block,  
			                     int count_width,
			                     int* d_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row_block = 0;
  //count[numofRowBlocks][numNonzerosOfX]:
  if (tid < x_nnz) {
  	//int col_index = d_sparse_x[tid].key;
    int col_index = d_sparse_x_key[tid];
    int begin = d_csc_col[col_index];
  	int end = d_csc_col[col_index+1];
    //std::cout << "beign = " << csc_col[col_index] << ", end = " 
              //<< end << ", len = " << end - csc_col[col_index] << std::endl;
    //std::cout << "len = " << end - csc_col[col_index] << std::endl;
  	for (int j = begin; j < end; j++) {
     	row_block = d_csc_row[j]/rows_per_block;//check.
  		d_count[row_block * count_width + tid]++;
      //atomicAdd(&d_count[row_block * count_width + tid], 1);
  	}
  }
}

//TODO: need to speedup.
template<typename iT, typename uiT, typename vT, int NUM_WARP>
__global__
void bucket_count_opt_kernel(int x_nnz,
                        /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         const iT* d_csc_col,
                         const iT* d_csc_row,
                         int rows_per_block, 
                         int count_height, 
                         int count_width,
                         int* d_count) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;

  int col = bid * COL_NUM_PER_BLOCK + lane_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);


  //read d_count[] to share-mem.
  __shared__ int share_count[MAX_BUCKETS][COL_NUM_PER_BLOCK];
  //for(int j=col; j<end_col; j+=WARP_SIZE){
      //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      for (int i = warp_id; i < count_height; i += NUM_WARP) {
        share_count[i][lane_id] = 0;
      }
  //}
  __syncthreads();

  //for(int j = col; j < end_col; j += WARP_SIZE){
  if (col < end_col) {
      //int col_index = d_sparse_x[col].key;
      int col_index = d_sparse_x_key[col];
      int begin = d_csc_col[col_index];
      int end = d_csc_col[col_index+1];
      for (int i = begin + warp_id; i < end; i += NUM_WARP) {
        row = d_csc_row[i];
        row_block = row/rows_per_block;//check.
        //d_count[row_block * count_width + tid]++;
        //share_count[row_block][j]++;
        atomicAdd(&(share_count[row_block][lane_id]), 1);
      }
  }
  //}

  __syncthreads();
  //for(int j=col; j<end_col; j+=WARP_SIZE){
  //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
  if (col < end_col) {
    for (int i = warp_id; i < count_height; i += NUM_WARP) {
        d_count[i*count_width + col] = share_count[i][lane_id];
    }
  }  
  //}
}

//TODO: need to speedup.
//LANE_LEN => WARP_SIZE = 32.
template<typename iT, typename uiT, typename vT, int LANE_LEN>
__global__
void bucket_count_opt_coalesced_kernel(int x_nnz,
                         /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         const iT* d_csc_col,
                         const iT* d_csc_row,
                         int rows_per_block, 
                         int count_height, 
                         int count_width,
                         int* d_count) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;

  int col = bid * COL_NUM_PER_BLOCK + warp_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);


  //read d_count[] to share-mem.
  __shared__ int share_count[MAX_BUCKETS][COL_NUM_PER_BLOCK];
  //for(int j=col; j<end_col; j+=WARP_SIZE){
      //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      for (int i = lane_id; i < count_height; i += LANE_LEN) {
        share_count[i][warp_id] = 0;
      }
  //}
  __syncthreads();

  //for(int j = col; j < end_col; j += WARP_SIZE){
  if (col < end_col) {
    //int col_index = d_sparse_x[col].key;
    int col_index = d_sparse_x_key[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
    for (int i = begin + lane_id; i < end; i += LANE_LEN) {
      row = d_csc_row[i];//合并访问
      row_block = row/rows_per_block;//check.
      //d_count[row_block * count_width + tid]++;
      //share_count[row_block][j]++;
      atomicAdd(&(share_count[row_block][warp_id]), 1);
    }
  }
  //}

  __syncthreads();
  //for(int j=col; j<end_col; j+=WARP_SIZE){
          //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      if (col < end_col) {
        for (int i = lane_id; i < count_height; i += LANE_LEN){
          d_count[i*count_width + col] = share_count[i][warp_id];//
        }
      } 
  //}
}

template<typename iT, typename uiT, typename vT, int LANE_LEN>
__global__
void bucket_count_opt_coalesced_avoid_bankconflict_kernel(int x_nnz,
                          /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         const iT* d_csc_col,
                         const iT* d_csc_row,
                         int rows_per_block, 
                         int count_height, 
                         int count_width,
                         int* d_count) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;

  int col = bid * COL_NUM_PER_BLOCK + warp_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);


  //read d_count[] to share-mem.
  __shared__ int share_count[COL_NUM_PER_BLOCK][MAX_BUCKETS];
  //for(int j=col; j<end_col; j+=WARP_SIZE){
      //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      for (int i = lane_id; i < count_height; i += LANE_LEN) {
        share_count[warp_id][i] = 0;
      }
  //}
  __syncthreads();

  //for(int j = col; j < end_col; j += WARP_SIZE){
  if (col < end_col) {
    //int col_index = d_sparse_x[col].key;
    int col_index = d_sparse_x_key[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
    for (int i = begin + lane_id; i < end; i += LANE_LEN) {
      row = d_csc_row[i];//合并访问
      row_block = row/rows_per_block;//check.
      //d_count[row_block * count_width + tid]++;
      //share_count[row_block][j]++;
      atomicAdd(&(share_count[warp_id][row_block]), 1);
    }
  }
  //}
  __syncthreads();
  //for(int j=col; j<end_col; j+=WARP_SIZE){
          //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      if (col < end_col) {
        for (int i = lane_id; i < count_height; i += LANE_LEN) {
          d_count[i*count_width + col] = share_count[warp_id][i];//
        }
      } 
  //}
}

//d_count和d_csc_row都是合并访问的。
template<typename iT, typename uiT, typename vT, int LANE_LEN>
__global__
void bucket_count_opt_all_coalesced_avoid_bankconflict_kernel(int x_nnz,
                        /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         const iT* d_csc_col,
                         const iT* d_csc_row,
                         int rows_per_block, 
                         int count_height, 
                         int count_width,
                         int* d_count) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;

  int col = bid * COL_NUM_PER_BLOCK + warp_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);


  //read d_count[] to share-mem.
  __shared__ int share_count[COL_NUM_PER_BLOCK][MAX_BUCKETS];
  //for(int j=col; j<end_col; j+=WARP_SIZE){
      //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      for (int i = lane_id; i < count_height; i += LANE_LEN) { 
          share_count[warp_id][i] = 0;
      }
  //}
  __syncthreads();

  //for(int j = col; j < end_col; j += WARP_SIZE){
  if (col < end_col) {
    //int col_index = d_sparse_x[col].key;
    int col_index = d_sparse_x_key[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
    for (int i = begin + lane_id; i < end; i += LANE_LEN) {
      row = d_csc_row[i];//合并访问
      row_block = row/rows_per_block;//check.
      //d_count[row_block * count_width + tid]++;
      //share_count[row_block][j]++;
      atomicAdd(&(share_count[warp_id][row_block]), 1);
    }
  }
  //}

  __syncthreads();
  //for(int j=col; j<end_col; j+=WARP_SIZE){
          //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
  //TODO: need to test.
  col = bid * COL_NUM_PER_BLOCK + lane_id;
  end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);
  if (col < end_col) {
    for (int i = warp_id; i < count_height; i += LANE_LEN) {
      d_count[i*count_width + col] = share_count[lane_id][i];
    }
  }   
  //}
}

template<typename iT, typename uiT, typename vT>
__global__
void BucketCopyKernel(int row, int col,
                        int* d_count,
                        int* d_out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if( tid <= row ){
      d_out[tid] = d_count[col * tid];
  } 
}

template<typename iT, typename uiT, typename vT>
__global__
void ExtractBucketKernel(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val, 
                         int x_nnz,   /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         int rows_per_block, 
                         int count_height, int count_width,
                         int* d_count, 
                         iT* d_bin_row, vT* d_bin_val) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = 0;
  int row_block = 0;
  int index = 0;

  if (tid < x_nnz) {
    //int col_index = d_sparse_x[tid].key;
    int col_index = d_sparse_x_key[tid];
    VALUE_TYPE xVal = d_sparse_x_val[tid];
    int end = d_csc_col[col_index + 1];
    for (int j = d_csc_col[col_index]; j < end; j++) {
        row = d_csc_row[j];
        row_block = row/rows_per_block;//check.
        index = d_count[row_block * count_width + tid]++;//
        
        d_bin_row[index] = row;
        d_bin_val[index] = d_csc_val[j] * xVal;
    }
  }
}

template<typename iT, typename uiT, typename vT, int NUM_WARP>
__global__
void bucket_extract_opt_kernel(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val, 
                         int x_nnz,   /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         int rows_per_block, 
                         int count_height, int count_width,
                         int* d_count, 
                         iT* d_bin_row, vT* d_bin_val) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;
  int index = 0;

  int col = bid * COL_NUM_PER_BLOCK + lane_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);

  //read d_count[] to share-mem.
  __shared__ int share_count[MAX_BUCKETS][COL_NUM_PER_BLOCK];
  if (col < end_col) {
    //for(int j=col; j<end_col; j+=WARP_SIZE){
      //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      for (int i = warp_id; i < count_height; i += NUM_WARP) {
        share_count[i][lane_id] = d_count[i*count_width + col];//
        // if(bid == 0){
        //     printf("share_count[%d][%d] = %d\n", i, j, share_count[i][j]);
        // }
      }
    //}
  }
  __syncthreads();

#if 1
  if (col < end_col) {
  //for(int j=col; j<end_col; j+=WARP_SIZE){
    //int col_index = d_sparse_x[col].key;
    int col_index = d_sparse_x_key[col];
    VALUE_TYPE xVal = d_sparse_x_val[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
       
#if 1
    for (int i = begin + warp_id; i < end; i += NUM_WARP) {//
      row = d_csc_row[i];//
      row_block = row/rows_per_block;//check.
      //index = share_count[row_block][j]++;// = d_count[i*count_width + col];
      index = atomicAdd(&(share_count[row_block][lane_id]), 1);
      //if (j==0)
      //   printf("index = %d\n", index);
      d_bin_row[index] = row;
      d_bin_val[index] = d_csc_val[i] * xVal;
    }
#endif
  //}
  }
#endif
}

//LANE_LEN <=>WARP_SIZE
template<typename iT, typename uiT, typename vT, int LANE_LEN>
__global__
void bucket_extract_opt_coalesced_kernel(iT* d_csc_row, iT* d_csc_col, 
                         vT* d_csc_val, 
                         int x_nnz,   /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         int rows_per_block, 
                         int count_height, int count_width,
                         int* d_count, 
                         iT* d_bin_row, vT* d_bin_val) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;
  int index = 0;

  int col = bid * COL_NUM_PER_BLOCK + warp_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);

  //read d_count[] to share-mem.
  __shared__ int share_count[MAX_BUCKETS][COL_NUM_PER_BLOCK];
  if (col < end_col) {
    //for(int j=col; j<end_col; j+=WARP_SIZE){
      //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
      for (int i = lane_id; i < count_height; i += LANE_LEN) {
        share_count[i][warp_id] = d_count[i*count_width + col];//
        // if(bid == 0){
        //     printf("share_count[%d][%d] = %d\n", i, j, share_count[i][j]);
        // }
      }
    //}
  }
  __syncthreads();

#if 1
  if (col < end_col) {
  //for(int j=col; j<end_col; j+=WARP_SIZE){
    //int col_index = d_sparse_x[col].key;
    int col_index = d_sparse_x_key[col];
    //VALUE_TYPE xVal = d_sparse_x[col].val;
    VALUE_TYPE xVal = d_sparse_x_val[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
     
#if 1
    for (int i = begin + lane_id; i < end; i += LANE_LEN) {
      row = d_csc_row[i];//合并访问d_csc_row
      row_block = row/rows_per_block;//check.
      //index = share_count[row_block][j]++;// = d_count[i*count_width + col];
      index = atomicAdd(&(share_count[row_block][warp_id]), 1);
      //if(j==0)
        //   printf("index = %d\n", index);
      d_bin_row[index] = row;
      d_bin_val[index] = d_csc_val[i] * xVal;
    }
#endif
  //}
  }
#endif
}

template<typename iT, typename uiT, typename vT, int LANE_LEN>
__global__
void bucket_extract_opt_coalesced_avoid_bankconflict_kernel(iT* d_csc_row, 
                         iT* d_csc_col, vT* d_csc_val, 
                         int x_nnz, /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         int rows_per_block, 
                         int count_height, int count_width,
                         int* d_count, 
                         iT* d_bin_row, vT* d_bin_val) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;
  int index = 0;

  int col = bid * COL_NUM_PER_BLOCK + warp_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);


  //read d_count[] to share-mem.
  __shared__ int share_count[COL_NUM_PER_BLOCK][MAX_BUCKETS];
  if(col < end_col){
      //for(int j=col; j<end_col; j+=WARP_SIZE){
          //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
          for(int i=lane_id; i<count_height; i+=LANE_LEN){
              share_count[warp_id][i] = d_count[i*count_width + col];//不合并访问d_count
              // if(bid == 0){
              //     printf("share_count[%d][%d] = %d\n", i, j, share_count[i][j]);
              // }
          }
      //}
  }
  __syncthreads();

#if 1
  if (col < end_col) {
  //for(int j=col; j<end_col; j+=WARP_SIZE){
    //int col_index = d_sparse_x[col].key;
    //VALUE_TYPE xVal = d_sparse_x[col].val;
    int col_index = d_sparse_x_key[col];
    VALUE_TYPE xVal = d_sparse_x_val[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
#if 1
    for (int i = begin + lane_id; i < end; i += LANE_LEN) {
      row = d_csc_row[i];//合并访问d_csc_row
      row_block = row/rows_per_block;//check.
      //index = share_count[row_block][j]++;// = d_count[i*count_width + col];
      index = atomicAdd(&(share_count[warp_id][row_block]), 1);
      //if(j==0)
        //   printf("index = %d\n", index);
      d_bin_row[index] = row;
      d_bin_val[index] = d_csc_val[i] * xVal;
    }
#endif
  //}
  }
#endif
}
 
template<typename iT, typename uiT, typename vT, int LANE_LEN>
__global__
void bucket_extract_opt_all_coalesced_avoid_bankconflict_kernel(
                         iT* d_csc_row, iT* d_csc_col, vT* d_csc_val, 
                         int x_nnz, /*SparseVec* d_sparse_x,*/
                        iT* d_sparse_x_key,
                        vT* d_sparse_x_val,
                         int rows_per_block, 
                         int count_height, int count_width,
                         int* d_count, 
                         iT* d_bin_row, vT* d_bin_val) {
  int bid = blockIdx.x;
  //int tid = bid * blockDim.x + threadIdx.x;
  // warp lane id
  int warp_id = threadIdx.x >> 5;//divide 32.
  int lane_id = 31 & threadIdx.x;//

  int row = 0;
  int row_block = 0;
  int index = 0;

  int col = bid * COL_NUM_PER_BLOCK + lane_id;//
  int end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);
  //read d_count[] to share-mem.
  __shared__ int share_count[COL_NUM_PER_BLOCK][MAX_BUCKETS];
  if (col < end_col) {
    //printf("tid = %d, col=%d, end_col=%d\n", tid, col, end_col);
    for (int i = warp_id; i < count_height; i += LANE_LEN) {
      share_count[lane_id][i] = d_count[i*count_width + col];//
      // if(bid == 0){
      //     printf("share_count[%d][%d] = %d\n", i, j, share_count[i][j]);
      // }
    }
  }
  __syncthreads();
  
  col = bid * COL_NUM_PER_BLOCK + warp_id;//
  end_col = min((bid+1)*COL_NUM_PER_BLOCK, count_width);

#if 1
  if (col < end_col) {
  //for(int j=col; j<end_col; j+=WARP_SIZE){
    // int col_index = d_sparse_x[col].key;
    // VALUE_TYPE xVal = d_sparse_x[col].val;
    int col_index = d_sparse_x_key[col];
    VALUE_TYPE xVal = d_sparse_x_val[col];
    int begin = d_csc_col[col_index];
    int end = d_csc_col[col_index+1];
   
#if 1
    for (int i = begin + lane_id; i < end; i += LANE_LEN) {
      row = d_csc_row[i];//
      row_block = row/rows_per_block;//check.
      //index = share_count[row_block][j]++;// = d_count[i*count_width + col];
      index = atomicAdd(&(share_count[warp_id][row_block]), 1);
      //if(j==0)
      //  printf("index = %d\n", index);
      d_bin_row[index] = row;
      d_bin_val[index] = d_csc_val[i] * xVal;
    }
#endif
  //}
  }
#endif
}

template<typename iT, typename uiT, typename vT>
__global__
void BucketReductionKernel(int buckets, int* d_addr, 
                            iT* d_bin_row, vT* d_bin_val, 
                            vT* d_y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int begin_index = 0;
  int end_index  = 0;
  if (tid < buckets) {
    begin_index = d_addr[tid];
    end_index  = d_addr[tid + 1];
    //printf("num_buckets = %d\n",  end_index - begin_index);
    //std::cout << "num_buckets " << tid << " = " << end_index - begin_index <<std::endl;
    for (int j = begin_index; j < end_index; j++) {
        d_y[d_bin_row[j]] += d_bin_val[j];
    }
  }
}

template<typename iT, typename uiT, typename vT>
__global__
void AtomicBucketReductionKernel(int buckets, int* d_addr, 
                            iT* d_bin_row, vT* d_bin_val, 
                            vT* d_y) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int begin_index = 0;
  int end_index  = 0;
  if (bid < buckets) {
    begin_index = d_addr[bid];
    end_index  = d_addr[bid + 1];
    //output num of each bucket.
    // if(tid == 0){
    //    printf("buckets=%d, num_buckets=%d\n", bid, end_index-begin_index);
    // }
    for (int j = begin_index + tid; j < end_index; j += blockDim.x) {
      int row_index = d_bin_row[j];
      int value = d_bin_val[j];
      //double atomicAdd(double* address, double val);
      atomicAdd(&d_y[row_index], value);
      //d_y[row_index] += value;
    }
  }
}

template<typename iT, typename uiT, typename vT>
void bucket_reduction_sort_based_driver(int buckets, int* d_addr, 
                            iT* d_bin_row, vT* d_bin_val, 
                            vT* d_y) {
    
}


#if 0 
//#include <cub/cub.cuh>
// Block-sorting CUDA kernel
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
     using namespace cub;

     // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads 
     // owning 16 integer items each
     //param: the data type being sorted, by the number of threads per block, 
     //by the number of keys per thread
     typedef BlockRadixSort<int, 128, 16>                    BlockRadixSort;
     typedef BlockLoad<int, 128, 16, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     typedef BlockStore<int, 128, 16, BLOCK_STORE_TRANSPOSE> BlockStore;
 
     // Allocate shared memory
     __shared__ union {
         typename BlockRadixSort::TempStorage  sort;
         typename BlockLoad::TempStorage       load; 
         typename BlockStore::TempStorage      store; 
     } temp_storage; 

     int block_offset = blockIdx.x * (128 * 16);	  // OffsetT for this block's ment

     // Obtain a segment of 2048 consecutive keys that are blocked across threads
     int thread_keys[16];
     BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
     __syncthreads();

     // Collectively sort the keys
     BlockRadixSort(temp_storage.sort).Sort(thread_keys);
     __syncthreads();

     // Store the sorted segment 
     BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}


//ref:
//http://nvlabs.github.io/cub/classcub_1_1_block_radix_sort.html#details

//http://nvlabs.github.io/cub/classcub_1_1_block_radix_sort.html

//如何在块内调用cudpp中的函数？？
template<typename iT, typename uiT, typename vT>
__global__
void bucket_reduction_opt_driver(int buckets, int* d_addr, 
                            iT* d_bin_row, vT* d_bin_val, 
                            vT* d_y)
{
	int bid = blockIdx;
    int tid = bid * blockDim.x + threadIdx.x;
    int begin_index = 0;
    int end_index  = 0;
    if(bid < buckets){
        begin_index = d_addr[tid];
        end_index  = d_addr[tid + 1];
        printf("num_buckets = %d\n",  end_index - begin_index);
        //std::cout << "num_buckets " << tid << " = " << end_index - begin_index <<std::endl;
        // for(int j=begin_index; j<end_index; j++){
        //     d_y[d_bin_row[j]] += d_bin_val[j];
        // }
    }
}
#endif

//dense y.
template<typename iT, typename uiT, typename vT>
int CscBasedBucketSpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
        				iT m, iT n, iT nnz, 
        				iT x_nnz, /*SparseVec* d_sparse_x,*/
        				const vT  alpha, 
                iT* d_sparse_x_key,
                vT* d_sparse_x_val,
                iT* ynnz, vT* d_y,
                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                int num_buckets,
                int blocksPerGrid, int threadsPerBLOCK) {
  int err = SPMSPV_SUCCESS;
#ifdef TIMING
  std::cout << "call cuda_smsv_bucket version." << std::endl;
  std::cout << "step-1: bucket." << std::endl;
#endif
  //std::cout << "debug: x_nnz = " << x_nnz << std::endl;

  //int rows_per_block = 128; //128;
  //int buckets = ceil(m/(double)rows_per_block);
  //TODO: how to set the bucket????
  int buckets = std::min<int>(num_buckets, (m+127/(double)128));//
  int rows_per_block = std::max<int>(128, ceil(m/(double)buckets));

#ifdef TIMING  
  std::cout << "rows_per_block = " << rows_per_block << ", buckets = " 
            << buckets <<std::endl;
#endif

  int count_width = x_nnz;
  int count_len = (count_width * buckets + 1);

  // int* d_count = NULL;
  // cudaErrCheck(cudaMalloc((void **)&d_count, (count_len * sizeof(int))));
  // cudaMemset(d_count, 0, count_len * sizeof(int));
  // int begin_addr_len = (buckets+1);
  // int* d_begin_addr = NULL;
  // cudaErrCheck(cudaMalloc((void **)&d_begin_addr, (begin_addr_len * sizeof(int))));
  int* d_count = (int*)d_pre_alloc_buffer;
  cudaErrCheck(cudaMemset(d_count, 0, count_len * sizeof(int)));

  int begin_addr_len = (buckets+1);
  int* d_begin_addr = (int*)(d_pre_alloc_buffer + count_len * sizeof(int));

  //std::cout << "debug" << (long)d_begin_addr-(long)d_count << std::endl;
#ifdef TIMING 
  std::cout << "count_width = " << count_width << ", count_len = " 
            << count_len << std::endl;
#endif

#ifdef TIMING
  SpmspvTimer step1_timer;
  step1_timer.Start();
#endif

#ifndef SHAMEM_COUNT
 	int threads = THREADS_PER_BLOCK;
    int num_blocks = ceil((x_nnz)/(double)(threads));
 	CountElemsPerBucketPerThreadKernel<iT, uiT, vT> <<< num_blocks, threads >>>
                                   (x_nnz, /*d_sparse_x,*/
                                   d_sparse_x_key,
                                   d_csc_col, d_csc_row, 
                                   rows_per_block, 
                                   count_width, d_count);
#else
  int threads = THREADS_PER_BLOCK;
  int num_blocks = (x_nnz+COL_NUM_PER_BLOCK-1)/COL_NUM_PER_BLOCK;
#ifdef TIMING
  printf("threads=%d, num_blocks=%d\n", threads, num_blocks);
#endif

  bucket_count_opt_kernel<iT, uiT, vT, THREADS_PER_BLOCK/WARP_SIZE> 
                        <<< num_blocks, threads >>> 
                       (x_nnz, /*d_sparse_x,*/
                       d_sparse_x_key, d_sparse_x_val, 
                       d_csc_col, d_csc_row, 
                       rows_per_block, 
                       buckets, count_width, 
                       d_count); 
   // bucket_count_opt_all_coalesced_avoid_bankconflict_kernel<iT, uiT, vT, 32> 
   //                     <<< num_blocks, threads >>> 
   //                      (x_nnz, d_sparse_x, 
   //                       d_csc_col, d_csc_row, 
   //                       rows_per_block, 
   //                       buckets, count_width, 
   //                       d_count);
#endif

#ifdef TIMING
  double step1_time = step1_timer.Stop();
  std::cout << "step-1 count time = " << step1_time << " ms." << std::endl;
  //cudaDeviceSynchronize();
  cudaError_t  err_r = cudaGetLastError();
  if (cudaSuccess != err_r) {
    std::cout << "bucket_count_opt_kernel() invocate error." << std::endl;
    //printf("bucket_count_opt_kernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
  }
#endif

#ifdef PRINT_TEMP_RES
  PrintMat2FileDevice(buckets, count_width, d_count, "count_matrix.info");
#endif

  thrust::device_ptr<int> d_count_ptr = thrust::device_pointer_cast(d_count);

#ifdef TIMING
  SpmspvTimer step2_timer;
  step2_timer.Start();
#endif


#ifndef USE_CUB_SCAN
  thrust::exclusive_scan(d_count_ptr, d_count_ptr + count_len, d_count_ptr);
#else
  void  *d_temp_storage = NULL;
  size_t  temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_count, d_count, count_len));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
              temp_storage_bytes));

  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
               d_count, d_count, count_len));

  if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

#ifdef TIMING
  double step2_time = step2_timer.Stop();
  std::cout << "step-2 exclusive_scan time = " << step2_time 
            << " ms." << std::endl;
#endif

#ifdef PRINT_TEMP_RES
  PrintMat2FileDevice(buckets, count_width, d_count, 
                          "scan_count_matrix.info");
#endif

#ifdef TIMING
  SpmspvTimer step3_timer;
  step3_timer.Start();
#endif

  threads = min(1024, (buckets + 1 + 31) / 32 * 32);
  num_blocks = (buckets + 1 + threads - 1)/threads;
  BucketCopyKernel<iT, uiT, vT><<< num_blocks, threads >>>
      (buckets, count_width, d_count, d_begin_addr);

#ifdef TIMING
  double step3_time = step3_timer.Stop();
  std::cout << "step-3 copy time = " << step3_time << " ms." << std::endl;
#endif

  int len;
  cudaErrCheck(cudaMemcpy(&len, &d_count[count_len-1], sizeof(int),   
                          cudaMemcpyDeviceToHost));
  const int bin_len = len;
#ifdef TIMING
  std::cout << "bebug: bin_len = " << bin_len <<std::endl;
#endif
  // int* buffer_row = (int*)malloc(bin_len * sizeof(int));
  // VALUE_TYPE* buffer_val = (VALUE_TYPE*)malloc(bin_len * sizeof(VALUE_TYPE));
  // CHECK_MALLOC(buffer_row);
  // CHECK_MALLOC(buffer_val);
#if 0
  iT* d_bin_row = NULL;
  vT* d_bin_val = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_bin_row,  bin_len * sizeof(iT)));
  cudaErrCheck(cudaMalloc((void **)&d_bin_val,  bin_len * sizeof(vT)));
#endif

   iT* d_bin_row = (iT*)(d_begin_addr + begin_addr_len * sizeof(int));
   
   void* temp = (void*)(d_bin_row + bin_len * sizeof(iT));
   int size_vT = sizeof(vT);
   vT* d_bin_val = (vT*)((reinterpret_cast<std::size_t>(temp) + size_vT - 1) /
                   size_vT * size_vT);
   
   //vT* d_bin_val = (vT*)((void*)d_bin_row + bin_len * sizeof(iT));
   //void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(d_bin_val) & 
	 //		                                   ~(std::size_t( sizeof(vT) - 1))) + sizeof(vT));
   //d_bin_val = (vT*)aligned;
   //for debug. 
   // //std::cout << "debug" << (long)d_bin_row-(long)d_begin_addr << std::endl;
   // //std::cout << "debug" << (long)d_bin_val-(long)d_bin_row << std::endl;
   // //std::cout << "debug" << d_bin_val << std::endl;

  // threads = ;
  // num_blocks = ;
#ifdef TIMING
  SpmspvTimer step4_timer;
  step4_timer.Start();
#endif

#ifndef SHAMEM_EXTRACT
  ExtractBucketKernel<iT, uiT, vT><<< blocksPerGrid, threadsPerBLOCK >>>
                                  (d_csc_row, d_csc_col, d_csc_val, 
                                   x_nnz, /*d_sparse_x,*/
                                   d_sparse_x_key, d_sparse_x_val, 
                                   rows_per_block, 
                                   buckets, count_width,
                                   d_count, 
                                   d_bin_row, d_bin_val);
#else
  threads = THREADS_PER_BLOCK;//1024
  num_blocks = (x_nnz+COL_NUM_PER_BLOCK-1)/COL_NUM_PER_BLOCK;//
  //printf("threads=%d, num_blocks=%d\n", threads, num_blocks);
  bucket_extract_opt_kernel<iT, uiT, vT, THREADS_PER_BLOCK/WARP_SIZE>
                          <<< num_blocks, threads >>>
                          (d_csc_row, d_csc_col, d_csc_val, 
                           x_nnz, /*d_sparse_x,*/
                           d_sparse_x_key, d_sparse_x_val, 
                           rows_per_block, 
                           buckets, count_width,
                           d_count, 
                           d_bin_row, d_bin_val);                               
  // bucket_extract_opt_all_coalesced_avoid_bankconflict_kernel<iT, uiT, vT, 32>
  //                         <<< num_blocks, threads >>>
  //                         (d_csc_row, d_csc_col, d_csc_val, 
  //                          x_nnz, d_sparse_x, 
  //                          rows_per_block, 
  //                          buckets, count_width,
  //                          d_count, 
  //                          d_bin_row, d_bin_val); 
#endif

#ifdef TIMING
  double step4_time = step4_timer.Stop();
  std::cout << "step-4 extract time = " << step4_time << " ms." << std::endl;
#endif                             

#if 0
#ifdef TIMING
  SpmspvTimer step5_timer;
  step5_timer.Start();
#endif

#ifndef ATOMIC_REDUCTION
  threads = min(1024, (buckets + 31) / 32 * 32);
  num_blocks = (buckets + threads - 1)/threads;

  BucketReductionKernel<iT, uiT, vT><<< num_blocks, threads >>>
                                  (buckets, d_begin_addr, 
                                  d_bin_row, d_bin_val, 
                                  d_y);
#else
  #ifndef SORT_BASED_REDUCE
    threads = 1024;
    num_blocks = buckets;//
    AtomicBucketReductionKernel<iT, uiT, vT><<< num_blocks, threads >>>
                                    (buckets, d_begin_addr, 
                                    d_bin_row, d_bin_val, 
                                    d_y);
  #else
      //http://nvlabs.github.io/cub/structcub_1_1_device_segmented_radix_sort.html#a0ce82f886816d1b7769ba148210538bb
    d_temp_storage = NULL;
	  temp_storage_bytes = 0;
	  CubDebugExit(DeviceSegmentedRadixSort::SortPairs(NULL, temp_storage_bytes,
  										  d_bin_row, d_bin_row, d_bin_val, d_bin_val,
  											bin_len, buckets, d_begin_addr, d_begin_addr + 1));
  	// Allocate temporary storage
  	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
                  temp_storage_bytes));
	//cudaMalloc(&d_temp_storage, temp_storage_bytes);
  	// Run sorting operation
  	CubDebugExit(DeviceSegmentedRadixSort::SortPairs(d_temp_storage, 
                           temp_storage_bytes,
      										 d_bin_row, d_bin_row, d_bin_val, d_bin_val,
    											 bin_len, buckets, d_begin_addr, d_begin_addr + 1));
  
    // CubDebugExit( DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
    //                                 (iT*)d_bin_row, (iT*)d_bin_row, (vT*)d_bin_val, 
    //                                 (vT*)d_bin_val, bin_len) );

    // CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    //                                 (iT*)d_bin_row, (iT*)d_bin_row, (vT*)d_bin_val, 
    //                                 (vT*)d_bin_val, bin_len) );

	if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    #if 0
        threads = 1024;
        num_blocks = buckets;//note
        AtomicBucketReductionKernel<iT, uiT, vT><<< num_blocks, threads >>>
                                        (buckets, d_begin_addr, 
                                        d_bin_row, d_bin_val, 
                                        d_y);
    #endif
  #endif
#endif

#ifdef TIMING
  double step5_time = step5_timer.Stop();
  std::cout << "step-5 reduction time = " << step5_time << " ms." << std::endl;
#endif

#ifdef PRINT_TEMP_RES
  PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, 
                                      "bin.info");
#endif

#endif

#ifdef TIMING
  // double all_time = step1_time + step2_time + step3_time 
  //                 + step4_time + step5_time;
  double all_time = step1_time + step2_time + step3_time 
                  + step4_time;             
  std::cout << "all time = " << all_time << "ms." << std::endl;
#endif

  // if (d_count)      cudaErrCheck(cudaFree(d_count));
  // if (d_begin_addr) cudaErrCheck(cudaFree(d_begin_addr));
  // if (d_bin_row)    cudaErrCheck(cudaFree(d_bin_row));
  // if (d_bin_val)    cudaErrCheck(cudaFree(d_bin_val));
  return err;
}
#endif //CSC_SPMSPV_H_
