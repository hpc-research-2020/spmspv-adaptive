#ifndef SORT_TWO_LAYER_BALANCE_METHOD_H_
#define SORT_TWO_LAYER_BALANCE_METHOD_H_


#include "spmspv/config.h"

#define SPMSPV_DIV_UP(x, y) (((x) + (y) - 1) / (y))

#ifdef _DEBUG
 #define MGPU_SYNC_CHECK(s) {                        \
  cudaError_t error = cudaDeviceSynchronize();              \
  if(cudaSuccess != error) {                        \
    printf("CUDA ERROR %d %s\n%s:%d.\n%s\n",              \
    error, cudaGetErrorString(error), __FILE__, __LINE__, s);   \
    exit(0);                              \
  }                                   \
 }
#else
#define MGPU_SYNC_CHECK(s)
#endif

template<typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
  return (a + b - 1) / b;
}

/*****************************************/
/*****************************************/
// DeviceMemToMem4Indirect<NT, 4>(x_nnz - i, csc_col, csc_row + i, 
//                                   tid, dest + i);

// template<int NT, int VT>
// MGPU_DEVICE void DeviceMemToMem4Indirect(int count, InputIt source, 
//   SourceIt load, int tid, OutputIt dest) {
//VT is 4.
template<int NT, int VT>
__device__ inline void twoLayerDeviceMemToMem4Indirect(
              int x_nnz, int* csc_col, 
              int* x_sparse_index, int tid, int* dest) {
  //typedef typename std::iterator_traits<int*>::value_type T;
  int x[VT];
  const int Count = (VT < 4) ? VT : 4;//
  if (x_nnz >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < Count; ++i)
      x[i] = csc_col[x_sparse_index[NT * i + tid]];
    #pragma unroll
    for(int i = 0; i < Count; ++i)
      dest[NT * i + tid] = x[i];
  } else {
    // deal with left.
    #pragma unroll
    for (int i = 0; i < Count; ++i) {
      int index = NT * i + tid;
      if(index < x_nnz)
        x[i] = csc_col[x_sparse_index[index]];
    }
    #pragma unroll
    for (int i = 0; i < Count; ++i) {
      int index = NT * i + tid;
      if(index < x_nnz)
        dest[index] = x[i];
    }
  }
  //__syncthreads();
}

template<int NT>
__device__ inline void twoLayerDeviceMemToMemLoopIndirect(
                                int x_nnz, int* csc_col, 
                                int* x_sparse_index, int tid, int* dest) {
  for (int i = 0; i < x_nnz; i += 4 * NT)
    twoLayerDeviceMemToMem4Indirect<NT, 4>(x_nnz - i, csc_col, 
                                          x_sparse_index + i, 
                                          tid, dest + i);
  __syncthreads();
}

template<int NT, int VT, typename Type>
__device__ inline void twoLayerDeviceGather(int left_over, 
                                            int bin_len, Type* csc_val, 
                                            int indices[VT], 
                                            int tid, Type* reg) {
  if (bin_len >= NT * VT) {//
    #pragma unroll
    for (int i = 0; i < VT; ++i)
      reg[i] = csc_val[indices[i]];
  } else {
    #pragma unroll
    if (tid * VT < bin_len) {
      int index_2 = (tid + 1) * VT;
      if (bin_len >= index_2) {
        for (int i = 0; i < VT; ++i) {
          reg[i] = csc_val[indices[i]];
        }
      } else {
        for (int i = 0; i < left_over; ++i)
          reg[i] = csc_val[indices[i]];
      }
    }
  }
}

template<int NT, int VT, typename Type>
__device__ __forceinline__ void twoLayerDeviceRegToGlobal(
                                  int left_over,
                                  int bin_len, const Type* reg, 
                                  int tid, Type* bin_row) {
  if (bin_len >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      int index = VT * tid + i;
      bin_row[index] = reg[i];
    }
  } else {
    #pragma unroll
    if (tid * VT < bin_len) {
      int index_2 = (tid + 1) * VT;
      if (index_2 <= bin_len) {
        for (int i = 0; i < VT; ++i) {
          int index = VT * tid + i;
           bin_row[index] = reg[i];
        }
      } else {
        for (int i = 0; i < left_over; ++i) {
          int index = VT * tid + i;
          bin_row[index] = reg[i];
        }
      }
    }
  }
}

template<int THREADS, int NNZ_PER_THREAD>
__device__ __forceinline__ int threadStartIds(int blockRowStart, 
                int blockNumRows, const int* __restrict__ d_scan, 
                int* smem_starting_rows) {
  //d_scan[r + blockRowStart] = scan_shared[r]
  for (int r = threadIdx.x; r < blockNumRows; r += THREADS) {
    int ain = static_cast<int>(d_scan[r + blockRowStart] 
            - blockIdx.x * THREADS * NNZ_PER_THREAD);
    int bin = d_scan[r + blockRowStart + 1] 
            - blockIdx.x * THREADS * NNZ_PER_THREAD;

    int a = max(0, ain);
    int b = min(static_cast<int>(THREADS * NNZ_PER_THREAD), bin) - 1;

    int threada = divup<int>(a, static_cast<int>(NNZ_PER_THREAD));
    int threadb = b / static_cast<int>(NNZ_PER_THREAD);

    //iterate over all threads that start with that row
    for (; threada <= threadb; ++threada) {
      smem_starting_rows[threada] = r;
      // printf("bid = %d, tid = %d, smem_starting_rows[%d]=%d\n", 
      //         blockIdx.x, threadIdx.x, 
      //         threada, smem_starting_rows[threada]);
    }
  }
}


template<int THREADS, int NNZ_PER_THREAD>
__device__ __forceinline__ int checkThreadStartIds(
                  int* smem_starting_rows) {

}


template<int THREADS, int NNZ_PER_THREAD>
__device__ __forceinline__ int threadStartIds_shared(int blockRowStart, 
                int blockNumRows, int* scan_shared, 
                int* smem_starting_rows) {
  //d_scan[r + blockRowStart] = scan_shared[r]
  for (int r = threadIdx.x; r < blockNumRows; r += THREADS) {
    //int ain = static_cast<int>(d_scan[r + blockRowStart] 
    //        - blockIdx.x * THREADS * NNZ_PER_THREAD);
    //int bin = d_scan[r + blockRowStart + 1] 
    //        - blockIdx.x * THREADS * NNZ_PER_THREAD;
    int ain = static_cast<int>(scan_shared[r] 
    	    - blockIdx.x * THREADS * NNZ_PER_THREAD);
    int bin = static_cast<int>(scan_shared[r + 1] 
    	    - blockIdx.x * THREADS * NNZ_PER_THREAD);
    int a = max(0, ain);
    int b = min(static_cast<int>(THREADS * NNZ_PER_THREAD), bin) - 1;

    int threada = divup<int>(a, static_cast<int>(NNZ_PER_THREAD));
    int threadb = b / static_cast<int>(NNZ_PER_THREAD);

    //iterate over all threads that start with that row
    for (; threada <= threadb; ++threada) {
      smem_starting_rows[threada] = r;
      // printf("bid = %d, tid = %d, smem_starting_rows[%d]=%d\n", 
      //         blockIdx.x, threadIdx.x, 
      //         threada, smem_starting_rows[threada]);
    }
  }
}


template<int THREADS, int NNZ_PER_THREAD>
__device__ __forceinline__ int allThreadsStartIds(
        int bid, int tid, int blockRowStart, int blockNumRows, 
        int bin_len,
        const int* __restrict__ d_scan, 
        int* smem_starting_rows,
        int* indices, int* rank) {
  int gid = bid * blockDim.x + tid;
  int active_thread = divup<int>(bin_len, NNZ_PER_THREAD);
  int left_over = bin_len%NNZ_PER_THREAD;
  int nnz_per_thread = NNZ_PER_THREAD;
  if (gid == active_thread-1) 
    nnz_per_thread = left_over?left_over:nnz_per_thread;

 
  //int threshhold = min(THREADS, blockNumRows);
  if (gid < active_thread) {
    //len = min(len, NNZ_PER_THREAD);
    int i = 0;
    int begin_row = smem_starting_rows[tid] + blockRowStart;
    int g_begin_row = begin_row + 1 ;
    int current_val = tid * NNZ_PER_THREAD + bid * THREADS * NNZ_PER_THREAD;
    int len = d_scan[g_begin_row] - current_val;
    int deal_len = NNZ_PER_THREAD;

    //d_scan[g_begin_row]
    //      =d_scan[smem_starting_rows[tid] + blockRowStart + 1]
    //      =scan_shared[smem_starting_rows[tid] + 1];
    //d_scan[indices[i]] = shared_scan[indices[i] - blockRowStart];
    while (i < nnz_per_thread) {
      //printf("bid = %d, tid = %d, len=%d\n", bid, tid, len);
      if (len >= deal_len) {
        for (int j=0; j < deal_len; i++, j++) {
          indices[i] = begin_row;
          rank[i] = current_val++ - d_scan[indices[i]];
          //printf("indices[%d]=%d, rank[%d]=%d\n", i, indices[i], i, rank[i]);
        }
      }
      else{
        for (int j=0; j < len; i++, j++) {
          indices[i] = begin_row;
          //TODO.
          rank[i] = current_val++ - d_scan[indices[i]];
          //printf("indices[%d]=%d, rank[%d]=%d\n", i, indices[i], i, rank[i]);
        }
        deal_len -= len;
        begin_row++;
        g_begin_row++;
        //current_val += i;
        len = d_scan[g_begin_row] - current_val;
      }
    } 
    // for(int i=0; i<nnz_per_thread; i++)
    //   printf("bid =%d, tid = %d, indices[%d]=%d, rank[%d]=%d\n", 
    //           bid, tid, i, indices[i], i, rank[i]);
  }
  
}

template<int THREADS, int NNZ_PER_THREAD>
__device__ __forceinline__ int allThreadsStartIds_shared(
        int bid, int tid, int blockRowStart, int blockNumRows, 
        int bin_len,
        int* scan_shared, 
        int* smem_starting_rows,
        int* indices, int* rank) {
  int gid = bid * blockDim.x + tid;
  int active_thread = divup<int>(bin_len, NNZ_PER_THREAD);
  int left_over = bin_len%NNZ_PER_THREAD;
  int nnz_per_thread = NNZ_PER_THREAD;
  if (gid == active_thread-1) 
    nnz_per_thread = left_over?left_over:nnz_per_thread;

  //int threshhold = min(THREADS, blockNumRows);
  if (gid < active_thread) {
    //len = min(len, NNZ_PER_THREAD);
    int i = 0;
    
    int begin_row_offset = smem_starting_rows[tid];
    int g_begin_row_offset = begin_row_offset + 1;
    int begin_row = begin_row_offset + blockRowStart;
    int g_begin_row = begin_row + 1 ;
    
    int current_val = tid * NNZ_PER_THREAD + bid * THREADS * NNZ_PER_THREAD;
    //int len = d_scan[g_begin_row] - current_val;
    int len = scan_shared[g_begin_row_offset] - current_val;
    int deal_len = NNZ_PER_THREAD;

    int* scan_shared2 = scan_shared - blockRowStart; 
    //d_scan[g_begin_row]
    //      =d_scan[smem_starting_rows[tid] + blockRowStart]
    //      =scan_shared[smem_starting_rows[tid]];
    //d_scan[indices[i]] = shared_scan[indices[i] - blockRowStart];
    while (i < nnz_per_thread) {
      //printf("bid = %d, tid = %d, len=%d\n", bid, tid, len);
      if (len >= deal_len) {
        for (int j=0; j < deal_len; i++, j++) {
          indices[i] = begin_row;
          //rank[i] = current_val++ - d_scan[indices[i]];
          rank[i] = current_val++ - scan_shared2[indices[i]];
          //printf("indices[%d]=%d, rank[%d]=%d\n", i, indices[i], i, rank[i]);
        }
      }
      else{
        for (int j=0; j < len; i++, j++) {
          indices[i] = begin_row;
          //TODO.
          //rank[i] = current_val++ - d_scan[indices[i]];
          rank[i] = current_val++ - scan_shared2[indices[i]];
          //printf("indices[%d]=%d, rank[%d]=%d\n", i, indices[i], i, rank[i]);
        }
        deal_len -= len;
        begin_row++;
        //begin_row_offset++;
        g_begin_row_offset++;
        //current_val += i;
        //len = d_scan[g_begin_row] - current_val;
        len = scan_shared[g_begin_row_offset] - current_val;
      }
    } 
    // for(int i=0; i<nnz_per_thread; i++)
    //   printf("bid =%d, tid = %d, indices[%d]=%d, rank[%d]=%d\n", 
    //           bid, tid, i, indices[i], i, rank[i]);
  }
  
}

__device__ __forceinline__ void checkIntervalsShared(
        int blockStart, int blockNumRows, int* intervals_shared) {
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i=0; i<blockNumRows; i++) {
      printf("bid=%d,intervals_shared[%d]=%d,intervals_shared-blockStart=%d\n",
              blockIdx.x, i, intervals_shared[i], 
              intervals_shared[i]-blockStart);
    }
  }
  __syncthreads();
}

__global__ void PrintBinRow(int len, int* d_bin_row) {
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i=0; i<len; i++) {
      printf("bin_row[%d]=%d\n", i, d_bin_row[i]);
    }
  }
  __syncthreads();
}

__global__ void PrintBinRow(int len, double* d_bin_row) {
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i=0; i<len; i++) {
      printf("bin_row[%d]=%f\n", i, d_bin_row[i]);
    }
  }
  __syncthreads();
}


template<typename Type, int NT, int VT>
__device__ __forceinline__ void computeGather(
      int bin_len, int blockRowStart, int* intervals_shared, 
      int* reg_interval, int* reg_rank, int* reg_gather) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    //int* intervals_shared2 = intervals_shared - range.z;
    int* intervals_shared2 = intervals_shared - blockRowStart;
    
    if (bin_len >= NT * VT) {
      #pragma unroll
      for (int i = 0; i < VT; ++i) {
        reg_gather[i] = intervals_shared2[reg_interval[i]] 
                      + reg_rank[i];
        //printf("bid=%d,tid=%d,gather[%d]=%d,intervals_shared2[interval[i]]=%d\n", bid, tid, i, reg_gather[i], intervals_shared2[reg_interval[i]]);
      }
    } else {
      if (tid * VT <= bin_len) {
        #pragma unroll
        for (int i = 0; i < VT; ++i) {
          int index = VT * tid + i;
          if (index < bin_len) {
            reg_gather[i] = intervals_shared2[reg_interval[i]] 
                        + reg_rank[i];
            //printf("bid=%d,tid=%d,gather[%d]=%d,intervals_shared2[interval[i]]=%d\n", bid, tid, i, reg_gather[i], intervals_shared2[reg_interval[i]]);
          }
        }
      }
    }   
}


template<typename Type, int THREADS, int NNZ_PER_THREAD, int NNZ_PER_BLOCK>
__global__ void TwoLayerExtractKernel(
    int bin_len, int* csc_col, const int* __restrict__ d_scan, 
    int x_nnz, Type* csc_val, int* x_sparse_index, 
    const int* d_startingIds, Type* bin_row) {

  const int NT = THREADS;
  const int VT = NNZ_PER_THREAD;
  __shared__ int smem_starting_rows[THREADS];
  //__shared__ int smem_indices[NT * (VT + 1)];
  __shared__ int intervals_shared[NT * (VT + 1)];
  
  int tid = threadIdx.x;//
  int bid = blockIdx.x;//

  __shared__ int blockRowStart;
  __shared__ int blockRowEnd;
  __shared__ int blockNumRows;

  if (tid == 0) {
    blockRowStart = d_startingIds[bid];
    blockRowEnd = d_startingIds[bid + 1];
    blockNumRows =  blockRowEnd - blockRowStart + 1;
  }
  __syncthreads();

  // if (tid == 0) {
  //   printf("blockRowStart=%d, blockRowEnd=%d, blockNumRows=%d\n", 
  //     blockRowStart, blockRowEnd, blockNumRows);
  // }
  // __syncthreads();

  //loadScanToShare(blockRowStart, blockRowEnd, d_scan, smem_starting_rows);
  
  threadStartIds<NT, VT>(blockRowStart, blockNumRows, 
            d_scan, smem_starting_rows);
  __syncthreads();
  
  //checkThreadStartIds<NT, VT>(smem_starting_rows);

  //acquire interval and rank.
  int interval[VT], rank[VT];
  allThreadsStartIds<NT, VT>(bid, tid, blockRowStart, blockNumRows, 
            bin_len, d_scan, smem_starting_rows, interval, rank);
  __syncthreads();

  // Load and distribute the gather and scatter indices.
  int gather[VT];
  twoLayerDeviceMemToMemLoopIndirect<NT>(blockNumRows, csc_col, 
                                    x_sparse_index + blockRowStart, 
                                    tid, intervals_shared);
  
  //checkIntervalsShared(blockRowStart, blockNumRows, intervals_shared);

  // Gather the data into register.
  int range_x = bid * NNZ_PER_BLOCK;
  int left_over = bin_len%NNZ_PER_BLOCK;
  int len = NNZ_PER_BLOCK;
  if (bid == gridDim.x - 1) 
    len = left_over ? left_over : NNZ_PER_BLOCK;

  computeGather<Type, NT, VT>(len, blockRowStart, intervals_shared, 
                              interval, rank, gather);
  
  // if (tid == 0) {
  //   printf("bid = %d, len = %d, range_x=%d.\n", blockIdx.x, len, range_x);
  // }
  // __syncthreads();

  Type data[VT];//from [] to register data[].
  int left = bin_len%NNZ_PER_THREAD;
  twoLayerDeviceGather<NT, VT, Type>(left, len, csc_val, gather, 
                                     tid, data);

  // for (int i = 0; i < VT; ++i) {
  //   printf("bid=%d, tid=%d, data[%d]=%d\n", bid, tid, i, data[i]);
  // }
  // __syncthreads();
  //int range_x = tid * NNZ_PER_THREAD + bid * NNZ_PER_THREAD * THREADS;
  twoLayerDeviceRegToGlobal<NT, VT, Type>(left, len, data, tid, 
                                          bin_row + range_x); 
#if 1
#endif
}

template<int THREADS>
__device__ __forceinline__ int loadScanToShare(
 		int blockRowStart, int blockNumRows, 
 		const int* __restrict d_scan, int* scan_shared) {
	
	for (int r = threadIdx.x; r <= blockNumRows; r += THREADS) {
		scan_shared[r] = d_scan[blockRowStart + r];
    //printf("bid=%d, tid=%d, scan_shared[%d]=%d\n", blockIdx.x, threadIdx.x, r, scan_shared[r]);
	}
 	__syncthreads();
 }

template<typename Type, int THREADS, int NNZ_PER_THREAD, int NNZ_PER_BLOCK>
__global__ void TwoLayerExtractKernel_shared(
    int bin_len, int* csc_col, const int* __restrict__ d_scan, 
    int x_nnz, Type* csc_val, int* x_sparse_index, 
    const int* d_startingIds, Type* bin_row) {

  const int NT = THREADS;
  const int VT = NNZ_PER_THREAD;

  __shared__ int smem_starting_rows[THREADS];
  //__shared__ int smem_indices[NT * (VT + 1)];
  __shared__ int intervals_shared[NT * (VT + 1)];
  __shared__ int scan_shared[NT * (VT + 1)];

  int tid = threadIdx.x;//
  int bid = blockIdx.x;//

  __shared__ int blockRowStart;
  __shared__ int blockRowEnd;
  __shared__ int blockNumRows;

  if (tid == 0) {
    blockRowStart = d_startingIds[bid];
    blockRowEnd = d_startingIds[bid + 1];
    blockNumRows =  blockRowEnd - blockRowStart + 1;
  }
  __syncthreads();

  loadScanToShare<NT>(blockRowStart, blockNumRows, d_scan, scan_shared);


  threadStartIds_shared<NT, VT>(blockRowStart, blockNumRows, 
            scan_shared, smem_starting_rows);
  __syncthreads();
  
  //checkThreadStartIds<NT, VT>(smem_starting_rows);
  

  //acquire interval and rank.
  int interval[VT], rank[VT];
  allThreadsStartIds_shared<NT, VT>(bid, tid, blockRowStart, blockNumRows, 
            bin_len, scan_shared, smem_starting_rows, interval, rank);
  __syncthreads();


  // Load and distribute the gather and scatter indices.
  int gather[VT];
  twoLayerDeviceMemToMemLoopIndirect<NT>(blockNumRows, csc_col, 
                                    x_sparse_index + blockRowStart, 
                                    tid, intervals_shared);
  
  //checkIntervalsShared(blockRowStart, blockNumRows, intervals_shared);

  // Gather the data into register.
  int block_begin_index = bid * NNZ_PER_BLOCK;
  int block_left_over = bin_len%NNZ_PER_BLOCK;
  int nnz_per_block = NNZ_PER_BLOCK;
  if (bid == gridDim.x - 1) 
    nnz_per_block = block_left_over ? block_left_over : NNZ_PER_BLOCK;

  computeGather<Type, NT, VT>(nnz_per_block, blockRowStart, intervals_shared, 
                              interval, rank, gather);

  // if (tid == 0) {
  //   printf("bid = %d, len = %d, range_x=%d.\n", blockIdx.x, len, range_x);
  // }
  // __syncthreads();

  Type data[VT];//from [] to register data[].
  //int left = bin_len%NNZ_PER_THREAD;
  int thread_left_over = bin_len%NNZ_PER_THREAD;
  twoLayerDeviceGather<NT, VT, Type>(thread_left_over, nnz_per_block, 
                                     csc_val, gather, 
                                     tid, data);

  // for (int i = 0; i < VT; ++i) {
  //   printf("bid=%d, tid=%d, data[%d]=%d\n", bid, tid, i, data[i]);
  // }
  // __syncthreads();
  //int range_x = tid * NNZ_PER_THREAD + bid * NNZ_PER_THREAD * THREADS;
  twoLayerDeviceRegToGlobal<NT, VT, Type>(thread_left_over, nnz_per_block, 
                                          data, tid, 
                                          bin_row + block_begin_index); 
#if 0
#endif
}

template<int NNZ_PER_BLOCK>
__global__ void DetermineBlockStarts(int x_nnz, 
                                     const int* __restrict__ d_scan, 
                                     int* startingIds) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id > x_nnz)
    return;

  int a = d_scan[id];
  int b = d_scan[min(id + 1, x_nnz)];

  int blocka = divup<int>(a, NNZ_PER_BLOCK);
  int blockb = (b - 1) / static_cast<int>(NNZ_PER_BLOCK);

  // printf("id =%d, a=%d, b=%d, blocka=%d, blockb=%d.\n", 
  //         id, a, b, blocka, blockb);
  //iterate over all blocks that start with that row
  if(a != b)
    for (; blocka <= blockb; ++blocka)
      startingIds[blocka] = id;

  //write last
  if (id == x_nnz)
    startingIds[divup<int>(b, NNZ_PER_BLOCK)] = id - 1;
}

__global__ void CheckBlockStarts(int numblocks, 
                                 int* startingIds) {
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i=0; i<=numblocks; i++) {
      printf("startingIds[%d] = %d\n", i, startingIds[i]);
    }
  }
}

template<typename Type, int NNZ_PER_THREAD, 
int THREADS, int NNZ_PER_BLOCK>
__host__  inline void TwoLayerLoadBalanceExtract(
          int bin_len, int* csc_col, 
          const int* __restrict__ d_scan, 
          int* d_startingIds,
          int x_nnz, Type* csc_val,
          int* x_sparse_index, Type* bin_row) {

  uint32_t requiredBlocks = divup<int>(bin_len, NNZ_PER_BLOCK);
  //TODO: 
  int tempmemsize = requiredBlocks + 2;

  int threads_per_block = THREADS;
  int blocks_per_grid = divup<int>(x_nnz + 1, THREADS);
  
  printf("bin_len = %d, x_nnz = %d.\n", bin_len, x_nnz);
  printf("nnz_per_thread = %d, threads_per_block=%d, blocks_per_grid=%d\n", NNZ_PER_THREAD, threads_per_block, blocks_per_grid);
  printf("tempmemsize = %d\n", tempmemsize);

  // int* d_startingIds = NULL;
  // cudaErrCheck(cudaMalloc((void **)&d_startingIds, tempmemsize * sizeof(int)));

  SpmspvTimer first_timer;
  first_timer.Start();
  DetermineBlockStarts<NNZ_PER_BLOCK><<<blocks_per_grid, threads_per_block>>>
                                     (x_nnz, d_scan, d_startingIds);
  
  double first_time = first_timer.Stop();
  std::cout << "blockPartition time = " << first_time << " ms." << std::endl;                                     
  //CheckBlockStarts<<<1, 1>>>(requiredBlocks, d_startingIds);
  SpmspvTimer second_timer;
  second_timer.Start();
  TwoLayerExtractKernel<Type, THREADS, NNZ_PER_THREAD, NNZ_PER_BLOCK>
                              <<<requiredBlocks, threads_per_block>>>
                              (bin_len, csc_col,
                              d_scan, x_nnz, csc_val, x_sparse_index,
                              d_startingIds, bin_row);
  double second_time = second_timer.Stop();
  std::cout << "inner time = " << second_time << " ms." << std::endl;
  //MGPU_SYNC_CHECK("TwoLayerExtractKernel");
  // if (d_startingIds)
  //    cudaErrCheck(cudaFree(d_startingIds));
}

template<typename iT, typename uiT, typename vT>
int TwoLayerLoadBalanceExtractDriver(
                            iT* d_csc_row, iT* d_csc_col, 
                            vT* d_csc_val,
                            iT m, iT n, iT nnz, 
                            iT x_nnz, 
                            int* d_sparse_x_inx,
                            vT* d_sparse_x_val,
                            int bin_len,
                            iT* d_bin_row, vT* d_bin_val,
                            int* d_ptr_col_len,
                            int* d_startingIds, 
                            int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;

  // TwoLayerLoadBalanceExtract<int, NNZ_PER_THREAD, THREADS, NNZ_PER_BLOCK>
  TwoLayerLoadBalanceExtract<int, 4, 512, 2048>
                         (bin_len, d_csc_col, d_ptr_col_len, 
                          d_startingIds, x_nnz,  
                          d_csc_row, d_sparse_x_inx, d_bin_row);
  //PrintBinRow<<<1, 1>>>(bin_len, d_bin_row);
  
  // SpmspvTimer second_timer;
  // second_timer.Start();
  // // TwoLayerLoadBalanceExtract<vT, NNZ_PER_THREAD, THREADS, NNZ_PER_BLOCK>
  //TwoLayerLoadBalanceExtract<vT, 4, 256, 1024>
  TwoLayerLoadBalanceExtract<vT, 4, 512, 2048>
                         (bin_len, d_csc_col, d_ptr_col_len, 
                          d_startingIds, x_nnz,  
                          d_csc_val, d_sparse_x_inx, d_bin_val);
  //PrintBinRow<<<1, 1>>>(bin_len, d_bin_val);
  
  // double second_time = second_timer.Stop();
  // std::cout << "second time = " << second_time << " ms." << std::endl;
  return err;
}
#endif 
