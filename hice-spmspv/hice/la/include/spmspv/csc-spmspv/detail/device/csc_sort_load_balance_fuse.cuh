#ifndef SORT_LOAD_BALANCE_FUSE_METHOD_H_
#define SORT_LOAD_BALANCE_FUSE_METHOD_H_

#include "spmspv/config.h"

template<int NT, int VT, typename Type>
__device__ inline void DeviceGatherFuse(int bin_len, 
                                    int* csc_row, Type* csc_val, 
                                    int indices[VT], 
                                    int tid, int* reg_row, Type* reg_val) {
  if (bin_len >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      reg_row[i] = csc_row[indices[i]];
      reg_val[i] = csc_val[indices[i]];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      int index = NT * i + tid;
      if (index < bin_len) {
        reg_row[i] = csc_row[indices[i]];
        reg_val[i] = csc_val[indices[i]];
      }  
    }
  }
  //__syncthreads();
}

template<int NT, int VT, typename Type>
__device__ inline void DeviceRegToGlobalFuse(int bin_len, const int* reg_row,
                                             const Type* reg_val, int tid, 
                                             int* bin_row, Type* bin_val) {
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    if (index < bin_len) {
      bin_row[index] = reg_row[i];
      bin_val[index] = reg_val[i];
    }
  }
  //__syncthreads();
}

template<typename Type>
__global__ void KernelIntervalMoveIndirectFuse(
            int bin_len, int* csc_col, int* d_scan, 
            int x_nnz, int* csc_row, Type* csc_val, 
            int* x_sparse_index, Type* x_sparse_val,
            const int* mp_global, 
            int* bin_row, Type* bin_val) {

  const int NT = 128;
  const int VT = 7;
  //nt=128, vt=7，128*8=1024, 1024*sizeof(int) = 4096 bytes
  __shared__ int indices_shared[NT * (VT + 1)];
  int tid = threadIdx.x;//
  int block = blockIdx.x;//

  // Load balance the move IDs (counting_iterator) 
  // over the scan of the interval sizes.
  
  int4 range = my_CTALoadBalance<NT, VT>(bin_len, d_scan, 
    x_nnz, block, tid, mp_global, indices_shared, true);
  // The interval indices are in the left part of shared memory (moveCount).
  // The scan of interval counts are in the right part (intervalCount).
  bin_len = range.y - range.x;   
  x_nnz = range.w - range.z;
  int* move_shared = indices_shared;
  int* intervals_shared = indices_shared + bin_len;
  
  int* intervals_shared2 = intervals_shared - range.z;

  // Read out the interval indices and scan offsets.
 
  int interval[VT], rank[VT];
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
   
    int index = NT * i + tid; 
   
    int gid = range.x + index;
    
    interval[i] = range.z;   
   
    if (index < bin_len) {   
      interval[i] = move_shared[index];
      rank[i] = gid - intervals_shared2[interval[i]];
    }
  }
  __syncthreads();

  // Load and distribute the gather and scatter indices.
  int gather[VT];
  // Load the gather pointers into intervals_shared. 
  // DeviceMemToMemLoop<NT>(intervalCount, scatter_global + range.z, tid, 
  //    intervals_shared);
  my_DeviceMemToMemLoopIndirect<NT>(x_nnz, csc_col, x_sparse_index + range.z, 
                                 tid, intervals_shared);
  // Make a second pass through shared memory. Grab the start indices of
  // the interval for each item and add the scan into it for the gather
  // index.
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    gather[i] = intervals_shared2[interval[i]] + rank[i];
    //gather[i] = gather_global[intervals_shared2[interval[i]]] + rank[i];
    //printf( "%d %d: %d %d %d %d\n", tid, i, gather[i], intervals_shared2[interval[i]], interval[i], rank[i]);
  }
  __syncthreads();
#if 1
  // Gather the data into register.
  int data_row[VT];//from [] to register data_row[].
  // Added by 
  Type data_val[VT];//from [] to register data_val[].
  //DeviceGather<NT, VT, int>(bin_len, csc_row, gather, tid, data_row);
  //DeviceGather<NT, VT, Type>(bin_len, csc_val, gather, tid, data_val);
  DeviceGatherFuse<NT, VT, Type>(bin_len, csc_row, csc_val, 
                                 gather, tid, data_row, data_val);
  // for ( int i=0; i<VT; i++ )
  // {
  //   //printf("%d %d: %d %d\n", tid, i, data[i], gather[i]);
  // }
  // Scatter the register data into global: from data[] to bin_row[]
  //DeviceRegToGlobal<NT, VT, int>(bin_len, data_row, tid, bin_row + range.x); 
  //DeviceRegToGlobal<NT, VT, Type>(bin_len, data_val, tid, bin_val + range.x); 
  DeviceRegToGlobalFuse<NT, VT, Type>(bin_len, data_row, data_val, tid, 
                                     bin_row + range.x, bin_val + range.x); 
#endif
}

template<typename Type>
__host__  inline void IntervalGatherIndirectFuse(
                      int bin_len, int* csc_col, 
                      int* d_scan, int x_nnz, 
                      int* csc_row, Type* csc_val,
                      int* x_sparse_index, Type* x_sparse_val, 
                      int* bin_row, Type* bin_val) {
  const int NT = 128;
  const int VT = 7;
  //NV = NT * VT = 128 * 7 = 896
  int NV = NT * VT;
  int numBlocks = SPMSPV_DIV_UP(bin_len + x_nnz, NV);
  
  printf("bin_len = %d, x_nnz = %d.\n", bin_len, x_nnz);
  printf("NT = %d, VT=%d, NV=%d, numBlocks=%d\n", NT, VT, NV, numBlocks);
  
  // MGPU_MEM(int) partitionsDevice = MergePathPartitions(
  //  mgpu::counting_iterator<int>(0), bin_len, d_scan,
  //  x_nnz, NV, 0, mgpu::less<int>(), context);
  int* partitionsDevice = my_MergePathPartitions(
    my_counting_iterator<int>(0), bin_len, d_scan,
    x_nnz, NV);

  KernelIntervalMoveIndirectFuse<Type><<<numBlocks, NT>>>
    (bin_len, csc_col,
    d_scan, x_nnz, csc_row, csc_val, 
    x_sparse_index, x_sparse_val,
    partitionsDevice, bin_row, bin_val);
  MGPU_SYNC_CHECK("KernelIntervalMoveIndirectFuse");
}

template<typename iT, typename uiT, typename vT>
int LoadBalanceFuseExtractDriver(
                            iT* d_csc_row, iT* d_csc_col, 
                            vT* d_csc_val,
                            iT m, iT n, iT nnz, 
                            iT x_nnz, 
                            int* d_sparse_x_inx,
                            vT* d_sparse_x_val,
                            int bin_len,
                            iT* d_bin_row, vT* d_bin_val,
                            int* d_ptr_col_len, 
                            int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;
  printf("call fuse version.\n");
  SpmspvTimer first_timer;
  first_timer.Start();
  IntervalGatherIndirectFuse<vT>(bin_len, d_csc_col, d_ptr_col_len, x_nnz,  
                               d_csc_row, d_csc_val, 
                               d_sparse_x_inx, d_sparse_x_val, 
                               d_bin_row, d_bin_val);
  double first_time = first_timer.Stop();
  std::cout << "fuse extract time = " << first_time << " ms." << std::endl;
  return err;
}
#endif 
