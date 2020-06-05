#ifndef SORT_LOAD_BALANCE_MODIFY_METHOD_H_
#define SORT_LOAD_BALANCE_MODIFY_METHOD_H_
#include "spmspv/config.h"

template<int NT, int VT, typename Type>
__device__ inline void DeviceMemToMem4IndirectVal(int x_nnz, int* csc_col, 
                                Type* x_sparse_val, int tid, Type* dest) {
  Type x[VT];
  //即，VT最大就是4.
  const int Count = (VT < 4) ? VT : 4;
  if(x_nnz >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < Count; ++i)
      x[i] = x_sparse_val[NT * i + tid];
    #pragma unroll
    for(int i = 0; i < Count; ++i)
      dest[NT * i + tid] = x[i];
  } else {
    // deal with left.
    #pragma unroll
    for (int i = 0; i < Count; ++i) {
      int index = NT * i + tid;
      if(index < x_nnz)
        //x[i] = csc_col[x_sparse_index[NT * i + tid]];
        x[i] = x_sparse_val[index];
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

template<int NT, typename Type>
__device__ inline void DeviceMemToMemLoopIndirectVal(int x_nnz, int* csc_col, 
                                    Type* x_sparse_val, int tid, Type* dest) {
  for(int i = 0; i < x_nnz; i += 4 * NT)
    DeviceMemToMem4IndirectVal<NT, 4, Type>(x_nnz - i, csc_col, 
                                            x_sparse_val + i, 
                                            tid, dest + i);
  __syncthreads();
}

template<int NT, int VT, typename Type>
__device__ inline void DeviceGatherModify(int bin_len, Type* csc_val, 
                                          int indices[VT], 
                                          Type reg_x_val[VT],
                                          int tid, Type* reg) {
  if (bin_len >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      reg[i] = csc_val[indices[i]] * reg_x_val[i];
      // if(blockIdx.x ==0)
      //   printf("debug: tid(%d), i(%d): res(%lf), x_val(%lf), gather(%d).\n", 
      //        tid, i, reg[i], reg_x_val[i], indices[i]);
    }
  } else {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      int index = NT * i + tid;
      if (index < bin_len) {
        reg[i] = csc_val[indices[i]] * reg_x_val[i];
        // if(blockIdx.x == 0)
        //   printf("left debug: tid(%d), i(%d): res(%lf), x_val(%lf), gather(%d).\n", tid, i, reg[i], reg_x_val[i], indices[i]);
      }
    }
  }
  //__syncthreads();
}

template<typename Type>
__global__ void KernelIntervalMoveIndirectModify(
          int bin_len, int* csc_col, int* d_scan, 
          int x_nnz, Type* csc_val, 
          int* x_sparse_index, Type* x_sparse_val,
          const int* mp_global, Type* bin_val) {
  const int NT = 128;
  const int VT = 7;
  //nt=128, vt=7，128*8=1024bytes
  __shared__ int indices_shared[NT * (VT + 1)];
  __shared__ Type x_val_shared[NT * (VT + 1)];

  int tid = threadIdx.x;//
  int block = blockIdx.x;//

  // Load balance the move IDs (counting_iterator) 
  // over the scan of the interval sizes.
  int4 range = my_CTALoadBalance<NT, VT>(bin_len, d_scan, 
    x_nnz, block, tid, mp_global, indices_shared, true);
  // The interval indices are in the left part of shared memory (moveCount).
  // The scan of interval counts are in the right part (intervalCount).
  bin_len = range.y - range.x;    //
  x_nnz = range.w - range.z;//
  int* move_shared = indices_shared;
  //intervals_shared存储的是d_scan的值.
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
  Type reg_x_val[VT];
  // Load the gather pointers into intervals_shared. 
  // DeviceMemToMemLoop<NT>(intervalCount, scatter_global + range.z, tid, 
  //    intervals_shared);
  my_DeviceMemToMemLoopIndirect<NT>(x_nnz, csc_col, x_sparse_index + range.z, 
                                  tid, intervals_shared);

  DeviceMemToMemLoopIndirectVal<NT, Type>(x_nnz, csc_col, 
                                          x_sparse_val + range.z, 
                                          tid, x_val_shared);
  // Make a second pass through shared memory. Grab the start indices of
  // the interval for each item and add the scan into it for the gather
  // index.
  Type* x_val_shared2 = x_val_shared - range.z;// range.z => b0
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    // from share-mem to registers.
    gather[i] = intervals_shared2[interval[i]] + rank[i];
    reg_x_val[i] = x_val_shared2[interval[i]];
    // reg_x_val[i] = x_sparse_index[interval[i]];
    // if (blockIdx.x == 0)
    //   printf("DEBUG: tid = %d, reg_x_val[%d] = %lf\n", tid, i, reg_x_val[i]);
    //gather[i] = gather_global[intervals_shared2[interval[i]]] + rank[i];
    //printf( "%d %d: %d %d %d %d\n", tid, i, gather[i], intervals_shared2[interval[i]], interval[i], rank[i]);
  }
  __syncthreads();
  // Gather the data into register.
  Type data[VT];//from [] to register data[].
  DeviceGatherModify<NT, VT, Type>(bin_len, csc_val, gather, reg_x_val, 
                                   tid, data);
  // for ( int i=0; i<VT; i++ )
  // {
  //   printf("after debug: %d, %d: %lf, %lf, %d.\n", tid, i, 
  //         data[i], reg_x_val[i], gather[i]);
  // }
  // Scatter the register data into global: from data[] to bin_row[]
  my_DeviceRegToGlobal<NT, VT, Type>(bin_len, data, tid, bin_val + range.x); 
}

template<typename Type>
__host__  inline void IntervalGatherIndirectModify(
                  int bin_len, int* csc_col, 
                  int* d_scan, int x_nnz, Type* csc_val,
                  int* x_sparse_index, Type* x_sparse_val,
                  Type* bin_val) {

  const int NT = 128;
  const int VT = 7;
  
  //NV = NT * VT = 128 * 7 = 896
  int NV = NT * VT;
 
  int numBlocks = SPMSPV_DIV_UP(bin_len + x_nnz, NV);
  
  printf("bin_len = %d, x_nnz = %d.\n", bin_len, x_nnz);
  printf("NT = %d, VT=%d, NV=%d, numBlocks=%d\n", NT, VT, NV, numBlocks);
  
 
  int* partitionsDevice = my_MergePathPartitions(
    my_counting_iterator<int>(0), bin_len, d_scan,
    x_nnz, NV);

  
  KernelIntervalMoveIndirectModify<Type><<<numBlocks, NT>>>
    (bin_len, csc_col,
    d_scan, x_nnz, csc_val, 
    x_sparse_index, x_sparse_val,
    partitionsDevice, bin_val);
  MGPU_SYNC_CHECK("KernelIntervalMoveIndirectModify");
}

template<typename iT, typename uiT, typename vT>
int LoadBalanceModifyExtractDriver(
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
  // extract csc_row.
  my_IntervalGatherIndirect(bin_len, d_csc_col, d_ptr_col_len, x_nnz,  
                         d_csc_row, d_sparse_x_inx,
                         d_bin_row);
  // extract csc_val.
  IntervalGatherIndirectModify<vT>(bin_len, d_csc_col, d_ptr_col_len, x_nnz,  
                         d_csc_val, d_sparse_x_inx, d_sparse_x_val, 
                         d_bin_val);
  return err;
}
#endif 
