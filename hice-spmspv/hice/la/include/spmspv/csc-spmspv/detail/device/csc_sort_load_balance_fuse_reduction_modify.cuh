#ifndef SORT_LOAD_BALANCE_FUSE_REDUCTION_MODIFY_METHOD_H_
#define SORT_LOAD_BALANCE_FUSE_REDUCTION_MODIFY_METHOD_H_

#include "spmspv/config.h"

template<typename Type>
__global__ void KernelIntervalMoveIndirectFuseAndReductionModify(
            int bin_len, int* csc_col, int* d_scan, 
            int x_nnz, int* csc_row, Type* csc_val, 
            int* x_sparse_index, Type* x_sparse_val,
            const int* mp_global, 
            int* bin_row, Type* bin_val, 
            Type* d_y) {

  const int NT = 128;
  const int VT = 7;
  //nt=128, vt=7，128*8=1024, 1024*sizeof(int) = 4096 bytes
  __shared__ int indices_shared[NT * (VT + 1)];
  __shared__ Type x_val_shared[NT * (VT + 1)];

  int tid = threadIdx.x;//
  int block = blockIdx.x;//

  // Load balance the move IDs (my_counting_iterator) 
  // over the scan of the interval sizes.
  int4 range = my_CTALoadBalance<NT, VT>(bin_len, d_scan, 
    x_nnz, block, tid, mp_global, indices_shared, true);
  // The interval indices are in the left part of shared memory (moveCount).
  // The scan of interval counts are in the right part (intervalCount).
  bin_len = range.y - range.x;    //
  x_nnz = range.w - range.z;//
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
    gather[i] = intervals_shared2[interval[i]] + rank[i];
    reg_x_val[i] = x_val_shared2[interval[i]];
    //gather[i] = gather_global[intervals_shared2[interval[i]]] + rank[i];
    //printf( "%d %d: %d %d %d %d\n", tid, i, gather[i], intervals_shared2[interval[i]], interval[i], rank[i]);
  }
  __syncthreads();
#if 1
  // Gather the data into register.
  int data_row[VT];//from [] to register data_row[].
  // Added by 
  Type data_val[VT];//from [] to register data_val[].

  DeviceGatherFuseModify<NT, VT, Type>(bin_len, csc_row, 
                                       csc_val, reg_x_val, gather, 
                                       tid, data_row, data_val);
  
  DeviceRegToGlobalFuseAndReduction<NT, VT, Type>(bin_len, data_row, 
                                    data_val, tid, 
                                    /*bin_row + range.x, bin_val + range.x, */
                                    d_y); 
#endif
}

template<typename Type>
__global__ void KernelIntervalMoveIndirectFuseAndReductionModify_write_back(
            int bin_len, int* csc_col, int* d_scan, 
            int x_nnz, int* csc_row, Type* csc_val, 
            int* x_sparse_index, Type* x_sparse_val,
            const int* mp_global, 
            int* bin_row, Type* bin_val, 
            Type* d_y) {

  const int NT = 128;
  const int VT = 7;
  //nt=128, vt=7，128*8=1024, 1024*sizeof(int) = 4096 bytes
  __shared__ int indices_shared[NT * (VT + 1)];
  __shared__ Type x_val_shared[NT * (VT + 1)];

  int tid = threadIdx.x;//
  int block = blockIdx.x;//

  // Load balance the move IDs (my_counting_iterator) 
  // over the scan of the interval sizes.
  int4 range = my_CTALoadBalance<NT, VT>(bin_len, d_scan, 
    x_nnz, block, tid, mp_global, indices_shared, true);
  // The interval indices are in the left part of shared memory (moveCount).
  // The scan of interval counts are in the right part (intervalCount).
  bin_len = range.y - range.x;    //
  x_nnz = range.w - range.z;//
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
    gather[i] = intervals_shared2[interval[i]] + rank[i];
    reg_x_val[i] = x_val_shared2[interval[i]];
    //gather[i] = gather_global[intervals_shared2[interval[i]]] + rank[i];
    //printf( "%d %d: %d %d %d %d\n", tid, i, gather[i], intervals_shared2[interval[i]], interval[i], rank[i]);
  }
  __syncthreads();
#if 1
  // Gather the data into register.
  int data_row[VT];//from [] to register data_row[].
  // Added by 
  Type data_val[VT];//from [] to register data_val[].

  DeviceGatherFuseModify<NT, VT, Type>(bin_len, csc_row, 
                                       csc_val, reg_x_val, gather, 
                                       tid, data_row, data_val);
  
  DeviceRegToGlobalFuseAndReductionWriteBack<NT, VT, Type>(bin_len, data_row, 
                                    data_val, tid, 
                                    bin_row + range.x, bin_val + range.x, 
                                    d_y); 
#endif
}

void my_preMallocMergePathPartitions(my_counting_iterator<int> array, 
                                    int bin_len, int* d_scan, int x_nnz, 
                                    int* partitionsDevice, int nv) {

  const int NT = 64;
  ////numofBlocks
  int numPartitions = SPMSPV_DIV_UP(bin_len + x_nnz, nv);
  ////(numPartitions + 1)/64
  int numPartitionBlocks = SPMSPV_DIV_UP(numPartitions + 1, NT);//
  //TODO: 

  // SpmspvTimer step1_timer;
  // step1_timer.Start();
  //MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);
  //int* partitionsDevice = NULL;
  //cudaErrCheck(cudaMalloc((void** )&partitionsDevice, 
  //                        numPartitions + 1 * sizeof(int)));

  //double step1_time = step1_timer.Stop();
  //std::cout << "malloc partition time = " << step1_time << " ms." << std::endl;
  //numPartitions的个数 <= (all_nnz)/(127*8). 
  //printf("numPartitions=%d, numPartitionBlocks=%d\n", numPartitions, numPartitionBlocks);

  my_KernelMergePartition<NT><<<numPartitionBlocks, NT>>>(array, bin_len,
    d_scan, x_nnz, nv, partitionsDevice, numPartitions + 1);
  MGPU_SYNC_CHECK("KernelMergePartition");

  //return partitionsDevice;
}


template<typename Type>
__host__  inline void IntervalGatherIndirectFuseAndReductionModify(
                      int bin_len, int* csc_col, 
                      int* d_scan, int x_nnz, 
                      int* csc_row, Type* csc_val,
                      int* x_sparse_index, Type* x_sparse_val, 
                      int* bin_row, Type* bin_val, bool isWriteBack,
                      /*int* partitionsDevice,*/
                      Type* d_y) {
  const int NT = 128;
  const int VT = 7;
  //NV = NT * VT = 128 * 7 = 896
  int NV = NT * VT;
  int numBlocks = SPMSPV_DIV_UP(bin_len + x_nnz, NV);

  int* partitionsDevice = my_MergePathPartitions(
    my_counting_iterator<int>(0), bin_len, d_scan,
    x_nnz, NV);

  if(isWriteBack){
    KernelIntervalMoveIndirectFuseAndReductionModify_write_back<Type><<<numBlocks, NT>>>
                                      (bin_len, csc_col,
                                      d_scan, x_nnz, csc_row, csc_val, 
                                      x_sparse_index, x_sparse_val,
                                      partitionsDevice, 
                                      bin_row, bin_val, 
                                      d_y);
  }else{
     KernelIntervalMoveIndirectFuseAndReductionModify<Type><<<numBlocks, NT>>>
                                      (bin_len, csc_col,
                                      d_scan, x_nnz, csc_row, csc_val, 
                                      x_sparse_index, x_sparse_val,
                                      partitionsDevice, 
                                      bin_row, bin_val, 
                                      d_y);
  }
 
  MGPU_SYNC_CHECK("KernelIntervalMoveIndirectFuseAndReductionModify");
}

template<typename iT, typename uiT, typename vT>
int LoadBalanceFuseExtractAndReductionModifyDriver(
                            iT* d_csc_row, iT* d_csc_col, 
                            vT* d_csc_val,
                            iT m, iT n, iT nnz, 
                            iT x_nnz, 
                            int* d_sparse_x_inx,
                            vT* d_sparse_x_val,
                            int bin_len,
                            iT* d_bin_row, vT* d_bin_val, bool isWriteBack,
                            vT* d_y,
                            int* d_ptr_col_len, 
                            int* partitionsDevice,
                            int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;
  // printf("call fuse&reduction&modify version.\n");
  // SpmspvTimer first_timer;
  // first_timer.Start();
  IntervalGatherIndirectFuseAndReductionModify<vT>(bin_len, d_csc_col, 
                               d_ptr_col_len, x_nnz,  
                               d_csc_row, d_csc_val, 
                               d_sparse_x_inx, d_sparse_x_val, 
                               d_bin_row, d_bin_val, isWriteBack,
                               /*partitionsDevice,*/
                               d_y);
  // double first_time = first_timer.Stop();
  // std::cout << "fuse-modify extract time = " << first_time << " ms." 
  //           << std::endl;
  return err;
}
#endif  //sort_fuse_modify.cuh
