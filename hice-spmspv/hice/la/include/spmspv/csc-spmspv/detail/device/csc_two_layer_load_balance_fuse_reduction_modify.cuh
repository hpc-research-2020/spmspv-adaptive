#ifndef SORT_TWO_LAYER_LOAD_BALANCE_FUSE_REDUCTION_MODIFY_METHOD_H_
#define SORT_TWO_LAYER_LOAD_BALANCE_FUSE_REDUCTION_MODIFY_METHOD_H_

#include "spmspv/config.h"

template<int NT, int VT, typename Type>
__device__ inline void DeviceGatherFuseModify(int bin_len, 
                                    int* csc_row, Type* csc_val, 
                                    Type reg_x_val[VT],
                                    int indices[VT], 
                                    int tid, int* reg_row, Type* reg_val) {
  if (bin_len >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      reg_row[i] = csc_row[indices[i]];
      reg_val[i] = csc_val[indices[i]] * reg_x_val[i];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      int index = NT * i + tid;
      if (index < bin_len) {
        reg_row[i] = csc_row[indices[i]];
        reg_val[i] = csc_val[indices[i]] * reg_x_val[i];
      }  
    }
  }
  //__syncthreads();
}

template<typename Type, int THREADS, int NNZ_PER_THREAD, int NNZ_PER_BLOCK>
__global__ void TwoLayerExtractFuseAndReductionKernel_shared_modify_write_back(
            int bin_len, int* csc_col, const int* __restrict__ d_scan, 
            int x_nnz, int* csc_row, Type* csc_val, 
            int* x_sparse_index, Type* x_sparse_val,
            const int* d_startingIds, 
            int* d_bin_row, Type* d_bin_val, 
            Type* d_y) {

  const int NT = THREADS;
  const int VT = NNZ_PER_THREAD;
  //nt=128, vt=7，1024 bytes
  __shared__ int smem_starting_rows[THREADS];
  //512*4*2=4096bytes.
  //512*8=4096bytes.
  __shared__ int intervals_shared[NT * (VT + 1)];
  __shared__ int scan_shared[NT * (VT + 1)];
  __shared__ Type x_val_shared[NT * (VT + 1)];

  int tid = threadIdx.x;//
  int bid = blockIdx.x;//

  int block_begin_index = bid * NNZ_PER_BLOCK;
  int block_left_over = bin_len%NNZ_PER_BLOCK;
  int nnz_per_block = NNZ_PER_BLOCK;
  if (bid == gridDim.x - 1) 
    nnz_per_block = block_left_over ? block_left_over : NNZ_PER_BLOCK;

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

  loadScanToShare<NT>(blockRowStart, blockNumRows, d_scan, scan_shared);

  threadStartIds_shared<NT, VT>(blockRowStart, blockNumRows, 
            scan_shared, smem_starting_rows);
  __syncthreads();
  
  //checkThreadStartIds<NT, VT>(smem_starting_rows);

  //acquire interval and rank.
  int interval[VT], rank[VT];
  allThreadsStartIds_shared<NT, VT>(bid, tid, blockRowStart, blockNumRows, 
            bin_len, scan_shared, smem_starting_rows, intervals_shared);
  __syncthreads();

 
  int* scan_shared2 = scan_shared - blockRowStart; 
  //int current_val = tid * NNZ_PER_THREAD + bid * THREADS * NNZ_PER_THREAD;
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid; 
    int gid = block_begin_index + index;
    interval[i] = blockRowStart;   
    if (index < nnz_per_block) {   
      interval[i] = intervals_shared[index];
      rank[i] = gid - scan_shared2[interval[i]];
      //rank[i] = gid - intervals_shared - range.z + interval[i];
    }
  }
  __syncthreads();



  // Load and distribute the gather and scatter indices.
  int gather[VT];
  Type reg_x_val[VT];
  // twoLayerDeviceMemToMemLoopIndirect<NT>(blockNumRows, csc_col, 
  //                                   x_sparse_index + blockRowStart, 
  //                                   tid, intervals_shared);
  twoLayerDeviceMemToMemLoopIndirect<NT>(blockNumRows, csc_col, 
                                        x_sparse_index + blockRowStart, 
                                        tid, scan_shared);
 
  //NOTE: interval_shared[] stores x_sparse_val.
  //Type* x_val_shared = (Type*)interval_shared;
  DeviceMemToMemLoopIndirectVal<NT, Type>(blockNumRows, csc_col, 
                                          x_sparse_val + blockRowStart, 
                                          tid, x_val_shared);
  //checkIntervalsShared(blockRowStart, blockNumRows, intervals_shared);
 #if 1
  Type* x_val_shared2 = x_val_shared - blockRowStart;
  #pragma unroll
  for (int i = 0; i < VT; ++i)
  {
    //TODO: 
    gather[i] = scan_shared2[interval[i]] + rank[i];
    reg_x_val[i] = x_val_shared2[interval[i]];
    //gather[i] = gather_global[intervals_shared2[interval[i]]] + rank[i];
    //printf( "%d %d: %d %d %d %d\n", tid, i, gather[i], intervals_shared2[interval[i]], interval[i], rank[i]);
  }
  __syncthreads();

  // Gather the data into register.
  int data_row[VT];//from [] to register data_row[].
  Type data_val[VT];//from [] to register data_val[].
 
  DeviceGatherFuseModify<NT, VT, Type>(nnz_per_block, csc_row, 
                                       csc_val, reg_x_val, gather, 
                                       tid, data_row, data_val);


  TwoLayerDeviceRegToGlobalFuseAndReductionWriteBack<NT, VT, Type>
  								    (nnz_per_block, data_row, 
                      data_val, tid, d_bin_row, d_bin_val,
  								    d_y); 

#endif
}

template<typename Type, int THREADS, int NNZ_PER_THREAD, int NNZ_PER_BLOCK>
__global__ void TwoLayerExtractFuseAndReductionKernel_shared_modify(
            int bin_len, int* csc_col, const int* __restrict__ d_scan, 
            int x_nnz, int* csc_row, Type* csc_val, 
            int* x_sparse_index, Type* x_sparse_val,
            const int* d_startingIds, 
            int* d_bin_row, Type* d_bin_val, 
            Type* d_y) {

  const int NT = THREADS;
  const int VT = NNZ_PER_THREAD;
  //nt=128, vt=7，1024 bytes
  __shared__ int smem_starting_rows[THREADS];
  //512*4*2=4096bytes.
  //512*8=4096bytes.
  __shared__ int intervals_shared[NT * (VT + 1)];
  __shared__ int scan_shared[NT * (VT + 1)];
  __shared__ Type x_val_shared[NT * (VT + 1)];

  int tid = threadIdx.x;//
  int bid = blockIdx.x;//

  int block_begin_index = bid * NNZ_PER_BLOCK;
  int block_left_over = bin_len%NNZ_PER_BLOCK;
  int nnz_per_block = NNZ_PER_BLOCK;
  if (bid == gridDim.x - 1) 
    nnz_per_block = block_left_over ? block_left_over : NNZ_PER_BLOCK;

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

  loadScanToShare<NT>(blockRowStart, blockNumRows, d_scan, scan_shared);

  threadStartIds_shared<NT, VT>(blockRowStart, blockNumRows, 
            scan_shared, smem_starting_rows);
  __syncthreads();
  
  //checkThreadStartIds<NT, VT>(smem_starting_rows);

  //acquire interval and rank.
  int interval[VT], rank[VT];
  allThreadsStartIds_shared<NT, VT>(bid, tid, blockRowStart, blockNumRows, 
            bin_len, scan_shared, smem_starting_rows, intervals_shared);
  __syncthreads();

 
  int* scan_shared2 = scan_shared - blockRowStart; 
  //int current_val = tid * NNZ_PER_THREAD + bid * THREADS * NNZ_PER_THREAD;
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid; 
    int gid = block_begin_index + index;
    interval[i] = blockRowStart;   
    if (index < nnz_per_block) {   
      interval[i] = intervals_shared[index];
      rank[i] = gid - scan_shared2[interval[i]];
      //rank[i] = gid - intervals_shared - range.z + interval[i];
    }
  }
  __syncthreads();



  // Load and distribute the gather and scatter indices.
  int gather[VT];
  Type reg_x_val[VT];
  // twoLayerDeviceMemToMemLoopIndirect<NT>(blockNumRows, csc_col, 
  //                                   x_sparse_index + blockRowStart, 
  //                                   tid, intervals_shared);
  twoLayerDeviceMemToMemLoopIndirect<NT>(blockNumRows, csc_col, 
                                        x_sparse_index + blockRowStart, 
                                        tid, scan_shared);
 
  //NOTE: interval_shared[] stores x_sparse_val.
  //Type* x_val_shared = (Type*)interval_shared;
  DeviceMemToMemLoopIndirectVal<NT, Type>(blockNumRows, csc_col, 
                                          x_sparse_val + blockRowStart, 
                                          tid, x_val_shared);
  //checkIntervalsShared(blockRowStart, blockNumRows, intervals_shared);
 #if 1
  Type* x_val_shared2 = x_val_shared - blockRowStart;
  #pragma unroll
  for (int i = 0; i < VT; ++i)
  {
    //TODO: 
    gather[i] = scan_shared2[interval[i]] + rank[i];
    reg_x_val[i] = x_val_shared2[interval[i]];
    //gather[i] = gather_global[intervals_shared2[interval[i]]] + rank[i];
    //printf( "%d %d: %d %d %d %d\n", tid, i, gather[i], intervals_shared2[interval[i]], interval[i], rank[i]);
  }
  __syncthreads();

  // Gather the data into register.
  int data_row[VT];//from [] to register data_row[].
  Type data_val[VT];//from [] to register data_val[].
 
  DeviceGatherFuseModify<NT, VT, Type>(nnz_per_block, csc_row, 
                                       csc_val, reg_x_val, gather, 
                                       tid, data_row, data_val);


  TwoLayerDeviceRegToGlobalFuseAndReduction<NT, VT, Type>
  								    (nnz_per_block, data_row, 
                      data_val, tid, d_bin_row, d_bin_val,
  								    d_y); 

#endif
}

template<typename Type, int NNZ_PER_THREAD, 
int THREADS, int NNZ_PER_BLOCK>
__host__  inline void TwoLayerIntervalGatherIndirectFuseAndReductionModify(
                      int bin_len, int* csc_col, 
                      const int* __restrict__ d_scan,  
                      int* d_startingIds, bool isMalloced,
                      int x_nnz, 
                      int* csc_row, Type* csc_val,
                      int* x_sparse_index, Type* x_sparse_val,
                      int* d_bin_row, Type* d_bin_val, bool isWriteBack,
                      Type* d_y) {
  uint32_t requiredBlocks = divup<int>(bin_len, NNZ_PER_BLOCK);
  //TODO: 
  int tempmemsize = requiredBlocks + 2;

  int threads_per_block = THREADS;
  int blocks_per_grid = divup<int>(x_nnz + 1, THREADS);
  
  // printf("bin_len = %d, x_nnz = %d.\n", bin_len, x_nnz);
  // printf("nnz_per_thread = %d, threads_per_block=%d, blocks_per_grid=%d\n", NNZ_PER_THREAD, threads_per_block, blocks_per_grid);
  // printf("tempmemsize = %d\n", tempmemsize);

  // int* d_startingIds = NULL;
  if(!isMalloced)
    cudaErrCheck(cudaMalloc((void **)&d_startingIds, tempmemsize * sizeof(int)));

  //SpmspvTimer first_timer;
  //first_timer.Start();
  DetermineBlockStarts<NNZ_PER_BLOCK><<<blocks_per_grid, threads_per_block>>>
                                     (x_nnz, d_scan, d_startingIds);
  
  //double first_time = first_timer.Stop();
  //std::cout << "blockPartition time = " << first_time << " ms." << std::endl;
  
  if(isWriteBack){
     TwoLayerExtractFuseAndReductionKernel_shared_modify_write_back<Type, THREADS, NNZ_PER_THREAD, NNZ_PER_BLOCK> 
                              <<<requiredBlocks, threads_per_block>>>
                              (bin_len, csc_col,
                              d_scan, x_nnz, csc_row, csc_val, 
                              x_sparse_index, x_sparse_val,
                              d_startingIds, 
                              d_bin_row, d_bin_val,
                              d_y);
  }else{
    TwoLayerExtractFuseAndReductionKernel_shared_modify<Type, THREADS, NNZ_PER_THREAD, NNZ_PER_BLOCK> 
                              <<<requiredBlocks, threads_per_block>>>
                              (bin_len, csc_col,
                              d_scan, x_nnz, csc_row, csc_val, 
                              x_sparse_index, x_sparse_val,
                              d_startingIds, 
                              d_bin_row, d_bin_val,
                              d_y);
  }
 
  // KernelIntervalMoveIndirectFuseAndReduction<Type><<<numBlocks, NT>>>
  //   (bin_len, csc_col,
  //   d_scan, x_nnz, csc_row, csc_val, 
  //   x_sparse_index, x_sparse_val,
  //   partitionsDevice, 
  //   /*bin_row, bin_val, */
  //   d_y);
  //MGPU_SYNC_CHECK("TwoLayerExtractFuseAndReductionKernel_shared");
  if(!isMalloced){
    if(d_startingIds)
      cudaErrCheck(cudaFree(d_startingIds));
  }

}

template<typename iT, typename uiT, typename vT>
int TwoLayerLoadBalanceFuseExtractAndReductionModifyDriver(
                            iT* d_csc_row, iT* d_csc_col, 
                            vT* d_csc_val,
                            iT m, iT n, iT nnz, 
                            iT x_nnz, 
                            int* d_sparse_x_inx,
                            vT* d_sparse_x_val,
                            int bin_len,
                            int* d_bin_row, vT* d_bin_val, bool isWriteBack,
                            vT* d_y,
                            int* d_ptr_col_len, 
                            int* d_startingIds, bool isMalloced, 
                            int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;
  //printf("call fuse&reduction&modify version.\n");
  //SpmspvTimer first_timer;
  //first_timer.Start();
  //4, 256, 1024>
  //<4, 128, 512>: best
  TwoLayerIntervalGatherIndirectFuseAndReductionModify<vT, 4, 128, 512>
  							               (bin_len, d_csc_col, 
                               d_ptr_col_len, d_startingIds, isMalloced,
                               x_nnz,  
                               d_csc_row, d_csc_val, 
                               d_sparse_x_inx, d_sparse_x_val,
                               d_bin_row, d_bin_val, isWriteBack,
                               d_y);
//  TwoLayerIntervalGatherIndirectFuseAndReductionModify<vT, 7, 128, 896>
//   							               (bin_len, d_csc_col, 
//                                d_ptr_col_len, d_startingIds, isMalloced,
//                                x_nnz,  
//                                d_csc_row, d_csc_val, 
//                                d_sparse_x_inx, d_sparse_x_val,
//                                d_bin_row, d_bin_val, isWriteBack,
//                                d_y);                               
  //  TwoLayerIntervalGatherIndirectFuseAndReductionModify<vT, 1, 1, 1>
  // 							               (bin_len, d_csc_col, 
  //                              d_ptr_col_len, d_startingIds, isMalloced,
  //                              x_nnz,  
  //                              d_csc_row, d_csc_val, 
  //                              d_sparse_x_inx, d_sparse_x_val,
  //                              d_bin_row, d_bin_val, isWriteBack,
  //                              d_y);                               
  //PrintBinRow<<<1, 1>>>(bin_len, d_bin_val);
  //double first_time = first_timer.Stop();
  //std::cout << "new fuse extract time = " << first_time << " ms." << std::endl;
  return err;
}
#endif //TWO_LAYER_LOAD_BALANCE_FUSE_REDUCTION_MODIFY_H 