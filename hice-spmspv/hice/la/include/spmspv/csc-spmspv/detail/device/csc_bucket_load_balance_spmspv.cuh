// This file provides the csc-based bucket (group based load-balance) spmspv method.

#ifndef LOAD_BALANCE_H_
#define LOAD_BALANCE_H_

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "../config.h"
#include "print_util.h"

#define TIMING 
#define PRINT_TEMP_RES

template<typename iT, typename uiT, typename vT>
__global__
void LoadBalanceCountElemsPerColKernel(const iT* d_csc_col,  
                        							const iT x_nnz, 
                        							/*SparseVec* d_sparse_x,*/
                                      iT* d_sparse_x_key,
                        							int* d_ptr_col_len) {
 
}


template<typename iT, typename uiT, typename vT, int NUM_WARP>
__global__
void LoadBalanceExtractKernel(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val, 
                         int x_nnz, /*SparseVec* d_sparse_x, */
                         iT* d_sparse_x_key, vT* d_sparse_x_val,
                         int rows_per_block, 
                         int count_height, int count_width,
                         int* d_count,
                         int small_len,
                         int small_len_blocks,
                         int* d_group,
                         iT* d_bin_row, vT* d_bin_val) {
 
  int bid = blockIdx.x;
  int tid = bid * blockDim.x + threadIdx.x;

  int row = 0;
  int row_block = 0;
 

 
}



template<typename iT, typename uiT, typename vT>
int CscBasedBucketLoadBalanceSpmspvDriver(
														iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
						                iT m, iT n, iT nnz, 
						                iT x_nnz, /*SparseVec* d_sparse_x, */
						                iT* d_sparse_x_key, vT* d_sparse_x_val,
						                const vT  alpha, 
						                iT* ynnz, vT* d_y,
						                int* d_ptr_col_len, void* d_pre_alloc_buffer,
						                int num_buckets,
						                int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;
  
  int buckets = std::min<int>(num_buckets, (m+127/(double)128));
  int rows_per_block = std::max<int>(128, ceil(m/(double)buckets));
 
#ifdef TIMING
  std::cout << "rows_per_block = " << rows_per_block << ", buckets = " 
            << buckets <<std::endl;
#endif

  int count_width = x_nnz;
  int count_len = (count_width * buckets + 1);
  
  int* d_count = (int*)d_pre_alloc_buffer;
  checkCudaErrors(cudaMemset(d_count, 0, count_len * sizeof(int)));

  int begin_addr_len = (buckets+1);
  int* d_begin_addr = (int*)(d_pre_alloc_buffer + count_len * sizeof(int));

#ifdef TIMING
  std::cout << "count_width = " << count_width << ", count_len = " 
            << count_len <<std::endl;

  std::cout << "call cuda_smsv_bucket version." << std::endl;
  std::cout << "step-1: count num of elems of each column." << std::endl;
#endif

#ifdef TIMING 
  SpmspvTimer step1_timer;
  step1_timer.Start();

  std::cout <<"x_nnz = " << x_nnz << " blocks_per_grid = " << blocks_per_grid 
       << " threads_per_block = " <<  threads_per_block <<std::endl;
#endif

  LoadBalanceCountElemsPerColKernel<iT, uiT, vT>
          <<< blocks_per_grid, threads_per_block >>>(
          d_csc_col, x_nnz, d_sparse_x_key, d_ptr_col_len);
  //cudaDeviceSynchronize();
  
  cudaError_t err_r;
  checkCudaRoutineReturn("smsv_extract_counter_kernel");

#ifdef TIMING  
  double step1_time = step1_timer.Stop();
  std::cout << "step-1 counter time = " << step1_time << " ms." << std::endl;
#endif
#ifdef PRINT_TEMP_RES
  PrintDenseVec2FileDevice(x_nnz, d_ptr_col_len, "num_of_elems_column.info");
#endif
  // SpmspvTimer step2_timer;
  // step2_timer.Start();

  // //cub's sort can not deal with the cases when value is a structure (not a pupular type).
  // thrust::sort_by_key(thrust::device, 
  //               d_ptr_col_len, d_ptr_col_len + x_nnz, d_sparse_x);

  // //cudaDeviceSynchronize();
  // double step2_time = step2_timer.Stop();
  // std::cout << "step-2 sort by key time = " << step2_time << " ms." << std::endl;

  // PrintDenseVec2FileDevice(x_nnz, d_ptr_col_len, "sort_num_of_elems_column.info");
  // PrintSparseVec2FileDevice(x_nnz, d_sparse_x, "sort_sparseX.info");

#ifdef TIMING
  SpmspvTimer step4_timer;
  step4_timer.Start();
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

  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

#ifdef TIMIGN
  double step4_time = step4_timer.Stop();
  std::cout << "step-4 exclusive_scan time = " << step4_time 
            << " ms." << std::endl;
#endif

#ifdef PRINT_TEMP_RES            
  //check-scan-result.
  PrintMat2FileDevice(buckets, count_width, d_count, 
  						            "load_balance_scan_count_matrix.info");
#endif

#ifdef TIMING 
  SpmspvTimer step5_timer;
  step5_timer.Start();
#endif

  //threads = (buckets + 1);
  threads = min(1024, (buckets + 1 + 31) / 32 * 32);
  blocks = (buckets + 1 + threads - 1)/threads;
  bucket_copy_kernel<iT, uiT, vT><<< blocks, threads >>>
      (buckets, count_width, d_count, d_begin_addr);

#ifdef TIMING
  double step5_time = step5_timer.Stop();
  std::cout << "step-5 copy time = " << step5_time << " ms." << std::endl;
#endif

	int len;
  checkCudaErrors(cudaMemcpy(&len, &d_count[count_len-1], sizeof(int),   
                            cudaMemcpyDeviceToHost));
  const int bin_len = len;

#ifdef TIMING
  std::cout << "bebug: bin_len = " << bin_len <<std::endl;
#endif

  //TODO:
   iT* d_bin_row = (iT*)(d_begin_addr + begin_addr_len * sizeof(int));
   
   void* temp = (void*)(d_bin_row + bin_len * sizeof(iT));
   int size_vT = sizeof(vT);
   vT* d_bin_val = (vT*)((reinterpret_cast<std::size_t>(temp) + size_vT - 1) / 
                    size_vT * size_vT);

  
#ifdef TIMING 
  SpmspvTimer step6_timer;
  step6_timer.Start();
#endif

  threads = THREADS_PER_BLOCK;//1024
  blocks = small_len_blocks + big_len_blocks;

#ifdef TIMING 
  printf("threads=%d, num_blocks=%d\n", threads, blocks);
#endif
  //num_blocks = (x_nnz+COL_NUM_PER_BLOCK-1)/COL_NUM_PER_BLOCK;//
  LoadBalanceExtractKernel<iT, uiT, vT, THREADS_PER_BLOCK/LM_WARP_SIZE>
                <<< blocks, threads >>>
                (d_csc_row, d_csc_col, d_csc_val, 
                 x_nnz, d_sparse_x_key,
                 d_sparse_x_val, 
                 rows_per_block, 
                 buckets, count_width,
                 d_count, 
                 small_len,
                 small_len_blocks,
                 d_group,
                 d_bin_row, d_bin_val);                               

#ifdef TIMING 
  double step6_time = step6_timer.Stop();
  std::cout << "step-6 extract time = " << step6_time << " ms." << std::endl;
#endif
#ifdef PRINT_TEMP_RES  
  PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, 
                                "load-balance-bin.info");
#endif
#ifdef TIMIGN
  SpmspvTimer step7_timer;
  step7_timer.Start();
#endif

  threads = 1024;
  blocks = buckets;//
  AtomicBucketReductionKernel<iT, uiT, vT><<< blocks, threads >>>
                          (buckets, d_begin_addr, 
                          d_bin_row, d_bin_val, 
                          d_y);
#ifdef TIMING 
  double step7_time = step7_timer.Stop();
  std::cout << "step-7 reduction time = " << step7_time << " ms." << std::endl;
  //PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, "segmented-sort.info");

  double all_time = step1_time + step2_time 
                  + step3_time + step4_time 
                  + step5_time + step6_time + step7_time;
  std::cout << "all time = " << all_time << "ms." << std::endl;
#endif

  if (h_colLen) free(h_colLen);

  return err;
}
#endif //LOAD_BALANCE_H_
