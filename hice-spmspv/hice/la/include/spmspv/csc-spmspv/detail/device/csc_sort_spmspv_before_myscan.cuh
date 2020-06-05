// This file provides the csc-based sort or no-sort spmspv method.

#ifndef SORT_BASED_METHOD_H_
#define SORT_BASED_METHOD_H_

//#include "common_cuda.h"
//#include "utils_cuda.h"
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "spmspv/config.h"
//#include "print_util.cuh"

//replay sort-based version in push-pull lib.
//output: dense-vector.

//because d_ptr_col_len is a member variable, 
//but I redefine a new d_ptr_col_len variable in a member routine and alloc for it.
template<typename iT, typename uiT, typename vT>
__global__
void CountNonzerosPerColKernel(const iT* d_csc_col,  
  							const iT x_nnz, 
  							/*SparseVec* d_sparse_x,*/
  							iT* d_sparse_x_key,
  							vT* d_sparse_x_val,
  							int* d_ptr_col_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < x_nnz) {
  	// iT col_index = d_sparse_x[tid].key;
  	// iT len = d_csc_col[col_index+1] - d_csc_col[col_index];
   //  d_ptr_col_len[tid] = len;
    iT col_index = __ldg(d_sparse_x_key + tid);
    iT start = __ldg(d_csc_col + col_index);
    iT end = __ldg(d_csc_col + col_index + 1);
    d_ptr_col_len[tid] = end - start;
  }
}

template<typename iT, typename uiT, typename vT>
__global__
void ExtractKernel(const iT* d_csc_row, 
                            const iT* d_csc_col, 
                            const vT* d_csc_val,
              							const iT m, 
              							const iT n, 
              							const iT nnz, 
              							const iT x_nnz, 
              							SparseVec* d_sparse_x,
              							iT* d_bin_row, vT* d_bin_val,
              							int* d_ptr_col_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
//#define OUTPUT_COLLEN
#ifdef OUTPUT_COLLEN
  if (tid == 0) {
  	for (int i = 0; i < x_nnz; i++)
  		printf("%d, ", d_ptr_col_len[i]);
  	printf("\n\n");
  }
  __syncthreads();
#endif
  //WARNING: don't forget to initilize bin_len to initilize addr.
  int bin_inx = d_ptr_col_len[tid];
  if (tid < x_nnz) {
  	iT col_index = d_sparse_x[tid].key;
  	vT x_val     = d_sparse_x[tid].val;
  	int end = d_csc_col[col_index+1];
  	for (int j = d_csc_col[col_index]; j < end; j++) {
  		d_bin_row[bin_inx] = d_csc_row[j];  
  		d_bin_val[bin_inx] = d_csc_val[j] * x_val;
      //printf(" %d : %f \n", d_bin_row[bin_inx], d_bin_val[bin_inx]);
  		bin_inx++;
  	}
  }
}

template<typename iT, typename uiT, typename vT>
__global__
void CorrectnessKernel(int bin_len, 
                             const iT* d_row,  
                             const vT* d_val,
                             int* y_nnz, 
                             SparseVec* d_y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  //printf("tid = %d \n", tid);
  if (tid == 0) {
    //printf("bin_len = %d \n", bin_len);
    int index = 0;
    int i=0;
    for (i = 0; i < bin_len - 1; i++) {
      if (d_row[i] != d_row[i+1]) {
        d_y[index].val = d_val[i];
        d_y[index].key = d_row[i];
        index++;
        //printf("i = %d \n", i);
        //printf("index = %d \n", index);
      }
    }
    //printf("here \n");
    //printf("i = %d \n", i);
    //printf("index = %d \n", index);
    d_y[index].val = d_val[i];
    d_y[index].key = d_row[i];
    //printf("d_y[index].key = %d, d_y[index].val = %f \n",d_y[index].key, d_y[index].val);
    index++;
    *y_nnz = index;
    //printf("y_nnz = %d \n", *y_nnz);
  }
}

template<typename iT, typename uiT, typename vT>
int ExtractDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
						iT m, iT n, iT nnz, 
						iT x_nnz, SparseVec* d_sparse_x,
						iT* d_bin_row, vT* d_bin_val,
						int* d_ptr_col_len, 
            int blocks_per_grid, int threads_per_block) {
	int err = SPMSPV_SUCCESS;
  ExtractKernel<iT, uiT, vT>
                <<< blocks_per_grid, threads_per_block >>>
                (d_csc_row, d_csc_col, d_csc_val,
  							m, n, nnz, x_nnz, d_sparse_x,
  							d_bin_row, d_bin_val,
  							d_ptr_col_len);
  return err;
}

template<typename iT, typename uiT, typename vT>
int CscBasedSortSpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
                        				iT m, iT n, iT nnz, 
                        				iT x_nnz, SparseVec* d_sparse_x,
                                int* d_sparse_x_index, vT* d_sparse_x_val,
                        				const vT  alpha, 
                                iT* y_nnz, iT* d_y_inx, vT* d_y_val,
                                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                                int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;
 	std::cout <<"x_nnz = " << x_nnz << " blocks_per_grid = " << blocks_per_grid 
            << " threads_per_block = " <<  threads_per_block <<std::endl;

  SpmspvTimer step1_timer;
  step1_timer.Start();
  CountNonzerosPerColKernel<iT, uiT, vT>
           <<< blocks_per_grid, threads_per_block >>>
           (d_csc_col, x_nnz, d_sparse_x_index, d_sparse_x_val, d_ptr_col_len);
  //cudaDeviceSynchronize();
  double step1_time = step1_timer.Stop();
  std::cout << "step-1 counter time = " << step1_time << " ms." << std::endl;
  
  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("CountNonzerosPerColKernel() invocate error-2.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
  }

  SpmspvTimer step2_timer;
  step2_timer.Start();

  thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_ptr_col_len);

#ifndef USE_CUB_SCAN
  ////thrust::exclusive_scan(thrust::device, d_ptr, d_ptr+x_nnz+1, d_ptr);
  thrust::exclusive_scan(d_ptr, d_ptr + x_nnz + 1, d_ptr);
#else
  void  *d_temp_storage = NULL;
	size_t  temp_storage_bytes = 0;
	CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_ptr_col_len, d_ptr_col_len, x_nnz+1));
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
              temp_storage_bytes));
	
	CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, 
               temp_storage_bytes, d_ptr_col_len, d_ptr_col_len, x_nnz+1));
	if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

  //cudaDeviceSynchronize();
  double step2_time = step2_timer.Stop();
  std::cout << "step-2 exclusive_scan time = " << step2_time 
            << " ms." << std::endl;

  int len;
  cudaErrCheck(cudaMemcpy(&len, &d_ptr_col_len[x_nnz], sizeof(int),   
                           cudaMemcpyDeviceToHost));
  
  const int bin_len = len;
  std::cout << "bin_len = " << bin_len << std::endl;
#if 0 
  iT* d_bin_row = NULL;
  vT* d_bin_val = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_bin_row,  bin_len * sizeof(iT)));
  cudaErrCheck(cudaMalloc((void **)&d_bin_val,  bin_len * sizeof(vT)));
#endif
  iT* d_bin_row = (iT*)d_pre_alloc_buffer;
  //vT* d_bin_val = (vT*)(d_pre_alloc_buffer + bin_len * sizeof(iT));//todo
  vT* d_bin_val = (vT*)(d_pre_alloc_buffer + bin_len * sizeof(vT));//todo

  SpmspvTimer step3_timer;
  step3_timer.Start();
	// ExtractDriver<iT, uiT, vT>
 //             (d_csc_row, d_csc_col, d_csc_val,
	// 					  m, n, nnz, x_nnz, d_sparse_x, 
	// 					  d_bin_row, d_bin_val,
	// 					  d_ptr_col_len, 
 //             blocks_per_grid, threads_per_block);
  LoadBalanceExtractDriver<iT, uiT, vT>
              (d_csc_row, d_csc_col, d_csc_val,
              m, n, nnz, x_nnz, 
              d_sparse_x_index, d_sparse_x_val, 
              bin_len,
              d_bin_row, d_bin_val,
              d_ptr_col_len, 
              blocks_per_grid, threads_per_block);
  //cudaDeviceSynchronize();
  double step3_time = step3_timer.Stop();
  std::cout << "step-3 extract time = " << step3_time << " ms." << std::endl;

  // PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, 
  //                                "extract.info");

  //std::cout << "step-4: sorting." << std::endl;

  thrust::device_ptr<iT> d_bin_row_ptr 
                          = thrust::device_pointer_cast(d_bin_row);
  thrust::device_ptr<vT> d_bin_val_ptr 
                          = thrust::device_pointer_cast(d_bin_val);
  
  thrust::device_ptr<iT> d_y_inx_ptr = thrust::device_pointer_cast(d_y_inx);
  thrust::device_ptr<vT> d_y_val_ptr = thrust::device_pointer_cast(d_y_val);

  SpmspvTimer step4_timer;
  step4_timer.Start();

#ifndef USE_CUB_RADIX_SORT
  thrust::sort_by_key(thrust::device, d_bin_row_ptr, 
                      d_bin_row_ptr + bin_len, d_bin_val_ptr);
#else
  int endbit = sizeof(int)*8;
  int DO_END_BIT_OPT = 0;
  if (DO_END_BIT_OPT)
  	endbit = min(endbit, (int)log2((float)m)+1);

  temp_storage_bytes = 0;
	d_temp_storage = NULL;
  CubDebugExit( DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
						    	(iT*)d_bin_row, (iT*)d_bin_row, (vT*)d_bin_val, 
						    	(vT*)d_bin_val, bin_len, 0, endbit) );

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
              temp_storage_bytes));

  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
								(iT*)d_bin_row, (iT*)d_bin_row, (vT*)d_bin_val, 
						    	(vT*)d_bin_val, bin_len, 0, endbit) );

  if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif
  //cudaDeviceSynchronize();
	double step4_time = step4_timer.Stop();
  std::cout << "step-4 sort by key time = " << step4_time 
            << " ms." << std::endl;
  //PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, 
  //                                    "sort.info");
  //thrust::reduce_by_key
  //https://thrust.github.io/doc/group__reductions.html#gad5623f203f9b3fdcab72481c3913f0e0
  //std::cout << "step-5: segmented-sum." << std::endl;
  SpmspvTimer step5_timer;
  step5_timer.Start();
 
#ifndef MODERNGPU_REDUCE_BY_KEY 
  //thrust::pair<iT*,vT*> new_end;
  thrust::pair< thrust::device_ptr<iT>, thrust::device_ptr<vT> > new_end;

  new_end = thrust::reduce_by_key(thrust::device, d_bin_row_ptr, d_bin_row_ptr+bin_len, d_bin_val_ptr, d_y_inx_ptr, d_y_val_ptr);

  *y_nnz = new_end.first - d_y_inx_ptr;
  std::cout << "y_nnz = " << * y_nnz << std::endl;
#else
  int  y_nvals_t = 0;
  //
  mgpu::ContextPtr d_context(mgpu::CreateCudaDevice(0));
	ReduceByKey( (iT*)d_bin_row, (vT*)d_bin_val, bin_len,
				(vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_inx, d_y_val,
				&y_nvals_t, (int*)0, *d_context);
	*y_nnz = y_nvals_t;
#endif 
  
  double step5_time = step5_timer.Stop();
  std::cout << "step-5 segmented sum time = " << step5_time 
            << " ms." << std::endl;
  
  //thrust::unique_by_key
  //https://thrust.github.io/doc/group__stream__compaction.html#gac6de1d1309dbe325096ceb61132f9749
  double all_time = step1_time + step2_time + step3_time 
                  + step4_time + step5_time;
  std::cout << "all time = " << all_time <<std::endl;

#if 0
  if (d_y_nnz)
	  cudaErrCheck(cudaFree(d_y_nnz));
  if (d_bin_row)
	  cudaErrCheck(cudaFree(d_bin_row));
  if (d_bin_val)
	  cudaErrCheck(cudaFree(d_bin_val));
#endif
  return err;
}

template<typename iT, typename uiT, typename vT>
__global__
void AtomicReductionKernel(iT* d_bin_row, 
                           vT* d_bin_val, 
                           vT* d_y,
                           int bin_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < bin_len) {
      int row_index = d_bin_row[tid];
      int value = d_bin_val[tid];
      atomicAdd(&d_y[row_index], value);
  }
}

template<typename iT, typename uiT, typename vT>
int CscBasedNoSortSpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
                iT m, iT n, iT nnz, 
                iT x_nnz, SparseVec* d_sparse_x,
                int* d_sparse_x_index, vT* d_sparse_x_val,
                const vT  alpha, 
                iT* y_nnz, vT* d_y,
                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;

  SpmspvTimer step1_timer;
  step1_timer.Start();

  std::cout << "x_nnz = " << x_nnz << " blocks_per_grid = " << blocks_per_grid 
            << " threads_per_block = " << threads_per_block << std::endl;
#if 1
  CountNonzerosPerColKernel<iT, uiT, vT>
            <<< blocks_per_grid, threads_per_block >>>
            (d_csc_col, x_nnz, d_sparse_x_index, d_sparse_x_val, 
            d_ptr_col_len);
  //cudaDeviceSynchronize();
  //PrintDenseVec2FileDevice(x_nnz, d_ptr_col_len, "colLen.info");

  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("CountNonzerosPerColKernel() invocate error-2.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
  }

  double step1_time = step1_timer.Stop();
  std::cout << "step-1 counter time = " << step1_time << " ms." << std::endl;
#endif

  SpmspvTimer step2_timer;
  step2_timer.Start();

#if 1
  thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_ptr_col_len);

#ifndef USE_CUB_SCAN
  ////thrust::exclusive_scan(thrust::device, d_ptr, d_ptr+x_nnz+1, d_ptr);
  thrust::exclusive_scan(d_ptr, d_ptr+x_nnz+1, d_ptr);
#else
  void* d_temp_storage = NULL;
  size_t  temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, 
              d_ptr_col_len, d_ptr_col_len, x_nnz+1));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
               temp_storage_bytes));
  
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,   
                                        d_ptr_col_len, d_ptr_col_len, 
                                        x_nnz+1));
  if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

#endif

  PrintDenseVec2FileDevice(x_nnz+1, d_ptr_col_len, "scan.info");
  //cudaDeviceSynchronize();
  double step2_time = step2_timer.Stop();
  std::cout << "step-2 exclusive_scan time = " << step2_time 
            << " ms." << std::endl;

  int len;
  cudaErrCheck(cudaMemcpy(&len, &d_ptr_col_len[x_nnz], sizeof(int),   
                           cudaMemcpyDeviceToHost));
  
  const int bin_len = len;
  std::cout << "bin_len = " << bin_len << std::endl;

  iT* d_bin_row = (iT* )d_pre_alloc_buffer;
  //vT* d_bin_val = (vT*)(d_pre_alloc_buffer + bin_len * sizeof(iT));
  vT* d_bin_val = (vT* )(d_pre_alloc_buffer + bin_len * sizeof(vT));

  int* d_startingIds = (int* )(d_pre_alloc_buffer + 2 * bin_len * sizeof(vT));

  SpmspvTimer step3_timer;
  step3_timer.Start();
  // ExtractDriver<iT, uiT, vT>
  //                     (d_csc_row, d_csc_col, d_csc_val,
  //                     m, n, nnz, x_nnz, d_sparse_x, 
  //                     d_bin_row, d_bin_val,
  //                     d_ptr_col_len, 
  //                     blocks_per_grid, threads_per_block);
  //
  // LoadBalanceFuseExtractDriver
  // LoadBalanceModifyExtractDriver
  // LoadBalanceFuseExtractAndReductionDriver
  // LoadBalanceExtractDriver<iT, uiT, vT>
  //                     (d_csc_row, d_csc_col, d_csc_val,
  //                     m, n, nnz, x_nnz, 
  //                     d_sparse_x_index, d_sparse_x_val,
  //                     bin_len,
  //                     d_bin_row, d_bin_val,
  //                     d_ptr_col_len, 
  //                     blocks_per_grid, threads_per_block);
  // LoadBalanceFuseExtractAndReductionDriver<iT, uiT, vT>
  //                     (d_csc_row, d_csc_col, d_csc_val,
  //                     m, n, nnz, x_nnz, 
  //                     d_sparse_x_index, d_sparse_x_val,
  //                     bin_len,
  //                     /*d_bin_row, d_bin_val,*/
  //                     d_y,
  //                     d_ptr_col_len, 
  //                     blocks_per_grid, threads_per_block);
  // TwoLayerLoadBalanceExtractDriver<iT, uiT, vT>
  //                     (d_csc_row, d_csc_col, d_csc_val,
  //                     m, n, nnz, x_nnz, 
  //                     d_sparse_x_index, d_sparse_x_val,
  //                     bin_len,
  //                     d_bin_row, d_bin_val,
  //                     d_ptr_col_len, 
  //                     d_startingIds,
  //                     blocks_per_grid, threads_per_block);
  TwoLayerLoadBalanceFuseExtractAndReductionDriver<iT, uiT, vT>
                      (d_csc_row, d_csc_col, d_csc_val,
                      m, n, nnz, x_nnz, 
                      d_sparse_x_index, d_sparse_x_val,
                      bin_len,
                      d_bin_row, d_bin_val,
                      d_y,
                      d_ptr_col_len, 
                      d_startingIds,
                      blocks_per_grid, threads_per_block);
  //cudaDeviceSynchronize();
  double step3_time = step3_timer.Stop();
  std::cout << "step-3 extract time = " << step3_time << " ms." << std::endl;
  // PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val,
  //                                     "extract.info");
  //std::cout << "step-4: atomic reduction." << std::endl;
#if 0
  SpmspvTimer step4_timer;
  step4_timer.Start();

  int threads = 1024;
  int num_blocks = (bin_len+threads-1)/threads;
  assert(num_blocks < 2147483647);//
  AtomicReductionKernel<iT, uiT, vT><<< num_blocks, threads >>>
                                                (d_bin_row, d_bin_val, 
                                                d_y, bin_len);

  double step4_time = step4_timer.Stop();
  std::cout << "step-4 atomic reduction time = " << step4_time 
            << " ms." << std::endl;
  double all_time = step1_time + step2_time + step3_time + step4_time;
  std::cout << "all time = " << all_time << std::endl;
#endif
  double all_time = step1_time + step2_time + step3_time; // + step4_time;
  std::cout << "all time = " << all_time << std::endl;


  if (gadsys)
    cudaErrCheck(cudaFree(gadsys));
  return err;
}
#endif //SORT_BASED_METHOD_H_
