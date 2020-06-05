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

#include "csc_sort_scan_fuse.cuh"
//#endif

// #ifdef MODERNGPU_REDUCE_BY_KEY
#include <moderngpu.cuh>
// #endif


//replay sort-based version in push-pull lib and nosort version of mine.
//output: dense-vector.


//#define STEP_PERF
//#define DEBUG

//when access d_ptr_col_len，ilegal memory access error:why？
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

//TODO: need to 加速
template<typename iT, typename uiT, typename vT>
__global__
void ExtractKernel(const iT* d_csc_row, 
                  const iT* d_csc_col, 
                  const vT* d_csc_val,
    							const iT m, 
    							const iT n, 
    							const iT nnz, 
    							const iT x_nnz, 
    							/*SparseVec* d_sparse_x,*/
						      int* d_sparse_x_key,
						      vT* d_sparse_x_val,
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
  	iT col_index = d_sparse_x_key[tid];
  	vT x_val     = d_sparse_x_val[tid];
    int begin = d_csc_col[col_index];
  	int end = d_csc_col[col_index + 1];
  	for (int j = begin; j < end; j++) {
  		d_bin_row[bin_inx] = d_csc_row[j];  
  		d_bin_val[bin_inx] = d_csc_val[j] * x_val;
      //printf(" %d : %f \n", d_bin_row[bin_inx], d_bin_val[bin_inx]);
  		bin_inx++;
  	}
  }
}

template<typename iT, typename uiT, typename vT>
__global__
void ExtractFuseKernel(const iT* d_csc_row, const iT* d_csc_col, const vT* d_csc_val,
        							const iT m, const iT n, const iT nnz, 
                      const iT x_nnz, int* d_sparse_x_key, vT* d_sparse_x_val,
        							/*iT* d_bin_row, vT* d_bin_val,*/
        							/*int* d_ptr_col_len,*/
        							vT* d_y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < x_nnz) {
  	iT col_index = d_sparse_x_key[gid];
  	vT x_val     = d_sparse_x_val[gid];
    int begin = d_csc_col[col_index];
  	int end = d_csc_col[col_index + 1];
    #if 1
  	for (int j = begin; j < end; j++) {
  		int reg_row_id = d_csc_row[j];
  		vT reg_val = d_csc_val[j] * x_val;
  		atomicAdd(&d_y[reg_row_id], reg_val);
  	}
    #endif
  }
}

#if 0
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
#endif


template<typename iT, typename uiT, typename vT>
int ExtractDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
						iT m, iT n, iT nnz, 
						iT x_nnz, /*SparseVec* d_sparse_x,*/
						int* d_sparse_x_key, vT* d_sparse_x_val,
						iT* d_bin_row, vT* d_bin_val,
						int* d_ptr_col_len, 
            int blocks_per_grid, int threads_per_block) {
	int err = SPMSPV_SUCCESS;
  ExtractKernel<iT, uiT, vT>
                <<< blocks_per_grid, threads_per_block >>>
                (d_csc_row, d_csc_col, d_csc_val,
  							 m, n, nnz, x_nnz, /*d_sparse_x,*/
                 d_sparse_x_key, d_sparse_x_val,
  							 d_bin_row, d_bin_val,
  							 d_ptr_col_len);
  return err;
}

template<typename iT, typename uiT, typename vT>
int ExtractFuseDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
						iT m, iT n, iT nnz, 
						iT x_nnz, /*SparseVec* d_sparse_x,*/
						int* d_sparse_x_key, vT* d_sparse_x_val,
						/*iT* d_bin_row, vT* d_bin_val,*/
						/*int* d_ptr_col_len, */
            int blocks_per_grid, int threads_per_block,
            vT* d_y) {
	int err = SPMSPV_SUCCESS;
  ExtractFuseKernel<iT, uiT, vT>
                <<< blocks_per_grid, threads_per_block >>>
                (d_csc_row, d_csc_col, d_csc_val,
  							m, n, nnz, x_nnz, /*d_sparse_x,*/
                d_sparse_x_key, d_sparse_x_val,
  							/*d_bin_row, d_bin_val,*/
  							/*d_ptr_col_len,*/ d_y);
  return err;
}

template<typename iT, typename uiT, typename vT>
int CscBasedSortNaiveSpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
                        				iT m, iT n, iT nnz, 
                        				iT x_nnz,
                                int* d_sparse_x_index, vT* d_sparse_x_val,
                        				const vT  alpha, 
                                iT* y_nnz, iT* d_y_inx, vT* d_y_val,
                                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                                int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;
 	// std::cout <<"x_nnz = " << x_nnz << " blocks_per_grid = " << blocks_per_grid 
  //           << " threads_per_block = " <<  threads_per_block <<std::endl;
#ifdef STEP_PERF 
  SpmspvTimer step1_timer;
  step1_timer.Start();
#endif 

  CountNonzerosPerColKernel<iT, uiT, vT>
           <<< blocks_per_grid, threads_per_block >>>
           (d_csc_col, x_nnz, d_sparse_x_index, d_sparse_x_val, d_ptr_col_len);
           
#ifdef STEP_PERF           
  //cudaDeviceSynchronize();
  double step1_time = step1_timer.Stop();
  std::cout << "step-1 counter time = " << step1_time << " ms." << std::endl;
#endif

  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("CountNonzerosPerColKernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
  //PrintDenseVec2FileDevice(x_nnz, d_ptr_col_len, "colLen.info");
#ifdef STEP_PERF  
  SpmspvTimer step2_timer;
  step2_timer.Start();
#endif

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

#ifdef STEP_PERF    
  //PrintDenseVec2FileDevice(x_nnz+1, d_ptr_col_len, "scan.info");

  //cudaDeviceSynchronize();
  double step2_time = step2_timer.Stop();
  std::cout << "step-2 exclusive_scan time = " << step2_time 
            << " ms." << std::endl;
#endif

  #if 1
  cudaDeviceSynchronize();
  err_r = cudaGetLastError();
  if (cudaSuccess != err_r) {
    printf("scan() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif

  int len;
  cudaErrCheck(cudaMemcpy(&len, &d_ptr_col_len[x_nnz], sizeof(int),   
                           cudaMemcpyDeviceToHost));
  
  const int bin_len = len;
  //std::cout << "bin_len = " << bin_len << std::endl;

#if 1 
  iT* d_bin_row = NULL;
  vT* d_bin_val = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_bin_row,  bin_len * sizeof(iT)));
  cudaErrCheck(cudaMalloc((void **)&d_bin_val,  bin_len * sizeof(vT)));
#endif

#if 0
  iT* d_bin_row = (iT*)d_pre_alloc_buffer;
  //vT* d_bin_val = (vT*)(d_pre_alloc_buffer + bin_len * sizeof(iT));//
  vT* d_bin_val = (vT*)(d_pre_alloc_buffer + bin_len * sizeof(vT));//
#endif

#ifdef STEP_PERF  
  SpmspvTimer step3_timer;
  step3_timer.Start();
#endif

	ExtractDriver<iT, uiT, vT>
             (d_csc_row, d_csc_col, d_csc_val,
						  m, n, nnz, x_nnz, d_sparse_x_index,
              d_sparse_x_val, 
						  d_bin_row, d_bin_val,
						  d_ptr_col_len, 
             blocks_per_grid, threads_per_block);
#ifdef STEP_PERF   
  //cudaDeviceSynchronize();
  double step3_time = step3_timer.Stop();
  std::cout << "step-3 extract time = " << step3_time << " ms." << std::endl;
#endif

#if 1
  cudaDeviceSynchronize();
  err_r = cudaGetLastError();
  if (cudaSuccess != err_r) {
    printf("ExtractDriver() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif

  // PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, 
  //                                 "extract.info");

  //std::cout << "step-4: sorting." << std::endl;
  thrust::device_ptr<iT> d_bin_row_ptr 
                          = thrust::device_pointer_cast(d_bin_row);
  thrust::device_ptr<vT> d_bin_val_ptr 
                          = thrust::device_pointer_cast(d_bin_val);
  
  thrust::device_ptr<iT> d_y_inx_ptr = thrust::device_pointer_cast(d_y_inx);
  thrust::device_ptr<vT> d_y_val_ptr = thrust::device_pointer_cast(d_y_val);

#ifdef STEP_PERF  
  SpmspvTimer step4_timer;
  step4_timer.Start();
#endif

#ifndef USE_CUB_RADIX_SORT
  //TODO: test this branch.
  thrust::sort_by_key(thrust::device, d_bin_row_ptr, 
                      d_bin_row_ptr + bin_len, d_bin_val_ptr);
#else
  //TODO: DeviceRadixSort::SortPairs is out-of-place api!!
  iT* d_bin_row_temp = NULL;
  vT* d_bin_val_temp = NULL;
  cudaErrCheck(cudaMalloc((void **)&d_bin_row_temp,  bin_len * sizeof(iT)));
  cudaErrCheck(cudaMalloc((void **)&d_bin_val_temp,  bin_len * sizeof(vT)));

  int endbit = sizeof(int)*8;
  int DO_END_BIT_OPT = 0;
  if (DO_END_BIT_OPT)
  	endbit = min(endbit, (int)log2((float)m)+1);

  temp_storage_bytes = 0;
	d_temp_storage = NULL;
  CubDebugExit( DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
						    	(iT*)d_bin_row, (iT*)d_bin_row_temp, (vT*)d_bin_val, 
						    	(vT*)d_bin_val_temp, bin_len, 0, endbit) );

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
              temp_storage_bytes));

  CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
								(iT*)d_bin_row, (iT*)d_bin_row_temp, (vT*)d_bin_val, 
						    	(vT*)d_bin_val_temp, bin_len, 0, endbit) );

  if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

#ifdef STEP_PERF  
  //cudaDeviceSynchronize();
	double step4_time = step4_timer.Stop();
  std::cout << "step-4 sort by key time = " << step4_time << " ms." << std::endl;
  //PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, "sort.info");
#endif

#if 1
  cudaDeviceSynchronize();
  err_r = cudaGetLastError();
  if (cudaSuccess != err_r) {
    printf("sortKernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif


#if 1
  //thrust::reduce_by_key
  //https://thrust.github.io/doc/group__reductions.html#gad5623f203f9b3fdcab72481c3913f0e0
  //std::cout << "step-5: segmented-sum." << std::endl;

#ifdef STEP_PERF  
  SpmspvTimer step5_timer;
  step5_timer.Start();
#endif

#ifndef MODERNGPU_REDUCE_BY_KEY 
  //test this branch: ok.
  //thrust::pair<iT*,vT*> new_end;
  thrust::pair< thrust::device_ptr<iT>, thrust::device_ptr<vT> > new_end;

#ifdef USE_CUB_RADIX_SORT
  thrust::device_ptr<iT> d_bin_row_temp_ptr 
                          = thrust::device_pointer_cast(d_bin_row_temp);
  thrust::device_ptr<vT> d_bin_val_temp_ptr 
                          = thrust::device_pointer_cast(d_bin_val_temp);
  new_end = thrust::reduce_by_key(thrust::device, d_bin_row_temp_ptr, d_bin_row_temp_ptr + bin_len, 
                                  d_bin_val_temp_ptr, d_y_inx_ptr, d_y_val_ptr);
#else
  new_end = thrust::reduce_by_key(thrust::device, d_bin_row_ptr, d_bin_row_ptr + bin_len, d_bin_val_ptr, 
                                  d_y_inx_ptr, d_y_val_ptr);
#endif

  *y_nnz = new_end.first - d_y_inx_ptr;
  //std::cout << "inner y_nnz (cub-reduce-by-key) = " << *y_nnz << std::endl;

#else
  //TODO: test this branch:
  int  y_nvals_t = 0;
  //mgpu::ContextPtr d_context(mgpu::CreateCudaDevice(0));
  mgpu::ContextPtr d_context;
  d_context = mgpu::CreateCudaDevice(0);
#ifdef USE_CUB_RADIX_SORT
  ReduceByKey( (iT*)d_bin_row_temp, (vT*)d_bin_val_temp, bin_len,
				(vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_inx, d_y_val,
				&y_nvals_t, (int*)0, *d_context);
#else
  ReduceByKey( (iT*)d_bin_row, (vT*)d_bin_val, bin_len,
				(vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_inx, d_y_val,
				&y_nvals_t, (int*)0, *d_context);
#endif
	
	*y_nnz = y_nvals_t;
   //std::cout << "inner y_nnz (mgpu-reduce-by-key) = " << *y_nnz << std::endl;
#endif 

#ifdef STEP_PERF   
  double step5_time = step5_timer.Stop();
  std::cout << "step-5 segmented sum time = " << step5_time 
            << " ms." << std::endl;
#endif

#endif

#ifdef STEP_PERF  
  double all_time = step1_time + step2_time + step3_time 
                  + step4_time;// + step5_time;
  std::cout << "all time = " << all_time <<std::endl;
#endif

#if 1
  if (d_bin_row)
	  cudaErrCheck(cudaFree(d_bin_row));
  if (d_bin_val)
	  cudaErrCheck(cudaFree(d_bin_val));
#endif

#ifdef USE_CUB_RADIX_SORT
  if (d_bin_row_temp)
	  cudaErrCheck(cudaFree(d_bin_row_temp));
  if (d_bin_val_temp)
	  cudaErrCheck(cudaFree(d_bin_val_temp));
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

//For test correctness.
__global__
void ScanSerial(int* d_csc_col, int len, int* d_sparse_x_index, 
				int* d_ptr_col_len) {
	if (threadIdx.x == 0) {
	  int temp;
		int accum = 0;
		int i ;

		for(i = 0 ; i < len ; i++){
			//temp = d_ptr_col_len[i];
      temp = d_csc_col[d_sparse_x_index[i]+1] - d_csc_col[d_sparse_x_index[i]];
			d_ptr_col_len[i] = accum;
			accum += temp;
      //printf("%d, %d\n", i, d_ptr_col_len[i]);
		}
    d_ptr_col_len[len] = accum;
    //printf("%d\n", d_ptr_col_len[len]);
	}
}


template<typename iT, typename uiT, typename vT>
int CscBasedNoSortNaiveSpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
                iT m, iT n, iT nnz, int s_bin_len,
                iT x_nnz, int* d_sparse_x_index, vT* d_sparse_x_val,
                const vT  alpha, 
                iT* y_nnz, vT* d_y,
                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                int blocks_per_grid, int threads_per_block) {
  int err = SPMSPV_SUCCESS;

  ExtractFuseDriver<iT, uiT, vT>
                      (d_csc_row, d_csc_col, d_csc_val,
                      m, n, nnz, x_nnz, /*d_sparse_x, */
                      d_sparse_x_index, d_sparse_x_val,
                      /*d_bin_row, d_bin_val,*/
                      /*d_ptr_col_len, */
                      blocks_per_grid, threads_per_block, d_y);

#if 1
  cudaDeviceSynchronize();
  cudaError_t err_r = cudaGetLastError();
  if (cudaSuccess != err_r) {
    printf("extractKernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif
  
  //cudaDeviceSynchronize();
  return err;
}

template<typename iT, typename uiT, typename vT>
int CscBasedMySpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
                iT m, iT n, iT nnz, int s_bin_len,
                iT x_nnz, int* d_x_index, vT* d_x_val,
                const vT  alpha, 
                iT* y_nnz, vT* d_y,
                iT* d_y_key, vT* d_y_val, //TODO:new added
                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                int blocks_per_grid, int threads_per_block, bool isBinLenComputed,
                bool isSorted) {
  int err = SPMSPV_SUCCESS;

  // std::cout << "x_nnz = " << x_nnz << " blocks_per_grid = " << blocks_per_grid 
  //           << " threads_per_block = " << threads_per_block << std::endl;
  if(isBinLenComputed){
    if (s_bin_len == 0) {
      return err;
    }
  }
  
#ifdef STEP_PERF
  SpmspvTimer step1_timer;
  step1_timer.Start();
#endif

#ifdef STEP_PERF
  double step1_time = step1_timer.Stop();
  std::cout << "step-1 counter time = " << step1_time << " ms." << std::endl;
#endif
  //PrintDenseVec2FileDevice(x_nnz, d_ptr_col_len, "colLen.info");
#ifdef STEP_PERF
  SpmspvTimer step2_timer;
  step2_timer.Start();
#endif

  //int elemnum = x_nnz + 1;
  int elemnum = x_nnz;
  int workgroup = elemnum/(LT * (REGP + LOGP) * CT);
  int taillen = elemnum - STEP_NUM * LT * workgroup;
  int tail_group = (taillen + 2047)/2048;

  int* gadsys = NULL;
  int adsyslen = workgroup + tail_group + 2;

  // std::cout << "elemnum = " << elemnum <<  ", STEP_NUM = " << STEP_NUM 
  //           <<  ", workgroup = " << workgroup << ", taillen = " 
  //           << taillen << ", tail_group" << tail_group 
  //           << "adsyslen = " << adsyslen << std::endl;

  //int* adsys = new int[adsyslen]; 
  //memset(adsys, 0, adsyslen * sizeof(int));
 //cudaErrCheck(cudaMalloc((void **)&gadsys, adsyslen * sizeof(int)));
  gadsys = (int* )d_pre_alloc_buffer;

  //ScanSerial<<<1,1>>>(d_csc_col, x_nnz, d_sparse_x_index, d_ptr_col_len); 
  ScanFuseDriver(d_csc_col, x_nnz, d_x_index, d_ptr_col_len, 
    gadsys, adsyslen, workgroup, taillen, tail_group);

#if 0
  cudaDeviceSynchronize();
  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("scanKernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
  }
#endif

#ifdef STEP_PERF
  //cudaDeviceSynchronize();
  double step2_time = step2_timer.Stop();
  std::cout << "step-2 exclusive_scan time = " 
           << step2_time << " ms." << std::endl;
  //PrintDenseVec2FileDevice(x_nnz+1, d_ptr_col_len, "scan.info");
#endif

#ifdef STEP_PERF
  SpmspvTimer step3_timer;
  step3_timer.Start();
#endif

  //PrintDenseVec2FileDevice(x_nnz + 1, d_ptr_col_len, "scan.info");

  int len = 0;
  int bin_len = s_bin_len;
  if(!isBinLenComputed){
    cudaErrCheck(cudaMemcpy(&len, &d_ptr_col_len[x_nnz], sizeof(int),   
                             cudaMemcpyDeviceToHost));
    bin_len = len;
    if (bin_len == 0) {
      return err;
    }
  }
  //std::cout << "inner bin_len = " << bin_len << std::endl;
 


  iT* d_bin_row = NULL;
  vT* d_bin_val = NULL;
  int* d_startingIds = NULL;
  if(isSorted){
    d_bin_row = (iT* )d_pre_alloc_buffer;
    d_bin_val = (vT* )(d_pre_alloc_buffer + bin_len * sizeof(vT));
    d_startingIds = (int* )(d_pre_alloc_buffer + 2 * bin_len * sizeof(vT));
  }else{
    d_startingIds = (int* )(d_pre_alloc_buffer);
  }
  //iT* d_bin_row = (iT* )d_pre_alloc_buffer;
  //vT* d_bin_val = (vT* )(d_pre_alloc_buffer + bin_len * sizeof(vT));
  //int* d_startingIds = (int* )(d_pre_alloc_buffer + 2 * bin_len * sizeof(vT));
  
  // bool isWriteBack = false;
  // int* d_startingIds = (int* )(d_pre_alloc_buffer);

  // LoadBalanceFuseExtractAndReductionDriver<iT, uiT, vT>
  //                     (d_csc_row, d_csc_col, d_csc_val,
  //                     m, n, nnz, x_nnz, 
  //                     d_sparse_x_index, d_sparse_x_val,
  //                     bin_len,
  //                     /*d_bin_row, d_bin_val,*/
  //                     d_y,
  //                     d_ptr_col_len, 
  //                     blocks_per_grid, threads_per_block);
  //  LoadBalanceFuseExtractAndReductionModifyDriver<iT, uiT, vT>
  //                     (d_csc_row, d_csc_col, d_csc_val,
  //                     m, n, nnz, x_nnz, 
  //                     d_x_index, d_x_val,
  //                     bin_len,
  //                     d_bin_row, d_bin_val, isSorted,
  //                     d_y,
  //                     d_ptr_col_len,
  //                     d_startingIds,
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
  TwoLayerLoadBalanceFuseExtractAndReductionModifyDriver<iT, uiT, vT>
                      (d_csc_row, d_csc_col, d_csc_val,
                      m, n, nnz, x_nnz, 
                      d_x_index, d_x_val,
                      bin_len,
                      d_bin_row, d_bin_val, isSorted,
                      d_y,
                      d_ptr_col_len, 
                      d_startingIds, true,
                      blocks_per_grid, threads_per_block);

#if 0
  cudaDeviceSynchronize();
  err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("extractKernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif
  
  //cudaDeviceSynchronize();

#ifdef STEP_PERF
  double step3_time = step3_timer.Stop();
  if(isSorted){
    std::cout << "step-3 extract time = " << step3_time << " ms." << std::endl;
  }else{
    std::cout << "step-3 extract and reduction time = " << step3_time << " ms." << std::endl;
  }
  
  // PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val,
  //                                     "extract.info");
  double step4_time = 0.0;
  double step5_time = 0.0;
#endif

  if(isSorted){
    //std::cout << "step-4: sorting." << std::endl;
    thrust::device_ptr<iT> d_bin_row_ptr 
                            = thrust::device_pointer_cast(d_bin_row);
    thrust::device_ptr<vT> d_bin_val_ptr 
                            = thrust::device_pointer_cast(d_bin_val);
    
    thrust::device_ptr<iT> d_y_inx_ptr = thrust::device_pointer_cast(d_y_key);
    thrust::device_ptr<vT> d_y_val_ptr = thrust::device_pointer_cast(d_y_val);

#ifdef STEP_PERF
    SpmspvTimer step4_timer;
    step4_timer.Start();
#endif

#ifndef USE_CUB_RADIX_SORT
    //TODO: test this branch.
    thrust::sort_by_key(thrust::device, d_bin_row_ptr, 
                        d_bin_row_ptr + bin_len, d_bin_val_ptr);
#else
    //TODO: DeviceRadixSort::SortPairs is out-of-place api!!
    iT* d_bin_row_temp = NULL;
    vT* d_bin_val_temp = NULL;
    cudaErrCheck(cudaMalloc((void **)&d_bin_row_temp,  bin_len * sizeof(iT)));
    cudaErrCheck(cudaMalloc((void **)&d_bin_val_temp,  bin_len * sizeof(vT)));

    int endbit = sizeof(int)*8;
    int DO_END_BIT_OPT = 0;
    if (DO_END_BIT_OPT)
      endbit = min(endbit, (int)log2((float)m)+1);

    size_t  temp_storage_bytes = 0;
    void* d_temp_storage = NULL;
    CubDebugExit( DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
                    (iT*)d_bin_row, (iT*)d_bin_row_temp, (vT*)d_bin_val, 
                    (vT*)d_bin_val_temp, bin_len, 0, endbit) );

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
                temp_storage_bytes));

    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                  (iT*)d_bin_row, (iT*)d_bin_row_temp, (vT*)d_bin_val, 
                    (vT*)d_bin_val_temp, bin_len, 0, endbit) );

    if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

    //cudaDeviceSynchronize();
#ifdef STEP_PERF
    step4_time = step4_timer.Stop();
    std::cout << "step-4 sort by key time = " << step4_time << " ms." << std::endl;
    //PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val, "sort.info");
#endif

#if 1
    cudaDeviceSynchronize();
    cudaError_t err_r = cudaGetLastError();
    if (cudaSuccess != err_r) {
      printf("sortKernel() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }
#endif

#ifdef STEP_PERF
    SpmspvTimer step5_timer;
    step5_timer.Start();
#endif

#ifndef MODERNGPU_REDUCE_BY_KEY 
    //test this branch: ok.
    //thrust::pair<iT*,vT*> new_end;
    thrust::pair< thrust::device_ptr<iT>, thrust::device_ptr<vT> > new_end;

  #ifdef USE_CUB_RADIX_SORT
    thrust::device_ptr<iT> d_bin_row_temp_ptr 
                            = thrust::device_pointer_cast(d_bin_row_temp);
    thrust::device_ptr<vT> d_bin_val_temp_ptr 
                            = thrust::device_pointer_cast(d_bin_val_temp);
    new_end = thrust::reduce_by_key(thrust::device, d_bin_row_temp_ptr, d_bin_row_temp_ptr + bin_len, 
                                    d_bin_val_temp_ptr, d_y_inx_ptr, d_y_val_ptr);
  #else
    new_end = thrust::reduce_by_key(thrust::device, d_bin_row_ptr, d_bin_row_ptr + bin_len, d_bin_val_ptr, 
                                    d_y_inx_ptr, d_y_val_ptr);
  #endif

    *y_nnz = new_end.first - d_y_inx_ptr;
    //std::cout << "inner y_nnz (cub-reduce-by-key) = " << *y_nnz << std::endl;

#else
    //TODO: test this branch:
    int  y_nvals_t = 0;
    //mgpu::ContextPtr d_context(mgpu::CreateCudaDevice(0));
    mgpu::ContextPtr d_context;
    d_context = mgpu::CreateCudaDevice(0);
  #ifdef USE_CUB_RADIX_SORT
    ReduceByKey( (iT*)d_bin_row_temp, (vT*)d_bin_val_temp, bin_len,
          (vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_key, d_y_val,
          &y_nvals_t, (int*)0, *d_context);
  #else
    ReduceByKey( (iT*)d_bin_row, (vT*)d_bin_val, bin_len,
          (vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_key, d_y_val,
          &y_nvals_t, (int*)0, *d_context);
  #endif
    
    *y_nnz = y_nvals_t;
    //std::cout << "inner y_nnz (mgpu-reduce-by-key) = " << *y_nnz << std::endl;
#endif 

#ifdef STEP_PERF 
    step5_time = step5_timer.Stop();
    std::cout << "step-5 reduce-by-key time = " << step5_time 
              << " ms." << std::endl;
#endif

  #ifdef USE_CUB_RADIX_SORT
    if (d_bin_row_temp)
      cudaErrCheck(cudaFree(d_bin_row_temp));
    if (d_bin_val_temp)
      cudaErrCheck(cudaFree(d_bin_val_temp));
  #endif

  }

#ifdef STEP_PERF
  double all_time = 0.0;
  if(isSorted){
    all_time = step1_time + step2_time + step3_time + step4_time + step5_time;
  }else{
    all_time = step1_time + step2_time + step3_time; 
  }
  std::cout << "all time = " << all_time << std::endl;
#endif

  return err;
}

template<typename iT, typename uiT, typename vT>
int CscBasedMergeSpmspvDriver(iT* d_csc_row, iT* d_csc_col, vT* d_csc_val,
                iT m, iT n, iT nnz, int s_bin_len,
                iT x_nnz, int* d_x_key, vT* d_x_val,
                const vT  alpha, 
                iT* y_nnz, vT* d_y,
                int* d_y_key, vT* d_y_val,
                int* d_ptr_col_len, void* d_pre_alloc_buffer,
                int blocks_per_grid, int threads_per_block, bool isBinLenComputed,
                bool isSorted) {
  int err = SPMSPV_SUCCESS;

  // std::cout << "x_nnz = " << x_nnz << " blocks_per_grid = " << blocks_per_grid 
  //           << " threads_per_block = " << threads_per_block << std::endl;
  if(isBinLenComputed){
    if (s_bin_len == 0) {
      return err;
    }
  }
  
#ifdef STEP_PERF
  SpmspvTimer step1_timer;
  step1_timer.Start();
#endif

#ifdef STEP_PERF
  double step1_time = step1_timer.Stop();
  std::cout << "step-1 counter time = " << step1_time << " ms." << std::endl;
#endif
  //PrintDenseVec2FileDevice(x_nnz, d_ptr_col_len, "colLen.info");
#ifdef STEP_PERF
  SpmspvTimer step2_timer;
  step2_timer.Start();
#endif

  //int elemnum = x_nnz + 1;
  int elemnum = x_nnz;
  int workgroup = elemnum/(LT * (REGP + LOGP) * CT);
  int taillen = elemnum - STEP_NUM * LT * workgroup;
  int tail_group = (taillen + 2047)/2048;

  int* gadsys = NULL;
  int adsyslen = workgroup + tail_group + 2;

  // std::cout << "elemnum = " << elemnum <<  ", STEP_NUM = " << STEP_NUM 
  //           <<  ", workgroup = " << workgroup << ", taillen = " 
  //           << taillen << ", tail_group" << tail_group 
  //           << "adsyslen = " << adsyslen << std::endl;

  //int* adsys = new int[adsyslen]; 
  //memset(adsys, 0, adsyslen * sizeof(int));
 //cudaErrCheck(cudaMalloc((void **)&gadsys, adsyslen * sizeof(int)));
  gadsys = (int* )d_pre_alloc_buffer;

  //ScanSerial<<<1,1>>>(d_csc_col, x_nnz, d_sparse_x_index, d_ptr_col_len); 
  ScanFuseDriver(d_csc_col, x_nnz, d_x_key, d_ptr_col_len, 
    gadsys, adsyslen, workgroup, taillen, tail_group);
#if 0
  cudaDeviceSynchronize();
  cudaError_t err_r = cudaGetLastError();
  if ( cudaSuccess != err_r) {
    printf("scanKernel() invocate error.\n");
    std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
    exit(1);
  }
#endif

#if 0
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

#ifdef STEP_PERF
  //cudaDeviceSynchronize();
  double step2_time = step2_timer.Stop();
  std::cout << "step-2 exclusive_scan time = " 
           << step2_time << " ms." << std::endl;
  //PrintDenseVec2FileDevice(x_nnz+1, d_ptr_col_len, "scan.info");
#endif

#ifdef STEP_PERF
  SpmspvTimer step3_timer;
  step3_timer.Start();
#endif

  //PrintDenseVec2FileDevice(x_nnz + 1, d_ptr_col_len, "scan.info");

  int len = 0;
  int bin_len = s_bin_len;
  if(!isBinLenComputed){
    cudaErrCheck(cudaMemcpy(&len, &d_ptr_col_len[x_nnz], sizeof(int),   
                             cudaMemcpyDeviceToHost));
    bin_len = len;
    if (bin_len == 0) {
      return err;
    }
  }
  //std::cout << "inner bin_len = " << bin_len << std::endl;
 

 #if 1
  //iT* d_bin_row = (iT* )d_pre_alloc_buffer;
  //vT* d_bin_val = (vT* )(d_pre_alloc_buffer + bin_len * sizeof(vT));
  //int* d_startingIds = (int* )(d_pre_alloc_buffer + 2 * bin_len * sizeof(vT));
  //int* d_startingIds = (int* )(d_pre_alloc_buffer);

  iT* d_bin_row = NULL;
  vT* d_bin_val = NULL;
  int* d_startingIds = NULL;
  if(isSorted){
    d_bin_row = (iT* )d_pre_alloc_buffer;
    d_bin_val = (vT* )(d_pre_alloc_buffer + bin_len * sizeof(vT));
    d_startingIds = (int* )(d_pre_alloc_buffer + 2 * bin_len * sizeof(vT));
  }else{
    d_startingIds = (int* )(d_pre_alloc_buffer);
  }
  
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
   LoadBalanceFuseExtractAndReductionModifyDriver<iT, uiT, vT>
                      (d_csc_row, d_csc_col, d_csc_val,
                      m, n, nnz, x_nnz, 
                      d_x_key, d_x_val,
                      bin_len,
                      d_bin_row, d_bin_val, isSorted,
                      d_y, 
                      d_ptr_col_len,
                      d_startingIds, 
                      blocks_per_grid, threads_per_block);
  //cudaDeviceSynchronize();


#ifdef STEP_PERF
  double step3_time = step3_timer.Stop();
  if(isSorted){
    std::cout << "step-3 extract time = " << step3_time << " ms." << std::endl;
  }else{
    std::cout << "step-3 extract and reduction time = " << step3_time << " ms." << std::endl;
  }
  // PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row, d_bin_val,
  //                                     "extract.info");
  double step4_time = 0.0;
  double step5_time = 0.0;
#endif

  if(isSorted){
    //std::cout << "step-4: sorting." << std::endl;
    thrust::device_ptr<iT> d_bin_row_ptr 
                            = thrust::device_pointer_cast(d_bin_row);
    thrust::device_ptr<vT> d_bin_val_ptr 
                            = thrust::device_pointer_cast(d_bin_val);
    
    thrust::device_ptr<iT> d_y_inx_ptr = thrust::device_pointer_cast(d_y_key);
    thrust::device_ptr<vT> d_y_val_ptr = thrust::device_pointer_cast(d_y_val);

#ifdef STEP_PERF
    SpmspvTimer step4_timer;
    step4_timer.Start();
#endif

#ifndef USE_CUB_RADIX_SORT
    //TODO: test this branch.
    thrust::sort_by_key(thrust::device, d_bin_row_ptr, 
                        d_bin_row_ptr + bin_len, d_bin_val_ptr);
#else
    //TODO: DeviceRadixSort::SortPairs is out-of-place api!!
    iT* d_bin_row_temp = NULL;
    vT* d_bin_val_temp = NULL;
    cudaErrCheck(cudaMalloc((void **)&d_bin_row_temp,  bin_len * sizeof(iT)));
    cudaErrCheck(cudaMalloc((void **)&d_bin_val_temp,  bin_len * sizeof(vT)));

    int endbit = sizeof(int)*8;
    int DO_END_BIT_OPT = 0;
    if (DO_END_BIT_OPT)
      endbit = min(endbit, (int)log2((float)m)+1);

    size_t  temp_storage_bytes = 0;
    void* d_temp_storage = NULL;
    CubDebugExit( DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, 
                    (iT*)d_bin_row, (iT*)d_bin_row_temp, (vT*)d_bin_val, 
                    (vT*)d_bin_val_temp, bin_len, 0, endbit) );

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, 
                temp_storage_bytes));

    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                  (iT*)d_bin_row, (iT*)d_bin_row_temp, (vT*)d_bin_val, 
                    (vT*)d_bin_val_temp, bin_len, 0, endbit) );

    if(d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
#endif

    //cudaDeviceSynchronize();
#ifdef STEP_PERF
    step4_time = step4_timer.Stop();
    std::cout << "step-4 sort by key time = " << step4_time << " ms." << std::endl;
    //PrintBucket2FileDevice<iT, uiT, vT>(bin_len, d_bin_row_temp, d_bin_val_temp, "sort.info");
#endif

#if 1
    cudaDeviceSynchronize();
    cudaError_t err_r = cudaGetLastError();
    if (cudaSuccess != err_r) {
      printf("sortKernel() invocate error.\n");
      std::cout << "err code: " << cudaGetErrorString(err_r) << std::endl;
      exit(1);
    }
#endif

#ifdef STEP_PERF
    SpmspvTimer step5_timer;
    step5_timer.Start();
#endif

#ifndef MODERNGPU_REDUCE_BY_KEY 
    //test this branch: ok.
    //thrust::pair<iT*,vT*> new_end;
    thrust::pair< thrust::device_ptr<iT>, thrust::device_ptr<vT> > new_end;

  #ifdef USE_CUB_RADIX_SORT
    thrust::device_ptr<iT> d_bin_row_temp_ptr 
                            = thrust::device_pointer_cast(d_bin_row_temp);
    thrust::device_ptr<vT> d_bin_val_temp_ptr 
                            = thrust::device_pointer_cast(d_bin_val_temp);
    new_end = thrust::reduce_by_key(thrust::device, d_bin_row_temp_ptr, d_bin_row_temp_ptr + bin_len, 
                                    d_bin_val_temp_ptr, d_y_inx_ptr, d_y_val_ptr);
  #else
    new_end = thrust::reduce_by_key(thrust::device, d_bin_row_ptr, d_bin_row_ptr + bin_len, d_bin_val_ptr, 
                                    d_y_inx_ptr, d_y_val_ptr);
  #endif

    *y_nnz = new_end.first - d_y_inx_ptr;
    //std::cout << "inner y_nnz (cub-reduce-by-key) = " << *y_nnz << std::endl;

#else
    //test this branch:
    int  y_nvals_t = 0;
    //mgpu::ContextPtr d_context(mgpu::CreateCudaDevice(0));
    mgpu::ContextPtr d_context;
    d_context = mgpu::CreateCudaDevice(0);
  #ifdef USE_CUB_RADIX_SORT
    ReduceByKey( (iT*)d_bin_row_temp, (vT*)d_bin_val_temp, bin_len,
          (vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_key, d_y_val,
          &y_nvals_t, (int*)0, *d_context);
  #else
    ReduceByKey( (iT*)d_bin_row, (vT*)d_bin_val, bin_len,
          (vT)0, mgpu::plus<vT>(), mgpu::equal_to<vT>(), d_y_key, d_y_val,
          &y_nvals_t, (int*)0, *d_context);
  #endif
    
    *y_nnz = y_nvals_t;
    //std::cout << "inner y_nnz (mgpu-reduce-by-key) = " << *y_nnz << std::endl;
#endif 

#ifdef STEP_PERF 
    step5_time = step5_timer.Stop();
    std::cout << "step-5 reduce-by-key time = " << step5_time 
              << " ms." << std::endl;
#endif

  #ifdef USE_CUB_RADIX_SORT
    if (d_bin_row_temp)
      cudaErrCheck(cudaFree(d_bin_row_temp));
    if (d_bin_val_temp)
      cudaErrCheck(cudaFree(d_bin_val_temp));
  #endif

  }

#ifdef STEP_PERF
  double all_time = 0.0;
  if(isSorted){
    all_time = step1_time + step2_time + step3_time + step4_time + step5_time;
  }else{
    all_time = step1_time + step2_time + step3_time; 
  }
  std::cout << "all time = " << all_time << std::endl;
#endif

#endif


  return err;
}
#endif //SORT_BASED_METHOD_H_
