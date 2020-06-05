#ifndef SORT_LOAD_BALANCE_METHOD_H_
#define SORT_LOAD_BALANCE_METHOD_H_

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
class my_counting_iterator : public std::iterator_traits<const T*> {
public:
  __host__ __device__ inline my_counting_iterator(T value) : _value(value) { }

   __host__ __device__ inline T operator[](ptrdiff_t i) { 
    return _value + i;
  }
   __host__ __device__ inline T operator*() {
    return _value;
  }
   __host__ __device__ inline my_counting_iterator operator+(ptrdiff_t diff) {
    return my_counting_iterator(_value + diff);
  }
   __host__ __device__ inline my_counting_iterator operator-(ptrdiff_t diff) {
    return my_counting_iterator(_value - diff);
  }
   __host__ __device__ inline my_counting_iterator& operator+=(ptrdiff_t diff) {
    _value += diff;
    return *this;
  }
   __host__ __device__ inline my_counting_iterator& operator-=(ptrdiff_t diff) {
    _value -= diff;
    return *this;
  }
private:
  T _value;
};


__host__ __device__ inline bool my_compareLess(int a, int b) {
  return a < b;
}
//////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline int4 my_ComputeMergeRange(
          int bin_len, int x_nnz, int block, 
          int NV, const int* mp_global) {

  // Load the merge paths computed by the partitioning kernel.
  int mp0 = mp_global[block];
  int mp1 = mp_global[block + 1];
  int gid = NV * block;//

  // Compute the ranges of the sources in global memory.
  // coop = false.
  int4 range;

  range.x = mp0;                      // a0
  range.y = mp1;                      // a1
  range.z = gid - range.x;            // b0
  range.w = min(bin_len + x_nnz, gid + NV) - range.y;  // b1

  return range;
}

// int mp = MergePath(array + a0, bin_len, d_scan + b0, x_nnz,
//      min(gid, bin_len + x_nnz), comp);
// template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
// MGPU_HOST_DEVICE int MergePath(It1 a, int aCount, It2 b, int bCount, int diag, Comp comp) {
__host__ __device__ inline int my_MergePath(
        my_counting_iterator<int> array, int bin_len, 
        int* d_scan, int x_nnz, int diag) {
  //typedef typename std::iterator_traits<int* >::value_type T;
 
  int begin = max(0, diag - x_nnz);
  int end = min(diag, bin_len);

  while (begin < end) {
    int mid = (begin + end)>> 1;//div 2.
    int aKey = array[mid];
    int bKey = d_scan[diag - 1 - mid];// mid + mid + 1 == diag.
    bool pred = my_compareLess(aKey, bKey);
    if(pred) begin = mid + 1; 
    else end = mid;
  }
  return begin;
}

 //<7, false>
 // DeviceSerialLoadBalanceSearch<VT>(b_shared, a0tid, a1, b0 - 1,
 //    b0tid, bCount, a_shared - a0);
template<int VT>
__device__ inline void my_DeviceSerialLoadBalanceSearch(
                    const int* b_shared, int aBegin,
                    int aEnd, int bFirst, 
                    int bBegin, int bEnd, 
                    int* a_shared) {

  int bKey = b_shared[bBegin];

  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    bool p;
    p = aBegin < bKey;
    if (p)
      a_shared[aBegin++] = bFirst + bBegin;
    else
      // Advance B (the haystack).
      bKey = b_shared[++bBegin];//
  }
}


template<int NT, int VT>
__device__ inline int4 my_CTALoadBalance(int bin_len, int* d_scan, 
  int x_nnz, int block, int tid, const int* mp_global, 
  int* indices_shared, bool loadPrecedingB) {
  //get each block's initial digonal.
  int4 range = my_ComputeMergeRange(bin_len, x_nnz, block, 
                                 NT * VT, mp_global);
 
  int a0 = range.x;
  int a1 = range.y;
  int b0 = range.z;
  int b1 = range.w;
  if(!b0) loadPrecedingB = false;

  // Load one trailing term from B. If we're already at the end, fill the 
  // end of the buffer with destCount.
  int aCount = a1 - a0;//
  int bCount = b1 - b0;//
   
  int extended = b1 < x_nnz;//b1 < x_nnz.
  int loadCount = bCount + extended;
  int fillCount = NT * VT + 1 - loadCount - aCount;

 
  int* a_shared = indices_shared;
  int* b_shared = indices_shared + aCount + (int)loadPrecedingB;

  // Load the B values.
  //  DeviceMemToMemLoop<NT>(bCount + extended + (int)loadPrecedingB, 
  //    b_global + b0 - (int)loadPrecedingB, tid, 
  //    b_shared - (int)loadPrecedingB);
 
  for (int i = tid - (int)loadPrecedingB; i < bCount + extended; i += NT)
    b_shared[i] = d_scan[b0 + i];

  // Fill the end of the array with destCount.
  for (int i = tid + extended; i < fillCount; i += NT)
    b_shared[bCount + i] = bin_len;
  __syncthreads();

  int diag = VT * tid;
  
  int mp = my_MergePath(my_counting_iterator<int>(a0),
    aCount, b_shared, bCount, diag);

  int a0tid = a0 + mp;
  int b0tid = diag - mp;//
  
  // Subtract 1 from b0 because we want to return upper_bound - 1.
  my_DeviceSerialLoadBalanceSearch<VT>(b_shared, a0tid, a1, b0 - 1,
    b0tid, bCount, a_shared - a0);
  //a_shared - a0 + a0 + mp =>

  __syncthreads();
  
  b0 -= (int)loadPrecedingB;
  return make_int4(a0, a1, b0, b1);
}


//VT is 4.
template<int NT, int VT>
__device__ inline void my_DeviceMemToMem4Indirect(int x_nnz, int* csc_col, 
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
        //x[i] = csc_col[x_sparse_index[NT * i + tid]];
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
__device__ inline void my_DeviceMemToMemLoopIndirect(int x_nnz, int* csc_col, 
                                int* x_sparse_index, int tid, int* dest) {
  for (int i = 0; i < x_nnz; i += 4 * NT)
    my_DeviceMemToMem4Indirect<NT, 4>(x_nnz - i, csc_col, x_sparse_index + i, 
                                   tid, dest + i);
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Functions to copy between shared and global memory where the average case is
// to transfer NT * VT elements.
template<int NT, int VT, typename T, typename OutputIt>
__device__ inline void my_DeviceSharedToGlobal(
      int count, const T* source, int tid, 
      OutputIt dest, bool sync) {
  typedef typename std::iterator_traits<int* >::value_type T2;
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    if (index < count) dest[index] = (T2)source[index];
  }
  if (sync) __syncthreads();
}


template<int NT, int VT, typename Type>
__device__ inline void my_DeviceGather(int bin_len, Type* csc_val, 
                                    int indices[VT], 
                                    int tid, Type* reg) {
  if (bin_len >= NT * VT) {
    #pragma unroll
    for (int i = 0; i < VT; ++i)
      reg[i] = csc_val[indices[i]];
  } else {
    #pragma unroll
    for (int i = 0; i < VT; ++i) {
      int index = NT * i + tid;
      if (index < bin_len)
        reg[i] = csc_val[indices[i]];
    }
  }
  //__syncthreads();
}

template<int NT, int VT, typename Type>
__device__ inline void my_DeviceRegToGlobal(int bin_len, const Type* reg, 
                                         int tid, Type* bin_row) {
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    if (index < bin_len)
      bin_row[index] = reg[i];
  }
  //__syncthreads();
}

template<typename Type>
__global__ void my_KernelIntervalMoveIndirect(
    int bin_len, int* csc_col, int* d_scan, 
    int x_nnz, Type* csc_val, int* x_sparse_index, 
    const int* mp_global, Type* bin_row) {

  const int NT = 128;
  const int VT = 7;

  //nt=128, vt=7，1024 bytes
  __shared__ int indices_shared[NT * (VT + 1)];
  int tid = threadIdx.x;//
  int block = blockIdx.x;//

  int4 range = my_CTALoadBalance<NT, VT>(bin_len, d_scan, 
    x_nnz, block, tid, mp_global, indices_shared, true);
 
}

template<int NT>
__global__ void my_KernelMergePartition(my_counting_iterator<int> array, 
  int bin_len, int* d_scan, 
  int x_nnz, int nv, int* mp_global, int numSearches) {
  
  int partition = NT * blockIdx.x + threadIdx.x;
  if (partition < numSearches) {
    int a0 = 0, b0 = 0;
    //gid = 128*8 * partition.
    int gid = nv * partition;

    int mp = my_MergePath(array + a0, bin_len, d_scan + b0, x_nnz,
      min(gid, bin_len + x_nnz));
    mp_global[partition] = mp;
  }
}


int* my_MergePathPartitions(my_counting_iterator<int> array, 
  int bin_len, int* d_scan,
  int x_nnz, int nv) {

  const int NT = 64;
  ////numofBlocks
  int numPartitions = SPMSPV_DIV_UP(bin_len + x_nnz, nv);
  ////(numPartitions + 1)/64
  int numPartitionBlocks = SPMSPV_DIV_UP(numPartitions + 1, NT);//
  //TODO: 

  //SpmspvTimer step1_timer;
  //step1_timer.Start();
  //MGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions + 1);
  int* partitionsDevice = NULL;
  cudaErrCheck(cudaMalloc((void** )&partitionsDevice, 
                          numPartitions + 1 * sizeof(int)));

  my_KernelMergePartition<NT><<<numPartitionBlocks, NT>>>(array, bin_len,
    d_scan, x_nnz, nv, partitionsDevice, numPartitions + 1);
  MGPU_SYNC_CHECK("KernelMergePartition");

  return partitionsDevice;
}

template<typename Type>
__host__  inline void my_IntervalGatherIndirect(int bin_len, int* csc_col, 
  int* d_scan, int x_nnz, Type* csc_val,
  int* x_sparse_index, Type* bin_row) {

  const int NT = 128;
  const int VT = 7;
  
  //NV = NT * VT = 128 * 7 = 896
  int NV = NT * VT;
  int numBlocks = SPMSPV_DIV_UP(bin_len + x_nnz, NV);
  
  printf("bin_len = %d, x_nnz = %d.\n", bin_len, x_nnz);
  printf("NT = %d, VT=%d, NV=%d, numBlocks=%d\n", NT, VT, NV, numBlocks);
  
  SpmspvTimer first_timer;
  first_timer.Start();
  int* partitionsDevice = my_MergePathPartitions(
    my_counting_iterator<int>(0), bin_len, d_scan,
    x_nnz, NV);
  double first_time = first_timer.Stop();
  std::cout << "blockPartition time = " << first_time << " ms." << std::endl;

  SpmspvTimer second_timer;
  second_timer.Start();
  my_KernelIntervalMoveIndirect<Type><<<numBlocks, NT>>>
    (bin_len, csc_col,
    d_scan, x_nnz, csc_val, x_sparse_index,
    partitionsDevice, bin_row);
  MGPU_SYNC_CHECK("KernelIntervalMove");
  double second_time = second_timer.Stop();
  std::cout << "inner time = " << second_time << " ms." << std::endl;
}
                   
template<typename iT, typename uiT, typename vT>
int LoadBalanceExtractDriver(
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

  SpmspvTimer first_timer;
  first_timer.Start();
  my_IntervalGatherIndirect(bin_len, d_csc_col, d_ptr_col_len, x_nnz,  
                         d_csc_row, d_sparse_x_inx, d_bin_row);
  double first_time = first_timer.Stop();
  std::cout << "first time = " << first_time << " ms." << std::endl;
  SpmspvTimer second_timer;
  second_timer.Start();
  my_IntervalGatherIndirect(bin_len, d_csc_col, d_ptr_col_len, x_nnz,  
                         d_csc_val, d_sparse_x_inx, d_bin_val);
  double second_time = second_timer.Stop();
  std::cout << "second time = " << second_time << " ms." << std::endl;
  return err;
}
#endif 
