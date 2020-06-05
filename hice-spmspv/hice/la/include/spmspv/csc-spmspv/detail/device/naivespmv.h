
#include <stdint.h>
#include "bitvector.h"

template<typename T>
inline __device__ T myLoad(const T* d)
{
	return *d;
	//return __ldg(d);
}

  template<typename ValueType, typename IndexType, typename OffsetType>
__global__ void naive_spmspv_ker(int num_non_zeroes, int out_size, int num_other, 
	const ValueType* __restrict matrix, const IndexType* __restrict inIndex, const OffsetType* __restrict offsets, 
	const ValueType* __restrict inVec, ValueType* __restrict outVec, const int* d_bit_vector)
{
	//uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= out_size)
		return;

	ValueType sum = 0;
	for (OffsetType j = myLoad(offsets + i); j < myLoad(offsets + i + 1); ++j)
	{
		IndexType ind = myLoad(inIndex + j);
		if(get_bitvector(ind, d_bit_vector)) {
		  sum += myLoad(inVec + ind) * myLoad(matrix + j);
    }
	}
	outVec[i] = sum;
}

template<typename ValueType, typename IndexType, typename OffsetType>
__global__ void naive_spmv_ker(int num_non_zeroes, int out_size, int num_other, 
	const ValueType* __restrict matrix, const IndexType* __restrict inIndex, const OffsetType* __restrict offsets, 
	const ValueType* __restrict inVec, ValueType* __restrict outVec)
{
	//uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= out_size)
		return;

	ValueType sum = 0;
	for (OffsetType j = myLoad(offsets + i); j < myLoad(offsets + i + 1); ++j)
	{
		IndexType ind = myLoad(inIndex + j);
		sum += myLoad(inVec + ind) * myLoad(matrix + j);
	}
	outVec[i] = sum;
}

template<typename T>
void naive_spmspv(int m, int n, int nnz, int* csr_row, int* csr_col, T* csr_val, T* v_data, T* res, const int* d_bit_vector)
{
	uint32_t blockSize = 256;
	naive_spmspv_ker<T, int, int> <<<(m + blockSize - 1) / blockSize, blockSize >>>(nnz, m, n, csr_val, csr_col, csr_row, v_data, res, d_bit_vector);
}


template<typename T>
void naive_spmv(int m, int n, int nnz, int* csr_row, int* csr_col, T* csr_val, T* v_data, T* res)
{
	uint32_t blockSize = 256;
	naive_spmv_ker<T, int, int> <<<(m + blockSize - 1) / blockSize, blockSize >>>(nnz, m, n, csr_val, csr_col, csr_row, v_data, res);
}

template void naive_spmv(int m, int n, int nnz, int* csr_row, int* csr_col, float* csr_val, float* v_data, float* res);
template void naive_spmv(int m, int n, int nnz, int* csr_row, int* csr_col, double* csr_val, double* v_data, double* res);

template void naive_spmspv(int m, int n, int nnz, int* csr_row, int* csr_col, float* csr_val, float* v_data, float* res, const int* d_bit_vector);
template void naive_spmspv(int m, int n, int nnz, int* csr_row, int* csr_col, double* csr_val, double* v_data, double* res, const int* d_bit_vector);
