#pragma once

#include <cuda.h>

template<typename T>
__host__ __device__ __forceinline__ T hola_divup(T a, T b)
{
	return (a + b - 1) / b;
}
