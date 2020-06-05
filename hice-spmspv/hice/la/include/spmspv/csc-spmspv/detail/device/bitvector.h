// This is the bitvector operation for csr based spmspv.
#ifndef BITVECTOR_H_
#define BITVECTOR_H_


void __host__ __device__ clear_bitvector(unsigned int idx, int* bitvec) {
    unsigned int neighbor_id = idx;
    int dword = (neighbor_id >> 5);
    int bit = neighbor_id  & 0x1F;
    unsigned int current_value = bitvec[dword];
    if ( (current_value & (1<<bit)))
    {
      bitvec[dword] = current_value ^ (1<<bit);//
    }
}

//count num of 1 in bit_vector.
//  void get_nnz(int num_ints, int* bit_vector, int* nnz) {
//    int len = 0;
//    for (int i = 0 ; i < num_ints ; i++) {
//      int p = _popcnt32(bit_vector[i]);
//      //if (p != 0)
//      // std::cout << "i =  " << i << ", p = " << p << std::endl;
//      len += p;
//    }
//    *nnz = len;
//  }


bool __host__ __device__ get_bitvector(unsigned int idx, const int* bitvec) {
    unsigned int neighbor_id = idx;
    int dword = (neighbor_id >> 5);
    int bit = neighbor_id  & 0x1F;
    unsigned int current_value = bitvec[dword];
    return ( (current_value & (1<<bit)) );//
}

void __device__ set_bitvector(unsigned int idx, int* bitvec) {
    unsigned int neighbor_id = idx;
    int dword = (neighbor_id >> 5);//
    int bit = neighbor_id  & 0x1F;//
    //printf("dword = %d, bit = %d.\n", dword, bit);
    // unsigned int current_value = bitvec[dword];
    // if ( (current_value & (1<<bit)) == 0)//
    // {
    //   bitvec[dword] = current_value | (1<<bit);
    // }
    atomicOr(&bitvec[dword], (1<<bit));
}


//bit_vector[]: ceil(n/32)
//num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);//
//memset(properties->bit_vector, 0, num_ints* sizeof(int));//
//TODO: 
__global__
void set_sparse_x_bitvector(int* d_sparse_x_inx, int x_nnz, 
                            int* d_bit_vector) {                        
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < x_nnz) {
    int idx = d_sparse_x_inx[tid];
    set_bitvector(idx, d_bit_vector);
  }
  
  // for (int tid = 0; tid < x_nnz; tid++) {
  //   int idx = d_sparse_x_inx[tid];
  //   //printf("col_id = %d\n", idx);
  //   set_bitvector(idx, d_bit_vector);
  // }
}

__global__
void clear_sparse_x_bitvector(int* d_sparse_x_inx, int x_nnz, 
                              int* d_bit_vector) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < x_nnz) {
    int idx = d_sparse_x_inx[tid];
    clear_bitvector(idx, d_bit_vector);
  }
}

#endif  // BITVECTOR_H_
