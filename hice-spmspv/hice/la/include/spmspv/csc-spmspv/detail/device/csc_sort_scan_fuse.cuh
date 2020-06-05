#ifndef SORT_SCAN_FUSE_H_
#define SORT_SCAN_FUSE_H_

#include "spmspv/scan_config.h"
#include "scan_kernel.cuh"

//template<typename iT, typename uiT, typename vT>
int ScanFuseDriver(int* d_csc_col,  
						       /*iT x_nnz, */
                   int elemnum, 
						       int* d_sparse_x_inx,
						       int* d_ptr_col_len,
                   int* gadsys, int adsyslen,
                   int workgroup, int taillen, int tail_group) {
 
	cudaErrCheck(cudaMemset(gadsys, 0, adsyslen * sizeof(int)));

 if (workgroup != 0) {
  switch(LT){
    case 64:
      switch(CT){
      case 8:
        scanFuseKernel<64, 6, 32, 63, 65, 8, 8, 3, 4, 3, 5, 7, 3, 2><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      case 16:
        scanFuseKernel<64, 6, 32, 63, 65, 4, 16, 4, 8, 7, 17, 15, 4, 3 ><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      case 32:
        scanFuseKernel<64, 6, 32, 63, 65, 2, 32, 5, 16, 15, 33, 31, 5, 4 ><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      case 64:
        scanFuseKernel<64, 6, 32, 63, 65, 1, 64, 6, 32, 31, 65, 63, 6, 5 ><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      }
      break;
    case 128:
      switch(CT){
      case 8:
        scanFuseKernel<128, 7, 64, 127, 129, 16, 8, 3, 4, 3, 5, 7, 3, 2><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      case 16:
        scanFuseKernel<128, 7, 64, 127, 129, 8, 16, 4, 8, 7, 17, 15, 4, 3><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      case 32:
        scanFuseKernel<128, 7, 64, 127, 129, 4, 32, 5, 16, 15, 33, 31, 5, 4><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      case 64:
        scanFuseKernel<128, 7, 64, 127, 129, 2, 64, 6, 32, 31, 65, 63, 6, 5><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
        break;
      }
      break;
    case 256:
      switch(CT){
        case 8:
          scanFuseKernel<256, 8, 128, 255, 257, 32, 8, 3, 4, 3, 5, 7, 3, 2><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
          break;
        case 16:
          scanFuseKernel<256, 8, 128, 255, 257, 16, 16, 4, 8, 7, 17, 15, 4, 3><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
          break;
        case 32:
          scanFuseKernel<256, 8, 128, 255, 257, 8, 32, 5, 16, 15, 33, 31, 5, 4><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
          break;
        case 64:
          scanFuseKernel<256, 8, 128, 255, 257, 4, 64, 6, 32, 31, 65, 63, 6, 5><<<workgroup, LT>>>(d_csc_col, elemnum, d_sparse_x_inx, d_ptr_col_len, gadsys, d_ptr_col_len, workgroup);
          break;
      }
      break;
  }
//#endif
 }

 if (taillen != 0) {
  scanFuseTail<64, 6, 32, 63, 65, 4, 16, 4, 8, 7, 17, 15, 4, 3>
          <<<tail_group, 64>>>(d_csc_col, elemnum, d_sparse_x_inx, 
  														 d_ptr_col_len, gadsys, d_ptr_col_len, 
    													 elemnum, workgroup, tail_group, taillen);
//#endif
 }
  return 0;
}

#endif 